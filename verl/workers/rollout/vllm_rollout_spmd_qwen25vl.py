# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, list]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    # Support both "image"/"images" and "video"/"videos" keys
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))
    elif "image" in multi_modal_data:
        # Handle singular "image" key from dataset
        image_data = multi_modal_data["image"]
        if isinstance(image_data, list):
            for image in image_data:
                images.append(process_image(image, min_pixels, max_pixels))
        else:
            images.append(process_image(image_data, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))
    elif "video" in multi_modal_data:
        # Handle singular "video" key
        video_data = multi_modal_data["video"]
        if isinstance(video_data, list):
            for video in video_data:
                videos.append(process_video(video, min_pixels, max_pixels, video_fps))
        else:
            videos.append(process_video(video_data, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.rollout_intervention = getattr(config, 'rollout_intervention', 'none')
        self.intervention_no_think_n = getattr(config, 'intervention_no_think_n', 0)
        self.intervention_think_n = getattr(config, 'intervention_think_n', 0)

        if self.intervention_no_think_n + self.intervention_think_n > self.config.n:
            raise ValueError(
                f"intervention_no_think_n + intervention_think_n should be less than or equal to n, "
                f"but got {self.intervention_no_think_n} + {self.intervention_think_n} > {self.config.n}."
            )

        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        intervention_think_n = prompts.meta_info.pop("intervention_think_n", 0)
        intervention_nothink_n = prompts.meta_info.pop("intervention_nothink_n", 0)
        prompts.meta_info.pop("intervention_think_n", None)
        prompts.meta_info.pop("intervention_nothink_n", None)

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            if self.rollout_intervention == 'none' or self.sampling_params.n == 1:
                completions: list[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
                )
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
                rollout_prob = [1.0 / self.sampling_params.n] * len(response_ids)
            else:
                # Use intervention_think_n and intervention_nothink_n from meta_info
                self.intervention_no_think_n = intervention_nothink_n
                self.intervention_think_n = intervention_think_n

                print(f"Using rollout intervention with intervention_nothink_n: {self.intervention_no_think_n}, intervention_think_n: {self.intervention_think_n}")

                if self.intervention_no_think_n + self.intervention_think_n > self.sampling_params.n:
                    raise ValueError(
                        f"intervention_think_n + intervention_nothink_n should be less than or equal to n, "
                        f"but got {self.intervention_no_think_n} + {self.intervention_think_n} > {self.sampling_params.n}."
                    )

                no_intervention_n = self.sampling_params.n - self.intervention_no_think_n - self.intervention_think_n

                # Generate without intervention
                if no_intervention_n > 0:
                    sampling_params_nointervention = deepcopy(self.sampling_params)
                    sampling_params_nointervention.n = no_intervention_n
                    vllm_inputs_nointervention = deepcopy(vllm_inputs)
                    completions_nointervention = self.inference_engine.generate(
                        vllm_inputs_nointervention,
                        sampling_params=sampling_params_nointervention,
                        use_tqdm=self.use_tqdm
                    )
                else:
                    completions_nointervention = [[] for _ in range(len(vllm_inputs))]

                # Generate nothinking responses (force <tool_call> token)
                if self.intervention_no_think_n > 0:
                    sampling_params_nothinking = deepcopy(self.sampling_params)
                    sampling_params_nothinking.n = self.intervention_no_think_n
                    sampling_params_nothinking.max_tokens = self.sampling_params.max_tokens - 1
                    vllm_inputs_nothinking = deepcopy(vllm_inputs)
                    for ipt in vllm_inputs_nothinking:
                        ipt['prompt_token_ids'] = ipt['prompt_token_ids'] + [151657]  # <tool_call> token id
                    completions_nothinking = self.inference_engine.generate(
                        prompts=vllm_inputs_nothinking,
                        sampling_params=sampling_params_nothinking,
                        use_tqdm=self.use_tqdm
                    )
                else:
                    completions_nothinking = [[] for _ in range(len(vllm_inputs))]

                # Generate thinking responses (force <thinking> token)
                if self.intervention_think_n > 0:
                    sampling_params_thinking = deepcopy(self.sampling_params)
                    sampling_params_thinking.n = self.intervention_think_n
                    sampling_params_thinking.max_tokens = self.sampling_params.max_tokens - 1
                    vllm_inputs_thinking = deepcopy(vllm_inputs)
                    for ipt in vllm_inputs_thinking:
                        ipt['prompt_token_ids'] = ipt['prompt_token_ids'] + [13708]  # <thinking> token id
                    completions_thinking = self.inference_engine.generate(
                        prompts=vllm_inputs_thinking,
                        sampling_params=sampling_params_thinking,
                        use_tqdm=self.use_tqdm
                    )
                else:
                    completions_thinking = [[] for _ in range(len(vllm_inputs))]

                # Combine all completions and compute rollout probabilities
                response_ids = []
                rollout_prob = []

                for completion_nointervention, completion_nothinking, completion_thinking in zip(
                    completions_nointervention, completions_nothinking, completions_thinking
                ):
                    # Calculate no_think_ratio based on actual completions
                    if completion_nointervention != [] and completion_nothinking != []:
                        no_think_ratio = (
                            len(completion_nothinking.outputs) +
                            len([s for s in completion_nointervention.outputs if s.token_ids[0] == 151657])
                        ) / self.sampling_params.n
                    elif completion_nointervention != [] and completion_nothinking == []:
                        no_think_ratio = len([s for s in completion_nointervention.outputs if s.token_ids[0] == 151657]) / self.sampling_params.n
                    elif completion_nointervention == [] and completion_nothinking != []:
                        no_think_ratio = len(completion_nothinking.outputs) / self.sampling_params.n
                    else:
                        no_think_ratio = 0
                    think_ratio = 1 - no_think_ratio

                    # Add thinking outputs
                    if completion_thinking != []:
                        for output in completion_thinking.outputs:
                            response_ids.append([13708] + output.token_ids)
                            rollout_prob.append(think_ratio)

                    # Add nothinking outputs
                    if completion_nothinking != []:
                        for output in completion_nothinking.outputs:
                            response_ids.append([151657] + output.token_ids)
                            rollout_prob.append(no_think_ratio)

                    # Add no intervention outputs
                    if completion_nointervention != []:
                        for output in completion_nointervention.outputs:
                            response_ids.append(output.token_ids)
                            if output.token_ids[0] == 151657:
                                rollout_prob.append(no_think_ratio)
                            else:
                                rollout_prob.append(think_ratio)

            # Determine enforce_nothinking based on first token
            enforce_nothinking = []
            for response_id in response_ids:
                if len(response_id) > 0 and response_id[0] == 151657:  # <tool_call>
                    enforce_nothinking.append(True)
                elif len(response_id) > 0 and response_id[0] == 13708:  # <thinking>
                    enforce_nothinking.append(False)
                else:
                    enforce_nothinking.append(False)
                    if len(response_id) > 0:
                        print(f"Unexpected first token {response_id[0]} in response_ids, defaulting to thinking mode.")

            # Pad response_ids
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            enforce_nothinking = torch.tensor(enforce_nothinking, dtype=torch.bool).to(input_ids.device)
            rollout_prob = torch.tensor(rollout_prob, dtype=torch.float32).to(input_ids.device)

            # Repeat inputs if n > 1
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        prompt_input_ids = input_ids
        prompt_attention_mask = attention_mask
        prompt_position_ids = position_ids

        # Build real sequences
        sequence_ids = torch.cat([prompt_input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )

        attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "enforce_nothinking": enforce_nothinking,
                "rollout_prob": rollout_prob,
            },
            batch_size=batch_size,
        )
        # multi_modal_data is only needed during generation, not in the output
        non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
