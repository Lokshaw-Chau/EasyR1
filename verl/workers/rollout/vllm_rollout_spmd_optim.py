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
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.tokenizer import get_processor
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
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
        # Store tokenizer and tool-call injection settings
        self.tokenizer = tokenizer
        self.tool_call_token_id = 151657  # <tool_call>
        self.tool_injection_text = '\n{"name": "gui_action", "arguments": {"action": "click", "coordinate": '
        # Pre-tokenize the injection prefix to append after <tool_call>
        self.tool_injection_prefix_ids = self.tokenizer.encode(
            self.tool_injection_text, add_special_tokens=False
        )
        self.rollout_intervention = config.rollout_intervention
        self.intervention_no_think_n = config.intervention_no_think_n
        self.intervention_think_n = config.intervention_think_n
        if self.intervention_no_think_n + self.intervention_think_n > self.config.n:
            raise ValueError(
                f"intervention_no_think_n + intervention_think_n should be less than or equal to n, "
                f"but got {self.intervention_no_think_n} + {self.intervention_think_n} > {self.config.n}."
            )
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

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
            limit_mm_per_prompt={"image": config.limit_images} if config.limit_images > 0 else None,
            disable_mm_preprocessor_cache=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
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
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            response_ids: List[List[int]] = []
            response_base_prompts: List[List[int]]  # prompt_token_ids actually fed to vLLM for each response
            already_injected: List[bool]  # whether the GUI JSON prefix has been pre-injected for this response

            if not self.rollout_intervention or self.sampling_params.n == 1:
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
                )
                # Align each sample with its base prompt
                for i, completion in enumerate(completions):
                    base_prompt_ids = vllm_inputs[i]["prompt_token_ids"]
                    for output in completion.outputs:
                        response_ids.append(output.token_ids)
                        response_base_prompts.append(base_prompt_ids)
                        already_injected.append(False)
            else:
                no_intervention_n = self.sampling_params.n - self.intervention_no_think_n - self.intervention_think_n
                if no_intervention_n > 0:
                    sampling_params_nointervention = deepcopy(self.sampling_params)
                    sampling_params_nointervention.n = no_intervention_n
                    vllm_inputs_nointervention = deepcopy(vllm_inputs)
                    completions_nointervention = self.inference_engine.generate(
                        vllm_inputs_nointervention,
                        sampling_params=sampling_params_nointervention,
                        use_tqdm=(self.rank == 0),
                    )
                else:
                    completions_nointervention = [[] for _ in range(len(vllm_inputs))]

                if self.intervention_no_think_n > 0:
                    sampling_params_nothinking = deepcopy(self.sampling_params)
                    sampling_params_nothinking.n = self.intervention_no_think_n
                    # enforce the first token to be "<tool_call>", then inject the GUI action prefix
                    injection_len = len(self.tool_injection_prefix_ids)
                    sampling_params_nothinking.max_tokens = max(
                        0, self.sampling_params.max_tokens - 1 - injection_len
                    )
                    vllm_inputs_nothinking = deepcopy(vllm_inputs)
                    for ipt in vllm_inputs_nothinking:
                        ipt["prompt_token_ids"] = (
                            ipt["prompt_token_ids"] + [self.tool_call_token_id] + self.tool_injection_prefix_ids
                        )
                    completions_nothinking = self.inference_engine.generate(
                        prompts=vllm_inputs_nothinking,
                        sampling_params=sampling_params_nothinking,
                        use_tqdm=(self.rank == 0),
                    )
                else:
                    completions_nothinking = [[] for _ in range(len(vllm_inputs))]

                if self.intervention_think_n > 0:
                    sampling_params_thinking = deepcopy(self.sampling_params)
                    sampling_params_thinking.n = self.intervention_think_n
                    # enforce the first token to be "<think>"
                    sampling_params_thinking.max_tokens = self.sampling_params.max_tokens - 1
                    vllm_inputs_thinking = deepcopy(vllm_inputs)
                    for ipt in vllm_inputs_thinking:
                        ipt["prompt_token_ids"] = ipt["prompt_token_ids"] + [13708]
                    completions_thinking = self.inference_engine.generate(
                        prompts=vllm_inputs_thinking,
                        sampling_params=sampling_params_thinking,
                        use_tqdm=(self.rank == 0),
                    )
                else:
                    completions_thinking = [[] for _ in range(len(vllm_inputs))]

                assert (
                    len(completions_nothinking) == len(completions_thinking) == len(completions_nointervention)
                ), f"{len(completions_nothinking)} != {len(completions_thinking)}"
                for i, (
                    completion_nointervention,
                    completion_nothinking,
                    completion_thinking,
                ) in enumerate(zip(completions_nointervention, completions_nothinking, completions_thinking)):
                    # thinking branch
                    if completion_thinking != []:
                        base_prompt_ids = vllm_inputs_thinking[i]["prompt_token_ids"]
                        for sample_id in range(len(completion_thinking.outputs)):
                            # prepend <think> just as a visible marker in responses
                            response_ids.append([13708] + completion_thinking.outputs[sample_id].token_ids)
                            response_base_prompts.append(base_prompt_ids)
                            already_injected.append(False)
                    else:
                        print("No thinking output!")
                    # nothinking branch (already injected in prompt)
                    if completion_nothinking != []:
                        base_prompt_ids = vllm_inputs_nothinking[i]["prompt_token_ids"]
                        for sample_id in range(len(completion_nothinking.outputs)):
                            response_ids.append(
                                [self.tool_call_token_id]
                                + self.tool_injection_prefix_ids
                                + completion_nothinking.outputs[sample_id].token_ids
                            )
                            response_base_prompts.append(base_prompt_ids)
                            already_injected.append(True)  # already injected via prompt
                    else:
                        print("No not thinking output!")
                    # no-intervention branch
                    if completion_nointervention != []:
                        base_prompt_ids = vllm_inputs_nointervention[i]["prompt_token_ids"]
                        for sample_id in range(len(completion_nointervention.outputs)):
                            response_ids.append(completion_nointervention.outputs[sample_id].token_ids)
                            response_base_prompts.append(base_prompt_ids)
                            already_injected.append(False)
                    else:
                        print("No no intervention output!")

            # Post-process: if any response contains <tool_call> token, stop there, inject JSON prefix, and continue
            indices_needing_continue: List[int] = []
            cut_positions: List[int] = []
            for idx, (resp, injected) in enumerate(zip(response_ids, already_injected)):
                if injected:
                    continue
                try:
                    pos = resp.index(self.tool_call_token_id)
                except ValueError:
                    continue
                indices_needing_continue.append(idx)
                cut_positions.append(pos)

            # For each item needing continuation, run a follow-up generation from the cut point with the injected prefix
            for local_idx, resp_idx in enumerate(indices_needing_continue):
                pos = cut_positions[local_idx]
                base_prompt = response_base_prompts[resp_idx]
                prefix_generated = response_ids[resp_idx][: pos + 1]
                # new prompt: base prompt + generated up to <tool_call> + injected JSON prefix
                new_prompt_ids = base_prompt + prefix_generated + self.tool_injection_prefix_ids
                remaining = max(
                    0, self.sampling_params.max_tokens - (pos + 1) - len(self.tool_injection_prefix_ids)
                )
                follow_params = deepcopy(self.sampling_params)
                follow_params.n = 1
                follow_params.max_tokens = remaining
                follow_out = self.inference_engine.generate(
                    prompts=[{"prompt_token_ids": new_prompt_ids}],
                    sampling_params=follow_params,
                    use_tqdm=False,
                )
                # Replace the response with prefix + injected + new tokens
                if len(follow_out) > 0 and len(follow_out[0].outputs) > 0:
                    new_tokens = follow_out[0].outputs[0].token_ids
                else:
                    new_tokens = []
                response_ids[resp_idx] = prefix_generated + self.tool_injection_prefix_ids + new_tokens

            # finalize tensors
            enforce_nothinking = []
            for response_id in response_ids:
                if len(response_id) > 0:
                    if response_id[0] == 151657:  # <tool_call>
                        enforce_nothinking.append(True)
                    elif response_id[0] == 13708:  # <thinking>
                        enforce_nothinking.append(False)
                    else:
                        enforce_nothinking.append(False)
                        print(
                            f"Unexpected first token {response_id[0]} in response_ids, defaulting to thinking mode."
                        )
                else:
                    print("Empty response_ids, defaulting to thinking mode.")
                    enforce_nothinking.append(False)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            enforce_nothinking = torch.tensor(enforce_nothinking).to(input_ids.device)
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
        
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "enforce_nothinking": enforce_nothinking,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)