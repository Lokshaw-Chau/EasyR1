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
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input
import numpy as np

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
import torch.distributed as dist
from ...utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad_and_slice_inputs,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    get_ulysses_sequence_parallel_rank,
)
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tool_call_token_id = getattr(self.config, "tool_call_token_id", 151657)

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = VF.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = VF.entropy_from_logits

        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
            self.entropy_from_logits = torch.compile(entropy_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
            self.entropy_from_logits = entropy_from_logits

        self.calculate_entropy = False
    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float, simko: bool = False, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            entropy: # (bs, response_len) or None
            log_probs: # (bs, response_len)
            max_token: # (bs, response_len) - if simko=True, else None
            topk_log_probs: # (bs, response_len, top_k) - if simko=True, else None
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        entropy = None
        max_token = None
        topk_log_probs = None
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            result = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            if len(result) == 3:
                input_ids_rmpad, indices, cu_seqlens = result
                max_seqlen = None
            elif len(result) == 4:
                input_ids_rmpad, indices, cu_seqlens, max_seqlen = result
            else:
                input_ids_rmpad, indices, cu_seqlens, *_ = result
                # raise ValueError(f"Unexpected number of return values: {len(result)}")

            cu_seqlens = cu_seqlens.to(dtype=torch.long, device=input_ids.device)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            inplace_backward = True
            if self.calculate_entropy or simko:
                inplace_backward = False
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

            # Compute max_token and topk_log_probs for SIMKO
            if simko:
                # Compute max_token (whether predicted token matches actual token)
                predicted_tokens_rmpad = torch.argmax(logits_rmpad, dim=-1)  # (chunk_len,)
                max_token_rmpad = (predicted_tokens_rmpad == input_ids_rmpad_rolled).float()  # (chunk_len,)

                topk_logp_rmpad = None
                if top_k > 0:
                    # Map first response logits to current shard to avoid materializing full vocab tensors
                    chunk_len = logits_rmpad.size(0)
                    if self.config.ulysses_sequence_parallel_size > 1:
                        sp_rank = get_ulysses_sequence_parallel_rank()
                    else:
                        sp_rank = 0
                    start_offset = sp_rank * chunk_len

                    cu_seqlens_device = cu_seqlens.to(device=logits_rmpad.device)
                    non_pad_lengths = cu_seqlens_device[1:] - cu_seqlens_device[:-1]
                    response_token_counts = attention_mask[:, -response_length:].sum(dim=1).to(dtype=torch.long)
                    first_token_offsets = non_pad_lengths - response_token_counts - 1
                    first_token_offsets = torch.clamp(first_token_offsets, min=0)
                    first_token_indices = cu_seqlens_device[:-1] + first_token_offsets

                    local_positions = first_token_indices - start_offset
                    valid_mask = (local_positions >= 0) & (local_positions < chunk_len)

                    topk_logp_rmpad = logits_rmpad.new_zeros((chunk_len, top_k))
                    if valid_mask.any():
                        valid_local_positions = local_positions[valid_mask].to(torch.long)
                        selected_logits = logits_rmpad[valid_local_positions]
                        topk_values, _ = torch.topk(selected_logits, k=top_k, dim=-1)
                        log_sum_exp = torch.logsumexp(selected_logits, dim=-1, keepdim=True)
                        topk_log_probs_valid = topk_values - log_sum_exp
                        topk_logp_rmpad.index_copy_(0, valid_local_positions, topk_log_probs_valid)

            if self.calculate_entropy:
                # compute the mode entropy (entropy of the first token)

                entropy_rmpad = torch.utils.checkpoint.checkpoint(
                    self.entropy_from_logits, logits_rmpad
                )

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                if self.calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                if simko:
                    max_token_rmpad = gather_outputs_and_unpad(
                        max_token_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if top_k > 0 and topk_logp_rmpad is not None:
                        topk_logp_rmpad = gather_outputs_and_unpad(
                            topk_logp_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
            if self.calculate_entropy:
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, 1)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            # Pad SIMKO tensors
            if simko:
                full_max_token = pad_input(
                    hidden_states=max_token_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                max_token = full_max_token.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if top_k > 0 and topk_logp_rmpad is not None:
                    full_topk_logp = pad_input(
                        hidden_states=topk_logp_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    topk_log_probs = full_topk_logp[:, -response_length - 1 : -1, :]  # (bsz, response_length, K)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

            entropy = None
            if self.calculate_entropy:
                entropy = VF.entropy_from_logits(logits)

            # Compute max_token and topk_log_probs for SIMKO
            if simko:
                # Compute max_token (whether predicted token matches actual token)
                predicted_tokens = torch.argmax(logits, dim=-1)  # (bsz, response_length)
                max_token = (predicted_tokens == responses).float()  # (bsz, response_length)

                # Compute top-K log probabilities ONLY for the first token (memory optimization)
                if top_k > 0:
                    B, T, V = logits.shape
                    # Only compute top-k for the first token to save memory
                    first_token_logits = logits[:, 0, :]  # (B, V)
                    topk_values, topk_idx = torch.topk(first_token_logits, k=top_k, dim=-1)  # (B, K)

                    # Compute log_softmax for top-k tokens
                    log_sum_exp = torch.logsumexp(first_token_logits, dim=-1, keepdim=True)  # (B, 1)
                    first_topk_logp = topk_values - log_sum_exp  # (B, K)

                    # Create full tensor with zeros for other positions
                    topk_log_probs = torch.zeros(B, T, top_k, device=logits.device, dtype=logits.dtype)
                    topk_log_probs[:, 0, :] = first_topk_logp  # Only fill first token

        return entropy, log_probs, max_token, topk_log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        entropy_lst = []
        max_token_lst = []
        topk_log_probs_lst = []

        # Check if SIMKO is enabled
        simko = self.config.simko
        top_k = self.config.top_k if simko else 0

        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            entropy, log_probs, max_token, topk_log_probs = self._forward_micro_batch(
                model_inputs, temperature=temperature, simko=simko, top_k=top_k
            )
            log_probs_lst.append(log_probs)
            if self.calculate_entropy:
                entropy_lst.append(entropy)
            if simko:
                max_token_lst.append(max_token)
                if top_k > 0:
                    topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        max_tokens = None
        topk_log_probs_out = None

        if self.calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if simko:
            max_tokens = torch.concat(max_token_lst, dim=0)
            if top_k > 0:
                topk_log_probs_out = torch.concat(topk_log_probs_lst, dim=0)

        # Return tuple: (log_probs, entropys, max_tokens, topk_log_probs)
        # For backward compatibility, components can be None if not computed
        return log_probs, entropys, max_tokens, topk_log_probs_out

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        training_process = data.meta_info["training_process"]

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "enforce_nothinking",
            "rollout_prob",
        ]
        if self.config.think_advantage_scaling or self.config.think_filtering:
            select_keys.append("advantage_scaling_factor")
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if self.config.enable_kl_no_think:
            select_keys.extend(["fake_input_ids", "fake_attention_mask", "fake_position_ids", "fake_responses"])

        # Add SIMKO-specific keys
        if self.config.simko:
            select_keys.extend(["old_log_probs_topk", "token_level_scores"])

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        # mini_batches['enforce_nothinking'] = torch.tensor(data.non_tensor_batch['enforce_nothinking']).bool().to(mini_batches['responses'].device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]

                    if self.config.old_rollout_probs:
                        rollout_prob = model_inputs['rollout_prob']
                        old_log_probs[:, 0] = torch.log(rollout_prob)
                    
                    advantages = model_inputs["advantages"]
                    enforce_nothinking = model_inputs['enforce_nothinking']

                    # all return: (bsz, response_length)
                    entropy, log_probs, max_token, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, simko=self.config.simko, top_k=self.config.top_k
                    )
                    force_think_resp_entropy = -VF.masked_mean(
                        log_probs[~enforce_nothinking, 1:], response_mask[~enforce_nothinking, 1:]
                    )
                    force_no_think_resp_entropy = -VF.masked_mean(
                        log_probs[enforce_nothinking, 1:], response_mask[enforce_nothinking, 1:]
                    )
                    first_eot_logprobs = log_probs[enforce_nothinking, 0]
                    first_eot_probs = first_eot_logprobs.exp()
                    first_t_logprobs = log_probs[~enforce_nothinking, 0]
                    first_t_probs = first_t_logprobs.exp()

                    # scaling_factor = self.config.clip_mode_scale_factor * (1 - 1 / (1 + np.exp(-self.config.sigmoid_k * (training_process - self.config.sigmoid_x0))))
                    focal_rho = self.config.focal_rho * (1 - training_process)
                    entropy_bonus = self.config.entropy_bonus_alpha * (1 - training_process)

                    # Use SIMKO policy loss if enabled, otherwise use standard policy loss
                    if self.config.simko:
                        old_log_probs_topk = model_inputs["old_log_probs_topk"]
                        token_level_scores = model_inputs["token_level_scores"]
                        entropy = entropy.detach()

                        pg_loss, pg_clipfrac_higher, ppo_kl, cond_loss, resp_loss, simko_metrics = core_algos.compute_policy_loss_simko(
                            old_log_prob=old_log_probs,
                            old_log_probs_topk=old_log_probs_topk,
                            log_prob=log_probs,
                            topk_log_probs=topk_log_probs,
                            entropy=entropy,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=self.config.clip_ratio_low,
                            token_level_scores=token_level_scores,
                            max_token=max_token,
                            mix_topk_coef=self.config.mix_topk_coef,
                            tau=self.config.tau,
                        )
                        # SIMKO doesn't return separate clipfrac metrics, so we duplicate
                        pg_clipfrac_lower = [torch.tensor(0.0), torch.tensor(0.0)]
                    else:
                        # Dynamic advantage scaling based on think_acc - nothink_acc difference
                        # This can be enabled independently of think_filtering via think_advantage_scaling config
                        scaling_metrics = None
                        if self.config.think_advantage_scaling: #  or self.config.think_filtering:
                            advantages = advantages.clone()

                            # Get scaling factor from batch data (if available)
                            scaling_factor = model_inputs.get("advantage_scaling_factor", None)

                            if scaling_factor is not None:
                                if not torch.is_tensor(scaling_factor):
                                    scaling_factor = torch.as_tensor(
                                        scaling_factor, dtype=advantages.dtype, device=advantages.device
                                    )
                                else:
                                    scaling_factor = scaling_factor.to(
                                        device=advantages.device, dtype=advantages.dtype
                                    )

                                scaling_factor = scaling_factor.view(-1)
                                enforce_mask = enforce_nothinking.to(device=advantages.device, dtype=torch.bool)
                                think_mask = ~enforce_mask

                                scaling_metrics = {
                                    "actor/advantage_scaling_factor_mean": 0.0,
                                    "actor/advantage_scaling_factor_max": 0.0,
                                    "actor/advantage_scaling_factor_min": 0.0,
                                }

                                if think_mask.any():
                                    think_scaling = torch.clamp(scaling_factor[think_mask], min=1e-4).detach()

                                    # Get responses to find <tool_call> token positions
                                    responses = model_inputs["responses"]
                                    tool_call_token_id = 151657

                                    # Apply scaling from second token (index 1) to <tool_call> token for each think sample
                                    think_indices = torch.where(think_mask)[0]
                                    for i, sample_idx in enumerate(think_indices):
                                        # Find the position of <tool_call> token in this sample
                                        tool_call_positions = (responses[sample_idx] == tool_call_token_id).nonzero(as_tuple=True)[0]

                                        if len(tool_call_positions) > 0:
                                            # Get the first occurrence of <tool_call> token
                                            tool_call_pos = tool_call_positions[0].item()
                                            # Apply scaling from token 1 to tool_call_pos
                                            advantages[sample_idx, 1:tool_call_pos] *= think_scaling[i]
                                        else:
                                            # If no <tool_call> token found, apply to all tokens from position 1 onwards (fallback)
                                            advantages[sample_idx, 1:] *= think_scaling[i]

                                    scaling_metrics = {
                                        "actor/advantage_scaling_factor_mean": think_scaling.mean().item(),
                                        "actor/advantage_scaling_factor_max": think_scaling.max().item(),
                                        "actor/advantage_scaling_factor_min": think_scaling.min().item(),
                                    }
                                    print(f"[Debug] Think advantage scaling applied. Mean: {scaling_metrics['actor/advantage_scaling_factor_mean']:.4f}")
                                    print(f"[Debug] Think advantage scaling applied. Max: {scaling_metrics['actor/advantage_scaling_factor_max']:.4f}")
                                    print(f"[Debug] Think advantage scaling applied. Min: {scaling_metrics['actor/advantage_scaling_factor_min']:.4f}")

                            else:
                                print("[Warning] advantage_scaling_factor not found in batch data; skipping advantage scaling.")
                            #     # Fallback to original hard-coded logic if scaling factor not available
                            #     advantages[~enforce_nothinking, 0] = 0.3  # think样本：大的正advantage
                            #     advantages[~enforce_nothinking, 1:] *= 3.0
                            #     advantages[enforce_nothinking, 0] = 0.1  # no-think样本：小的advantage
                            # # set old_log_probs first token to 0
                            # # old_log_probs[:, 0] = torch.zeros_like(old_log_probs[:, 0])
                        
                        pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl, cond_loss, resp_loss = core_algos.compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_mode_high=self.config.clip_ratio_high, #+scaling_factor,
                            clip_ratio_mode_low=self.config.clip_ratio_low,# +scaling_factor,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            thinkless_alpha= self.config.think_alpha,
                            focal_rho=focal_rho,
                        )

                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        mode_mask = response_mask.clone()
                        mode_mask[:, 1:] = 0
                        resp_mask = response_mask.clone()
                        resp_mask[:, 0] = 0
                        mode_kl_loss = (kld * mode_mask).sum() / (response_mask.sum() + 1e-8)
                        resp_kl_loss = (kld * resp_mask).sum() / (response_mask.sum() + 1e-8)
                        metrics["actor/mode_kl_loss"] = mode_kl_loss.detach().item()
                        metrics["actor/resp_kl_loss"] = resp_kl_loss.detach().item()
                        metrics["actor/kl_loss"] = (mode_kl_loss + resp_kl_loss).detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                        pg_loss = pg_loss + mode_kl_loss * self.config.mode_kl_coef + resp_kl_loss * self.config.kl_coef


                    kl_no_think_metrics = {}
                    if (
                        self.config.enable_kl_no_think
                        and self.config.kl_no_think_coef != 0.0
                        and all(k in micro_batch.batch.keys() for k in [
                            "fake_input_ids", "fake_attention_mask", "fake_position_ids", "fake_responses"
                        ])
                    ):
                        # Build fake (no-think) micro-batch and forward without grad
                        fake_batch = {
                            "input_ids": micro_batch.batch["fake_input_ids"],
                            "attention_mask": micro_batch.batch["fake_attention_mask"],
                            "position_ids": micro_batch.batch["fake_position_ids"],
                            "responses": micro_batch.batch["fake_responses"],
                        }
                        if "multi_modal_inputs" in model_inputs:
                            fake_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        with torch.no_grad():
                            _, fake_log_probs, _, _ = self._forward_micro_batch(fake_batch, temperature=temperature)

                        # Map existing think log_probs (computed above on original responses)
                        # to align with fake_responses tokens, avoiding an extra forward.
                        # For no-think samples, fake_responses == original responses.
                        # For think samples, fake_responses removes <thinking> section and starts at <tool_call>.
                        fake_responses = micro_batch.batch["fake_responses"]
                        bs, Lf = fake_responses.size()
                        think_log_probs_for_fake = torch.zeros_like(fake_responses, dtype=log_probs.dtype, device=log_probs.device)
                        response_len = responses.size(1)
                        # Build non-pad lengths per row to limit indexing
                        actor_cfg = getattr(self.actor_module, "config", None)
                        pad_token_id = getattr(actor_cfg, "pad_token_id", 0) if actor_cfg is not None else 0
                        if pad_token_id is None:
                            pad_token_id = 0
                        nonpad_mask_rows = (fake_responses != pad_token_id)
                        for i in range(bs):
                            # number of valid tokens in fake_responses[i]
                            flen = int(nonpad_mask_rows[i].sum().item())
                            if flen <= 0:
                                continue
                            if enforce_nothinking[i]:
                                # no-think sample: positions align from start
                                think_log_probs_for_fake[i, :flen] = log_probs[i, :flen]
                            else:
                                # think sample: align from first <tool_call> in original responses
                                tool_mask = (responses[i] == self.tool_call_token_id)
                                if tool_mask.any():
                                    start = int(torch.where(tool_mask)[0][0].item())
                                else:
                                    start = 0
                                end = min(start + flen, response_len)
                                copy_len = max(0, end - start)
                                if copy_len > 0:
                                    think_log_probs_for_fake[i, :copy_len] = log_probs[i, start:end]

                        # Token mask: exclude padding, <tool_call>, and no-think samples
                        not_pad = (fake_responses != pad_token_id).to(think_log_probs_for_fake.dtype)
                        not_tool = (fake_responses != self.tool_call_token_id).to(think_log_probs_for_fake.dtype)
                        think_mask = (~enforce_nothinking).unsqueeze(1).to(think_log_probs_for_fake.dtype)
                        # mask out samples not start with 151657
                        first_token_mask = (fake_responses[:,0] == self.tool_call_token_id).unsqueeze(1).to(think_log_probs_for_fake.dtype)
                        token_mask = not_pad * not_tool * think_mask * first_token_mask
                        # KL(think || no-think) over masked tokens
                        kld_t2nt = core_algos.compute_kl(
                            log_probs=think_log_probs_for_fake,
                            ref_log_probs=fake_log_probs.detach(),
                            kl_penalty=self.config.kl_penalty,
                        )
                        kldnt2t = core_algos.compute_kl(
                            log_probs=fake_log_probs,
                            ref_log_probs=think_log_probs_for_fake.detach(),
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_t2nt = (kld_t2nt * token_mask).sum() / (token_mask.sum() + 1e-8)
                        kl_nt2t = (kldnt2t * token_mask).sum() / (token_mask.sum() + 1e-8)
                        #weighted_kl_t2nt = kl_t2nt * getattr(self.config, "kl_think_to_nothink_weight", 1.0)
                        # scaled_kl_no_think = weighted_kl_t2nt * self.config.kl_no_think_coef
                        tnt_mutual_kl = kl_t2nt + kl_nt2t
                        pg_loss = pg_loss + tnt_mutual_kl * self.config.kl_no_think_coef

                        kl_no_think_metrics.update({
                            "actor/kl_no_think_tokens": token_mask.sum().detach().item(),
                            "actor/kl_no_think_samples": (~enforce_nothinking).sum().detach().item(),
                            "actor/kl_think_to_nothink": kl_t2nt.detach().item(),
                            "actor/kl_nothink_to_think": kl_nt2t.detach().item(),
                            "actor/tnt_mutual_kl": tnt_mutual_kl.detach().item(),
                        })

                    # if self.calculate_entropy:
                    #     pg_loss = pg_loss - mode_entropy.mean() * self.config.entropy_bonus_alpha
                    #     metrics["actor/mode_entropy"] = mode_entropy.mean().detach().item()
                    
                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    cond_loss = cond_loss / gradient_accumulation
                    resp_loss = resp_loss / gradient_accumulation


                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher[0].detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower[0].detach().item(),
                        "actor/pg_clipfrac_higher_mode": pg_clipfrac_higher[1].detach().item(),
                        "actor/pg_clipfrac_lower_mode": pg_clipfrac_lower[1].detach().item(),
                        # "actor/entropy_bonus": entropy_bonus.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/cond_loss": cond_loss.mean().detach().item(),
                        "actor/clip_ratio_mode_high": self.config.clip_ratio_high,# +scaling_factor,
                        "advantage/sum_of_absolute": advantages[:, 0].abs().sum().detach().item(),
                        "advantage/think_advantage_sum": advantages[~enforce_nothinking, 0].sum().detach().item(),
                        "advantage/think_absolute_sum": advantages[~enforce_nothinking, 0].abs().sum().detach().item(),
                        "advantage/nothink_advantage_sum": advantages[enforce_nothinking, 0].sum().detach().item(),
                        "advantage/nothink_absolute_sum": advantages[enforce_nothinking, 0].abs().sum().detach().item(),
                    }

                    if scaling_metrics is not None:
                        batch_metrics.update(scaling_metrics)

                    # Add SIMKO-specific metrics if available
                    if self.config.simko and 'simko_metrics' in locals():
                        for key, value in simko_metrics.items():
                            batch_metrics[f"actor/simko_{key}"] = value.detach().item()

                    if kl_no_think_metrics:
                        batch_metrics.update(kl_no_think_metrics)
                    # Add conditional checks for non-empty tensors
                    if len(first_eot_probs) > 0:
                        batch_metrics['adapt_think/first_eot_token_probs/mean'] = first_eot_probs.mean().detach().item()
                        batch_metrics["actor/force_no_think_resp_entropy"] = force_no_think_resp_entropy.detach().item()
                        batch_metrics["actor/resp_loss_no_think/mean"] = resp_loss[enforce_nothinking].mean().detach().item()

                    if len(first_t_probs) > 0:
                        batch_metrics['adapt_think/first_t_token_probs/mean'] = first_t_probs.mean().detach().item()
                        batch_metrics["actor/force_think_resp_entropy"] = force_think_resp_entropy.detach().item()
                        batch_metrics["actor/resp_loss_think/mean"] = resp_loss[~enforce_nothinking].mean().detach().item()

                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics