# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF
from copy import deepcopy


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,) - index for grouping
        eps: `(float)`
            small value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


# NOTE: GRPO with separated thinking/non-thinking groups
@torch.no_grad()
def compute_grpo_sep_outcome_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    index: torch.Tensor, 
    enforce_nothinking: torch.Tensor, 
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO with separated thinking/non-thinking groups.
    The first token uses standard GRPO (no grouping by thinking/non-thinking).
    Subsequent tokens use separated groups for normalization.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,) - index for grouping
        enforce_nothinking: `(torch.Tensor)`
            shape: (bs,) - boolean tensor indicating whether to enforce no thinking
        eps: `(float)`
            small value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    bsz, seq_len = token_level_rewards.shape
    print("Separated GRPO with mixed computation")
    
    # Initialize advantage tensor
    advantages = torch.zeros_like(token_level_rewards)
    
    # For the first token (t=0): use standard GRPO approach (no separation)
    # first_token_rewards = token_level_rewards[:, 0]
    scores = token_level_rewards.sum(dim=-1)
    id2score_first = defaultdict(list)
    id2mean_overall, id2std_overall = {}, {}
    
    for i in range(bsz):
        id2score_first[index[i]].append(scores[i])
    
    for idx in id2score_first:
        assert len(id2score_first[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean_overall[idx] = torch.mean(torch.tensor(id2score_first[idx]))
        id2std_overall[idx] = torch.std(torch.tensor(id2score_first[idx]))
    
    # Normalize first token using standard GRPO
    for i in range(bsz):
        advantages[i, 0] = (scores[i] - id2mean_overall[index[i]]) / (id2std_overall[index[i]] + eps)
    
    # For subsequent tokens (t>=1): use separated groups approach
    if seq_len > 1:
        # subsequent_rewards = token_level_rewards[:, 1:]  # (bs, seq_len-1)
        subsequent_scores = scores  # (bs,)
        
        # Group by both index and enforce_nothinking flag
        id2score_think = defaultdict(list)      # enforce_nothinking = False (thinking)
        id2score_nothink = defaultdict(list)    # enforce_nothinking = True (no thinking)
        id2mean_sep, id2std_sep = {}, {}
        
        for i in range(bsz):
            if enforce_nothinking[i]:
                id2score_nothink[index[i]].append(subsequent_scores[i])
            else:
                id2score_think[index[i]].append(subsequent_scores[i])
        
        # Compute mean and std for thinking group
        for idx in id2score_think:
            if len(id2score_think[idx]) > 1:
                id2mean_sep[(idx, False)] = torch.mean(torch.tensor(id2score_think[idx]))
                id2std_sep[(idx, False)] = torch.std(torch.tensor(id2score_think[idx]))
            else:
                # If only one sample, no normalization needed (keep original score)
                id2mean_sep[(idx, False)] = torch.mean(torch.tensor(id2score_think[idx]))
                id2std_sep[(idx, False)] = torch.tensor(1.0)
        
        # Compute mean and std for no-thinking group  
        for idx in id2score_nothink:
            if len(id2score_nothink[idx]) > 1:
                id2mean_sep[(idx, True)] = torch.mean(torch.tensor(id2score_nothink[idx]))
                id2std_sep[(idx, True)] = torch.std(torch.tensor(id2score_nothink[idx]))
            else:
                # If only one sample, no normalization needed (keep original score)
                id2mean_sep[(idx, True)] = torch.mean(torch.tensor(id2score_nothink[idx]))
                id2std_sep[(idx, True)] = torch.tensor(1.0)
        
        # Normalize subsequent tokens based on their respective groups
        normalized_subsequent_scores = torch.zeros_like(subsequent_scores)
        for i in range(bsz):
            group_key = (index[i], enforce_nothinking[i].item())
            # if group_key in id2mean_sep:
            normalized_subsequent_scores[i] = (subsequent_scores[i] - id2mean_sep[group_key]) / (id2std_sep[group_key] + eps)
            # else:
            #     # If group_key not found, keep original score (shouldn't happen if data is consistent)
            #     print(f"Warning: Group key {group_key} not found in id2mean_sep. Using original score.")
            #     normalized_subsequent_scores[i] = subsequent_scores[i]
        advantages[:, 1:] = normalized_subsequent_scores.unsqueeze(-1) * response_mask[:, 1:]
        # Distribute the normalized subsequent score across subsequent tokens
        # Use the response mask to only apply to valid tokens
        # for i in range(bsz):
        #     for t in range(1, seq_len):
        #         if response_mask[i, t] > 0:  # Only for valid tokens
        #             advantages[i, t] = normalized_subsequent_scores[i]

    # Apply response mask to final advantages
    
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns


@torch.no_grad()
def compute_grpo_dge_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    enforce_nothinking: torch.Tensor,
    cross_mode_diversity: torch.Tensor,
    first_token_probs: torch.Tensor,
    temperature: float = 0.25,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage for GRPO-DGE with diversity-guided first-token scaling."""

    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    enforce_nothinking = enforce_nothinking.view(-1).to(torch.bool)
    first_token_probs = first_token_probs.view(-1).to(device=device, dtype=dtype)
    cross_mode_diversity = torch.as_tensor(cross_mode_diversity, device=device, dtype=dtype)
    if cross_mode_diversity.ndim == 0:
        cross_mode_diversity = cross_mode_diversity.repeat(bsz)
    else:
        cross_mode_diversity = cross_mode_diversity.view(bsz, -1)[:, 0]

    advantages = torch.zeros_like(token_level_rewards)

    scores = token_level_rewards.sum(dim=-1)
    id2score_first = defaultdict(list)
    id2prob_think = defaultdict(list)
    id2prob_no_think = defaultdict(list)
    for i in range(bsz):
        if enforce_nothinking[i]:
            id2prob_no_think[index[i]].append(first_token_probs[i])
        else:
            id2prob_think[index[i]].append(first_token_probs[i])
    id2ratio = {}
    for idx in id2prob_think:
        think_mean_prob = torch.mean(torch.tensor(id2prob_think[idx])) if idx in id2prob_think else torch.tensor(0.0)
        id2ratio[(idx, False)] = (1.0 / (think_mean_prob + eps)) ** temperature
        no_think_mean_prob = torch.mean(torch.tensor(id2prob_no_think[idx])) if idx in id2prob_no_think else torch.tensor(0.0)
        id2ratio[(idx, True)] = (1.0 / (no_think_mean_prob + eps)) ** temperature

    id2mean_overall, id2std_overall = {}, {}
    # scale_factor = [1 + cross_mode_diversity[i] * (id2ratio[(index[i], enforce_nothinking[i].item())]-1) for i in range(bsz)]
    scale_factor = []
    for i in range(bsz):
        if (index[i], enforce_nothinking[i].item()) in id2ratio:
            scale_factor.append(1 + cross_mode_diversity[i] * (id2ratio[(index[i], enforce_nothinking[i].item())]-1))
        else:
            scale_factor.append(torch.tensor(1.0, device=device, dtype=dtype))
    
    print('Think scale factors:', [scale_factor[i].item() for i in range(bsz) if not enforce_nothinking[i]])
    print('No-think scale factors:', [scale_factor[i].item() for i in range(bsz) if enforce_nothinking[i]])

    first_token_scores = scores * torch.tensor(scale_factor, device=device, dtype=dtype)
    for i in range(bsz):
        id2score_first[index[i]].append(first_token_scores[i])

    for idx in id2score_first:
        assert len(id2score_first[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean_overall[idx] = torch.mean(torch.tensor(id2score_first[idx]))
        id2std_overall[idx] = torch.std(torch.tensor(id2score_first[idx]))
    
    # Normalize first token using standard GRPO
    for i in range(bsz):
        advantages[i, 0] = (first_token_scores[i] - id2mean_overall[index[i]]) / (id2std_overall[index[i]] + eps)
    
    # For subsequent tokens (t>=1): use separated groups approach
    if seq_len > 1:
        # subsequent_rewards = token_level_rewards[:, 1:]  # (bs, seq_len-1)
        subsequent_scores = scores  # (bs,)
        
        # Group by both index and enforce_nothinking flag
        id2score_think = defaultdict(list)      # enforce_nothinking = False (thinking)
        id2score_nothink = defaultdict(list)    # enforce_nothinking = True (no thinking)
        id2mean_sep, id2std_sep = {}, {}
        
        for i in range(bsz):
            if enforce_nothinking[i]:
                id2score_nothink[index[i]].append(subsequent_scores[i])
            else:
                id2score_think[index[i]].append(subsequent_scores[i])
        
        # Compute mean and std for thinking group
        for idx in id2score_think:
            if len(id2score_think[idx]) > 1:
                id2mean_sep[(idx, False)] = torch.mean(torch.tensor(id2score_think[idx]))
                id2std_sep[(idx, False)] = torch.std(torch.tensor(id2score_think[idx]))
            else:
                # If only one sample, no normalization needed (keep original score)
                id2mean_sep[(idx, False)] = torch.mean(torch.tensor(id2score_think[idx]))
                id2std_sep[(idx, False)] = torch.tensor(1.0)
        
        # Compute mean and std for no-thinking group  
        for idx in id2score_nothink:
            if len(id2score_nothink[idx]) > 1:
                id2mean_sep[(idx, True)] = torch.mean(torch.tensor(id2score_nothink[idx]))
                id2std_sep[(idx, True)] = torch.std(torch.tensor(id2score_nothink[idx]))
            else:
                # If only one sample, no normalization needed (keep original score)
                id2mean_sep[(idx, True)] = torch.mean(torch.tensor(id2score_nothink[idx]))
                id2std_sep[(idx, True)] = torch.tensor(1.0)
        
        # Normalize subsequent tokens based on their respective groups
        normalized_subsequent_scores = torch.zeros_like(subsequent_scores)
        for i in range(bsz):
            group_key = (index[i], enforce_nothinking[i].item())
            # if group_key in id2mean_sep:
            normalized_subsequent_scores[i] = (subsequent_scores[i] - id2mean_sep[group_key]) / (id2std_sep[group_key] + eps)
            # else:
            #     # If group_key not found, keep original score (shouldn't happen if data is consistent)
            #     print(f"Warning: Group key {group_key} not found in id2mean_sep. Using original score.")
            #     normalized_subsequent_scores[i] = subsequent_scores[i]
        advantages[:, 1:] = normalized_subsequent_scores.unsqueeze(-1) * response_mask[:, 1:]
        # Distribute the normalized subsequent score across subsequent tokens
        # Use the response mask to only apply to valid tokens
        # for i in range(bsz):
        #     for t in range(1, seq_len):
        #         if response_mask[i, t] > 0:  # Only for valid tokens
        #             advantages[i, t] = normalized_subsequent_scores[i]

    # Apply response mask to final advantages
    
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns

@torch.no_grad()
def compute_grpo_b_dge_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    enforce_nothinking: torch.Tensor,
    cross_mode_diversity: torch.Tensor,
    rollout_prob: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage for GRPO-DGE with diversity-guided first-token scaling."""

    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    enforce_nothinking = enforce_nothinking.view(-1).to(torch.bool)
    cross_mode_diversity = torch.as_tensor(cross_mode_diversity, device=device, dtype=dtype)
    if cross_mode_diversity.ndim == 0:
        cross_mode_diversity = cross_mode_diversity.repeat(bsz)
    else:
        cross_mode_diversity = cross_mode_diversity.view(bsz, -1)[:, 0]

    advantages = scores.unsqueeze(-1) * response_mask

    id2score_first = defaultdict(list)

    id2mean_overall, id2std_overall = {}, {}
    # scale_factor = [1 + cross_mode_diversity[i] * (1/(rollout_prob[i]+eps)-1) if rollout_prob[i] < 0.3 else torch.tensor(1.0, device=device, dtype=dtype) for i in range(bsz)]
    # scale_factor = [1 + (1/(rollout_prob[i]+eps)-1) if rollout_prob[i] < 0.3 else torch.tensor(1.0, device=device, dtype=dtype) for i in range(bsz)]
    # DivScore(mode) = -log( p_batch(mode) )
    div_score = [ -torch.log(rollout_prob[i]) * alpha if scores[i] >= 1.5 else torch.tensor(0.0, device=device, dtype=dtype) for i in range(bsz)]
    
    print('Think div_score:', [div_score[i].item() for i in range(bsz) if not enforce_nothinking[i]])
    print('No-think div_score:', [div_score[i].item() for i in range(bsz) if enforce_nothinking[i]])

    first_token_scores = scores + torch.tensor(div_score, device=device, dtype=dtype)
    for i in range(bsz):
        id2score_first[index[i]].append(first_token_scores[i])

    for idx in id2score_first:
        assert len(id2score_first[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean_overall[idx] = torch.mean(torch.tensor(id2score_first[idx]))
        id2std_overall[idx] = torch.std(torch.tensor(id2score_first[idx]))
    
    # Normalize first token using standard GRPO
    for i in range(bsz):
        advantages[i, 0] = (first_token_scores[i] - id2mean_overall[index[i]]) / (id2std_overall[index[i]] + eps)
    
    
    advantages = advantages * response_mask
    returns = advantages.clone()
    
    return advantages, returns

@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def compute_policy_loss_simko(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_mode_high: float,
    clip_ratio_mode_low: float,
    clip_ratio_dual: float,
    thinkless_alpha: float,
    focal_rho: float = -1.0,
    positive_smooth_alpha: float = -1.0,
    negation_penalty: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_mode_high: float,
    clip_ratio_mode_low: float,
    clip_ratio_dual: float,
    thinkless_alpha: float,
    focal_rho: float = -1.0,
    positive_smooth_alpha: float = -1.0,
    negation_penalty: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """
    if focal_rho >= 0:
        with torch.no_grad(): # 确保这部分计算不产生梯度
            # 1. 获取第一个token (模式控制token) 的新策略概率 π_θ
            # log_probs的形状是 (bs, response_length)
            mode_log_prob = log_probs[:, 0]
            mode_prob = torch.exp(mode_log_prob)

            # 2. 计算Focal Loss调制因子 (1 - π_θ)^ρ
            # 为了数值稳定性，在π_θ接近1时进行clamp
            focal_factor = torch.pow(1 - mode_prob.clamp(max=1-1e-7), focal_rho)
            
            # 增加一个维度以匹配advantages的形状
            focal_factor = focal_factor.unsqueeze(1) # shape: (bs, 1)

        # 3. 创建一个新的advantages张量，只调制第一个token的优势
        # 我们复制advantages以避免原地修改可能带来的副作用
        original_advantages = advantages
        modulated_advantages = torch.clone(original_advantages)
        
        # 只修改第一个token (t=0) 的优势
        modulated_advantages[:, 0] = modulated_advantages[:, 0] * focal_factor[:, 0]
        
        # 将advantages替换为被调制过的版本
        advantages = modulated_advantages

    negative_approx_kl = log_probs - old_log_probs
    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(negative_approx_kl)

    think_clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl[:,0], np.log(1 - clip_ratio_mode_low), np.log(1 + clip_ratio_mode_high))
    )

    think_clipped_ratio = think_clipped_ratio.unsqueeze(1)

    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl[:,1:], np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    clipped_ratio = torch.cat((think_clipped_ratio, clipped_ratio), dim=1)

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()
    # Masks    
    cond_mask = response_mask.clone()
    cond_mask[:, 1:] = 0                    # only t = 0, the control token
    resp_mask = response_mask.clone()
    resp_mask[:, 0]  = 0                    # t ≥ 1, the response tokens

    # avg_len = response_mask.sum(dim=1).float().mean()
    cond_loss_for_log = VF.masked_mean(final_pg_loss, cond_mask, dim=1)  # average over the control token
    resp_loss_for_log = VF.masked_mean(final_pg_loss, resp_mask, dim=1)  # average over the response tokens
    
    if thinkless_alpha >= 0: # Decoupled GRPO
        # print("Decoupled GRPO")
        cond_loss = VF.masked_mean(final_pg_loss, cond_mask)  # average over the control token
        resp_loss = VF.masked_mean(final_pg_loss, resp_mask)  # average over
        final_pg_loss = thinkless_alpha * cond_loss + resp_loss
    else: 
        final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    pg_clipfrac_higher_all = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_higher_mode = VF.masked_mean(pg_clipfrac_higher, cond_mask)
    pg_clipfrac_lower_all = VF.masked_mean(pg_clipfrac_lower, response_mask)
    pg_clipfrac_lower_mode = VF.masked_mean(pg_clipfrac_lower, cond_mask)

    pg_clipfrac_higher = [pg_clipfrac_higher_all, pg_clipfrac_higher_mode]
    pg_clipfrac_lower = [pg_clipfrac_lower_all, pg_clipfrac_lower_mode]

    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl, cond_loss_for_log, resp_loss_for_log


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
