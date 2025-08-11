# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
implement PPO-like algorithms.
"""

__all__ = ['register', "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE) # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
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
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO) # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.GRPO_PASSK) # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE) # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor,
                                                           epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.RLOO) # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.OPO) # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6,
                                  config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS) # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.REMAX) # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
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
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def apply_token_level_ema_smoothing(
    raw_weights: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """
    Apply token-level EMA smoothing within each sequence.
    
    For each sequence i, apply EMA along the token dimension t:
    w'[i,t] = β × w[i,t] + (1-β) × w'[i,t-1]
    
    This is the core innovation: "Temporal Smoothing of Importance Weights"
    
    Args:
        raw_weights: [batch_size, seq_len] - Raw importance weights w[i,t]
        response_mask: [batch_size, seq_len] - Mask for valid response tokens
        beta: float - EMA smoothing factor (0 < β ≤ 1)
    
    Returns:
        tuple: (smoothed_weights, ema_metrics)
    """
    import numpy as np
    
    batch_size, seq_len = raw_weights.shape
    smoothed_weights = torch.zeros_like(raw_weights)
    
    sequence_variance_reductions = []
    sequence_smoothing_effects = []
    
    # 对每个序列应用token级EMA
    for i in range(batch_size):
        sequence_mask = response_mask[i]  # [seq_len]
        sequence_weights = raw_weights[i]  # [seq_len]
        smoothed_sequence = torch.zeros_like(sequence_weights)
        
        # 找到有效token位置
        valid_positions = torch.where(sequence_mask > 0)[0]
        if len(valid_positions) == 0:
            smoothed_weights[i] = sequence_weights
            continue
            
        # 初始化第一个有效token: w'[i,0] = w[i,0]
        first_pos = valid_positions[0].item()
        smoothed_sequence[first_pos] = sequence_weights[first_pos]
        
        # 对后续有效token应用EMA: w'[i,t] = β × w[i,t] + (1-β) × w'[i,t-1]
        prev_smoothed = smoothed_sequence[first_pos]
        for pos_idx in range(1, len(valid_positions)):
            t = valid_positions[pos_idx].item()
            current_raw = sequence_weights[t]
            current_smoothed = beta * current_raw + (1 - beta) * prev_smoothed
            smoothed_sequence[t] = current_smoothed
            prev_smoothed = current_smoothed
        
        # 复制无效token
        invalid_mask = sequence_mask == 0
        smoothed_sequence[invalid_mask] = sequence_weights[invalid_mask]
        
        smoothed_weights[i] = smoothed_sequence
        
        # 计算每个序列的指标
        if len(valid_positions) > 1:
            valid_raw = sequence_weights[valid_positions]
            valid_smoothed = smoothed_sequence[valid_positions]
            
            raw_var = valid_raw.var()
            smoothed_var = valid_smoothed.var()
            
            if raw_var > 1e-8:
                var_reduction = raw_var / (smoothed_var + 1e-8)
                sequence_variance_reductions.append(var_reduction.item())
            
            smoothing_effect = torch.norm(valid_raw - valid_smoothed).item()
            sequence_smoothing_effects.append(smoothing_effect)
    
    # 调试输出：显示token级平滑效果
    if torch.distributed.get_rank() == 0 and batch_size > 0:
        i = 0
        valid_mask = response_mask[i] > 0
        if valid_mask.sum() > 1:
            raw_seq = raw_weights[i][valid_mask]
            smoothed_seq = smoothed_weights[i][valid_mask]
            print(f"🔍 [TOKEN-EMA] 序列{i} token级时序平滑:")
            print(f"  有效token数: {valid_mask.sum().item()}")
            print(f"  原始权重前5个: {raw_seq[:5].tolist()}")
            print(f"  平滑权重前5个: {smoothed_seq[:5].tolist()}")
            print(f"  序列内方差变化: {raw_seq.var().item():.6f} → {smoothed_seq.var().item():.6f}")
            print(f"  token级平滑强度: {torch.norm(raw_seq - smoothed_seq).item():.6f}")
            print(f"  beta={beta} (时序平滑因子)")
    
    """
    Apply EMA smoothing to importance weights.
    
    Args:
        raw_weights: Raw importance weights, shape (batch_size, response_length)
        ema_weights_state: Dictionary storing EMA state for each sequence
        sequence_ids: List of sequence IDs for current batch
        response_mask: Mask for valid tokens, shape (batch_size, response_length)
        beta: EMA smoothing factor
        
    Returns:
        smoothed_weights: EMA smoothed weights, same shape as raw_weights
        ema_metrics: Dictionary of metrics for logging
    """
    smoothed_weights = raw_weights.clone()
    batch_size, seq_len = raw_weights.shape
    
    # Initialize metrics
    variance_reductions = []
    smoothing_effects = []
    
    for i, seq_id in enumerate(sequence_ids):
        if seq_id not in ema_weights_state:
            # Initialize EMA state for new sequence
            # Use the current raw weights as initial state instead of ones
            ema_weights_state[seq_id] = {
                'prev_weights': raw_weights[i].detach().cpu().clone(),
                'step_count': 0
            }
        
        # Get previous EMA weights for this sequence
        prev_ema = ema_weights_state[seq_id]['prev_weights'].to(raw_weights.device)
        
        # Apply EMA smoothing: w'_{i,t} = β * w_{i,t} + (1 - β) * w'_{i,t-1}
        current_raw = raw_weights[i]
        current_smoothed = beta * current_raw + (1 - beta) * prev_ema
        
        # Debug: 打印详细信息（仅第一个序列，避免过多输出）
        if i == 0 and torch.distributed.get_rank() == 0:
            step_count = ema_weights_state[seq_id]['step_count']
            print(f"🔍 [EMA-DEBUG] seq_id={seq_id}, step={step_count}")
            print(f"  raw_weights范围: [{current_raw.min().item():.4f}, {current_raw.max().item():.4f}]")
            print(f"  prev_ema范围: [{prev_ema.min().item():.4f}, {prev_ema.max().item():.4f}]")
            print(f"  smoothed范围: [{current_smoothed.min().item():.4f}, {current_smoothed.max().item():.4f}]")
            print(f"  beta={beta}, 权重差异={torch.norm(current_raw - current_smoothed).item():.6f}")
        
        # Update smoothed weights
        smoothed_weights[i] = current_smoothed
        
        # Update EMA state
        ema_weights_state[seq_id]['prev_weights'] = current_smoothed.detach().cpu()
        ema_weights_state[seq_id]['step_count'] += 1
        
        # Compute metrics for this sequence (only for valid tokens)
        mask = response_mask[i]
        if mask.sum() > 0:
            raw_var = (current_raw * mask).var()
            smoothed_var = (current_smoothed * mask).var()
            if raw_var > 0:
                variance_reduction = (raw_var - smoothed_var) / raw_var
                variance_reductions.append(variance_reduction.item())
            
            smoothing_effect = torch.norm(current_raw - current_smoothed, p=2) * mask.sum() / mask.sum()
            smoothing_effects.append(smoothing_effect.item())
    
    # Compute overall metrics
    raw_variance = (raw_weights * response_mask).var()
    smoothed_variance = (smoothed_weights * response_mask).var()
    
    # Calculate additional metrics for detailed analysis
    weight_diff = raw_weights - smoothed_weights
    relative_change = torch.abs(weight_diff) / (torch.abs(raw_weights) + 1e-8)
    
    # Stability metrics
    raw_weights_masked = raw_weights * response_mask
    smoothed_weights_masked = smoothed_weights * response_mask
    
    # Percentile analysis for paper
    raw_weights_flat = raw_weights_masked[response_mask > 0]
    smoothed_weights_flat = smoothed_weights_masked[response_mask > 0]
    
    if len(raw_weights_flat) > 0:
        raw_p95 = torch.quantile(raw_weights_flat, 0.95).item()
        raw_p05 = torch.quantile(raw_weights_flat, 0.05).item()
        smoothed_p95 = torch.quantile(smoothed_weights_flat, 0.95).item()
        smoothed_p05 = torch.quantile(smoothed_weights_flat, 0.05).item()
        
        # Range reduction
        raw_range = raw_p95 - raw_p05
        smoothed_range = smoothed_p95 - smoothed_p05
        range_reduction = (raw_range - smoothed_range) / (raw_range + 1e-8)
    else:
        raw_p95 = raw_p05 = smoothed_p95 = smoothed_p05 = 0.0
        range_reduction = 0.0

    ema_metrics = {
        # === Core Variance Metrics ===
        'ema/raw_weights_variance': raw_variance.item(),
        'ema/smoothed_weights_variance': smoothed_variance.item(),
        'ema/variance_reduction_ratio': (raw_variance / (smoothed_variance + 1e-8)).item(),
        'ema/variance_reduction_absolute': (raw_variance - smoothed_variance).item(),
        
        # === Basic Statistics ===
        'ema/raw_weights_mean': (raw_weights * response_mask).mean().item(),
        'ema/raw_weights_std': (raw_weights * response_mask).std().item(),
        'ema/smoothed_weights_mean': (smoothed_weights * response_mask).mean().item(),
        'ema/smoothed_weights_std': (smoothed_weights * response_mask).std().item(),
        
        # === Difference Metrics ===
        'ema/weights_diff_l2': torch.norm(weight_diff, p=2).item(),
        'ema/weights_diff_l1': torch.norm(weight_diff, p=1).item(),
        'ema/weights_diff_max': torch.max(torch.abs(weight_diff)).item(),
        'ema/relative_change_mean': (relative_change * response_mask).mean().item(),
        'ema/relative_change_max': torch.max(relative_change * response_mask).item(),
        
        # === Percentile Analysis ===
        'ema/raw_weights_p95': raw_p95,
        'ema/raw_weights_p05': raw_p05,
        'ema/smoothed_weights_p95': smoothed_p95,
        'ema/smoothed_weights_p05': smoothed_p05,
        'ema/range_reduction': range_reduction,
        
        # === Configuration & State ===
        'ema/beta': beta,
        'ema/active_sequences': len(ema_weights_state),
        'ema/avg_variance_reduction': np.mean(variance_reductions) if variance_reductions else 0.0,
        'ema/avg_smoothing_effect': np.mean(smoothing_effects) if smoothing_effects else 0.0,
        
        # === Stability Indicators ===
        'ema/smoothing_strength': 1.0 - (smoothed_variance / (raw_variance + 1e-8)).item(),
        'ema/use_ema': True,
    }
    
    return smoothed_weights, ema_metrics


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.
    This is the original version without EMA smoothing.
    """
    return compute_policy_loss_with_ema(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=loss_agg_mode,
        ema_weights_state=None,
        sequence_ids=None,
        beta=None,
        use_ema=False,
    )[:-1]  # Remove ema_metrics from return


def compute_policy_loss_with_ema(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    ema_weights_state=None,
    sequence_ids=None,
    beta=0.9,
    use_ema=True,
):
    """
    Compute the clipped policy objective and related metrics for PPO with optional EMA smoothing.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        ema_weights_state (dict, optional):
            Dictionary storing EMA state for each sequence. Required if use_ema=True.
        sequence_ids (list, optional):
            List of sequence IDs for current batch. Required if use_ema=True.
        beta (float, optional):
            EMA smoothing factor. Defaults to 0.9.
        use_ema (bool, optional):
            Whether to apply EMA smoothing to importance weights. Defaults to True.
            
    Returns:
        pg_loss: Policy gradient loss
        pg_clipfrac: Fraction of clipped importance weights
        ppo_kl: KL divergence between old and new policy
        pg_clipfrac_lower: Lower clipping fraction
        ema_metrics: Dictionary of EMA-related metrics (only if use_ema=True)
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    raw_ratio = torch.exp(negative_approx_kl)  # Raw importance weights w_{i,t}
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Apply EMA smoothing if enabled
    ema_metrics = {}
    if use_ema:
        ratio, ema_metrics = apply_token_level_ema_smoothing(
            raw_weights=raw_ratio,
            response_mask=response_mask,
            beta=beta,
        )
    else:
        ratio = raw_ratio
        # Add basic metrics even without EMA
        ema_metrics = {
            'ema/raw_weights_variance': (raw_ratio * response_mask).var().item(),
            'ema/raw_weights_mean': (raw_ratio * response_mask).mean().item(),
            'ema/raw_weights_std': (raw_ratio * response_mask).std().item(),
            'ema/use_ema': False,
        }

    # Compute policy gradient loss using (possibly smoothed) importance weights
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ema_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
