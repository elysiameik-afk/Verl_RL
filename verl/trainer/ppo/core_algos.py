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
import math

import verl.utils.torch_functional as verl_F


def is_main_process():
    """检查是否为主进程"""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

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
                var_reduction = 1.0 - (smoothed_var / (raw_var + 1e-8))
                sequence_variance_reductions.append(var_reduction.item())

            smoothing_effect = torch.norm(valid_raw - valid_smoothed).item()
            sequence_smoothing_effects.append(smoothing_effect)
    
    # 调试输出：显示token级平滑效果
    if is_main_process() and batch_size > 0:
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
    
    # 计算整体指标（只对有效token）
    valid_mask = response_mask > 0
    valid_raw_weights = raw_weights[valid_mask]
    valid_smoothed_weights = smoothed_weights[valid_mask]

    raw_variance = valid_raw_weights.var()
    smoothed_variance = valid_smoothed_weights.var()
    overall_variance_reduction = 1.0 - (smoothed_variance / (raw_variance + 1e-8))

    # 编译最终指标
    ema_metrics = {
        # 核心方差指标
        'ema/raw_weights_variance': raw_variance.item(),
        'ema/smoothed_weights_variance': smoothed_variance.item(),
        'ema/variance_reduction_ratio': overall_variance_reduction.item(),

        # 平滑效果指标
        'ema/avg_sequence_variance_reduction': np.mean(sequence_variance_reductions) if sequence_variance_reductions else 0.0,
        'ema/avg_sequence_diff_l2': np.mean(sequence_smoothing_effects) if sequence_smoothing_effects else 0.0,

        # 基础统计
        'ema/raw_weights_mean': valid_raw_weights.mean().item(),
        'ema/smoothed_weights_mean': valid_smoothed_weights.mean().item(),
        'ema/raw_weights_std': valid_raw_weights.std().item(),
        'ema/smoothed_weights_std': valid_smoothed_weights.std().item(),

        # 差异指标
        'ema/weights_diff_l2': torch.norm(valid_raw_weights - valid_smoothed_weights, p=2).item(),
        'ema/weights_diff_l1': torch.norm(valid_raw_weights - valid_smoothed_weights, p=1).item(),

        # 配置信息
        'ema/beta': beta,
        'ema/use_ema': True,
        'ema/processed_sequences': batch_size,
        'ema/total_valid_tokens': response_mask.sum().item(),
    }
    
    return smoothed_weights, ema_metrics


def apply_gradient_adaptive_weighting(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    创新点 2.2: 梯度自适应重要性加权

    计算基于梯度范数的贡献权重 c_{i,t}

    Args:
        log_probs: [batch_size, seq_len] - 对数概率
        response_mask: [batch_size, seq_len] - 有效token的mask
        temperature: float - softmax温度参数

    Returns:
        tuple: (contribution_weights, metrics)
    """
    batch_size, seq_len = log_probs.shape
    contribution_weights = torch.ones_like(log_probs)

    sequence_gradient_norms = []

    for i in range(batch_size):
        sequence_log_probs = log_probs[i]  # [seq_len]
        sequence_mask = response_mask[i]   # [seq_len]

        valid_positions = torch.where(sequence_mask > 0)[0]
        if len(valid_positions) <= 1:
            continue

        # 计算每个token的梯度范数 (简化版本，使用对数概率的绝对值作为代理)
        gradient_norms = []
        for pos in valid_positions:
            token_log_prob = sequence_log_probs[pos]
            # 使用对数概率的绝对值作为梯度范数的代理
            grad_norm = abs(token_log_prob.item())
            gradient_norms.append(grad_norm)

        if len(gradient_norms) > 0:
            # 应用softmax得到归一化的贡献权重
            gradient_norms_tensor = torch.tensor(gradient_norms, device=log_probs.device)
            softmax_weights = torch.softmax(gradient_norms_tensor / temperature, dim=0)

            # 乘以序列长度保持尺度
            scaled_weights = softmax_weights * len(valid_positions)

            # 分配权重
            for j, pos in enumerate(valid_positions):
                contribution_weights[i, pos] = scaled_weights[j]

            sequence_gradient_norms.extend(gradient_norms)

    # 计算指标
    metrics = {
        'gradient_adaptive/avg_gradient_norm': np.mean(sequence_gradient_norms) if sequence_gradient_norms else 0.0,
        'gradient_adaptive/max_gradient_norm': np.max(sequence_gradient_norms) if sequence_gradient_norms else 0.0,
        'gradient_adaptive/min_gradient_norm': np.min(sequence_gradient_norms) if sequence_gradient_norms else 0.0,
        'gradient_adaptive/weight_variance': contribution_weights[response_mask > 0].var().item(),
        'gradient_adaptive/weight_mean': contribution_weights[response_mask > 0].mean().item(),
        'gradient_adaptive/temperature': temperature,
        'gradient_adaptive/use_gradient_adaptive': True,
    }

    return contribution_weights, metrics


def apply_amic_aggregation(
    raw_weights: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    创新点 2.3: 算术平均重要性校正 (AMIC)

    计算序列级的算术平均权重 s'_i = (1/|y_i|) * Σ w_{i,t}

    Args:
        raw_weights: [batch_size, seq_len] - 原始重要性权重
        response_mask: [batch_size, seq_len] - 有效token的mask

    Returns:
        tuple: (sequence_weights, metrics)
    """
    batch_size = raw_weights.shape[0]
    sequence_weights = torch.zeros(batch_size, device=raw_weights.device)

    sequence_lengths = []
    sequence_variances = []

    for i in range(batch_size):
        sequence_mask = response_mask[i]
        valid_positions = torch.where(sequence_mask > 0)[0]

        if len(valid_positions) > 0:
            valid_weights = raw_weights[i][valid_positions]
            # 算术平均
            sequence_weights[i] = valid_weights.mean()

            sequence_lengths.append(len(valid_positions))
            if len(valid_positions) > 1:
                sequence_variances.append(valid_weights.var().item())
        else:
            sequence_weights[i] = 1.0  # 默认值

    # 计算指标
    metrics = {
        'amic/sequence_weights_mean': sequence_weights.mean().item(),
        'amic/sequence_weights_std': sequence_weights.std().item(),
        'amic/sequence_weights_variance': sequence_weights.var().item(),
        'amic/avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0.0,
        'amic/avg_token_variance': np.mean(sequence_variances) if sequence_variances else 0.0,
        'amic/use_amic': True,
    }

    return sequence_weights, metrics


def apply_temporal_decay_weighting(
    sequence_length: int,
    gamma: float = 0.95,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    创新点 2.5: 基于时序衰减的优势塑造

    计算时序衰减权重 d(t) = γ^(t-1)

    Args:
        sequence_length: int - 序列长度
        gamma: float - 衰减因子 γ ∈ (0, 1]
        normalize: bool - 是否归一化权重

    Returns:
        tuple: (decay_weights, metrics)
    """
    # 计算衰减权重
    positions = torch.arange(1, sequence_length + 1, dtype=torch.float32)
    decay_weights = gamma ** (positions - 1)

    if normalize:
        # 可选归一化：使总和等于1（保持相对权重比例）
        decay_weights = decay_weights / decay_weights.sum()
    # 注意：不归一化时，早期token获得更大的绝对权重，更符合创新点原理

    # 计算指标
    metrics = {
        'temporal_decay/gamma': gamma,
        'temporal_decay/normalize': normalize,
        'temporal_decay/weight_sum': decay_weights.sum().item(),
        'temporal_decay/weight_mean': decay_weights.mean().item(),
        'temporal_decay/weight_std': decay_weights.std().item(),
        'temporal_decay/first_weight': decay_weights[0].item(),
        'temporal_decay/last_weight': decay_weights[-1].item(),
        'temporal_decay/use_temporal_decay': True,
    }

    return decay_weights, metrics


def apply_ptrw_objective(
    importance_weights: torch.Tensor,
    advantages: torch.Tensor,
    sigma: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    创新点 2.4: 概率性信任区域加权 (PTRW)

    计算高斯信任权重 φ(s_i) = exp(-(s_i - 1)^2 / (2σ^2))

    Args:
        importance_weights: [batch_size] - 重要性权重
        advantages: [batch_size] - 优势值
        sigma: float - 高斯信任区域宽度

    Returns:
        tuple: (ptrw_loss, metrics)
    """
    # 计算高斯信任权重
    trust_weights = torch.exp(-((importance_weights - 1.0) ** 2) / (2 * sigma ** 2))

    # 计算PTRW损失
    ptrw_loss = -trust_weights * importance_weights * advantages

    # 计算指标
    metrics = {
        'ptrw/sigma': sigma,
        'ptrw/trust_weights_mean': trust_weights.mean().item(),
        'ptrw/trust_weights_std': trust_weights.std().item(),
        'ptrw/trust_weights_min': trust_weights.min().item(),
        'ptrw/trust_weights_max': trust_weights.max().item(),
        'ptrw/loss_mean': ptrw_loss.mean().item(),
        'ptrw/use_ptrw': True,
    }

    return ptrw_loss, metrics


def apply_asymmetric_clipping(
    importance_weights: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_pos: float = 0.3,
    clip_ratio_neg: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    创新点 2.6: 正负优势的非对称策略优化

    根据优势符号使用不同的裁剪范围

    Args:
        importance_weights: [batch_size] - 重要性权重
        advantages: [batch_size] - 优势值
        clip_ratio_pos: float - 正优势的裁剪范围
        clip_ratio_neg: float - 负优势的裁剪范围

    Returns:
        tuple: (clipped_weights, metrics)
    """
    clipped_weights = torch.zeros_like(importance_weights)

    # 正优势样本
    pos_mask = advantages > 0
    if pos_mask.any():
        clipped_weights[pos_mask] = torch.clamp(
            importance_weights[pos_mask],
            1.0 - clip_ratio_pos,
            1.0 + clip_ratio_pos
        )

    # 负优势样本
    neg_mask = advantages <= 0
    if neg_mask.any():
        clipped_weights[neg_mask] = torch.clamp(
            importance_weights[neg_mask],
            1.0 - clip_ratio_neg,
            1.0 + clip_ratio_neg
        )

    # 计算指标
    pos_count = pos_mask.sum().item()
    neg_count = neg_mask.sum().item()
    total_count = len(advantages)

    metrics = {
        'asymmetric/clip_ratio_pos': clip_ratio_pos,
        'asymmetric/clip_ratio_neg': clip_ratio_neg,
        'asymmetric/pos_advantage_ratio': pos_count / total_count if total_count > 0 else 0.0,
        'asymmetric/neg_advantage_ratio': neg_count / total_count if total_count > 0 else 0.0,
        'asymmetric/pos_clipped_ratio': (torch.abs(importance_weights[pos_mask] - clipped_weights[pos_mask]) > 1e-6).float().mean().item() if pos_count > 0 else 0.0,
        'asymmetric/neg_clipped_ratio': (torch.abs(importance_weights[neg_mask] - clipped_weights[neg_mask]) > 1e-6).float().mean().item() if neg_count > 0 else 0.0,
        'asymmetric/use_asymmetric': True,
    }

    return clipped_weights, metrics


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
    return compute_policy_loss_with_innovations(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=loss_agg_mode,
        use_ema_smoothing=False,
    )[:-1]  # Remove innovation_metrics from return


def compute_policy_loss_with_innovations(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    # 创新点配置
    use_ema_smoothing=False,
    ema_beta=0.9,
    use_gradient_adaptive_weighting=False,
    gradient_weighting_temperature=1.0,
    use_amic=False,
    use_ptrw=False,
    ptrw_sigma=0.2,
    use_temporal_decay=False,
    temporal_decay_gamma=0.95,
    temporal_decay_normalize=True,
    use_asymmetric_clipping=False,
    clip_ratio_pos=0.3,
    clip_ratio_neg=0.1,
):
    """
    综合所有创新点的策略损失计算函数

    支持的创新点：
    - 2.1: 时序平滑 (EMA) 的重要性权重
    - 2.2: 梯度自适应重要性加权
    - 2.3: 算术平均重要性校正 (AMIC)
    - 2.4: 概率性信任区域加权 (PTRW)
    - 2.5: 基于时序衰减的优势塑造
    - 2.6: 正负优势的非对称策略优化
    """
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

    batch_size, seq_len = old_log_prob.shape

    # 收集所有指标
    all_metrics = {}

    # 创新点 2.1: EMA 时序平滑
    if use_ema_smoothing:
        ratio, ema_metrics = apply_token_level_ema_smoothing(
            raw_weights=raw_ratio,
            response_mask=response_mask,
            beta=ema_beta,
        )
        all_metrics.update(ema_metrics)
        if is_main_process():
            print(f"🎯 [创新点2.1-EMA] 应用时序平滑, beta={ema_beta}")
    else:
        ratio = raw_ratio

    # 创新点 2.2: 梯度自适应重要性加权
    contribution_weights = torch.ones_like(ratio)
    if use_gradient_adaptive_weighting:
        contribution_weights, grad_metrics = apply_gradient_adaptive_weighting(
            log_probs=log_prob,
            response_mask=response_mask,
            temperature=gradient_weighting_temperature,
        )
        all_metrics.update(grad_metrics)
        if is_main_process():
            print(f"🎯 [创新点2.2-梯度自适应] 应用梯度加权, temperature={gradient_weighting_temperature}")

    # 创新点 2.3: AMIC 算术平均重要性校正
    if use_amic:
        # 计算序列级权重
        sequence_weights, amic_metrics = apply_amic_aggregation(
            raw_weights=ratio,
            response_mask=response_mask,
        )
        all_metrics.update(amic_metrics)
        if is_main_process():
            print(f"🎯 [创新点2.3-AMIC] 应用算术平均重要性校正")

        # 将序列级权重扩展到token级
        ratio = sequence_weights.unsqueeze(1).expand(-1, seq_len)

    # 创新点 2.5: 时序衰减优势塑造
    temporal_weights = torch.ones_like(ratio)
    if use_temporal_decay:
        all_decay_weights = []
        for i in range(batch_size):
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) > 0:
                decay_weights, _ = apply_temporal_decay_weighting(
                    sequence_length=len(valid_positions),
                    gamma=temporal_decay_gamma,
                    normalize=temporal_decay_normalize,
                )
                # 只对有效位置设置衰减权重，无效位置保持1（但会被mask掉）
                temporal_weights[i, valid_positions] = decay_weights.to(ratio.device)
                all_decay_weights.extend(decay_weights.tolist())
            # 对于没有有效token的序列，保持全1权重

        # 计算整体的时序衰减指标
        if all_decay_weights:
            # 计算单个序列的平均长度（用于归一化总和）
            num_sequences = len([i for i in range(batch_size) if torch.sum(response_mask[i]) > 0])
            avg_sequence_length = len(all_decay_weights) / num_sequences if num_sequences > 0 else 1

            decay_metrics = {
                'temporal_decay/gamma': temporal_decay_gamma,
                'temporal_decay/normalize': temporal_decay_normalize,
                'temporal_decay/weight_sum_per_sequence': sum(all_decay_weights) / num_sequences if num_sequences > 0 else 0,
                'temporal_decay/weight_mean': np.mean(all_decay_weights),
                'temporal_decay/weight_std': np.std(all_decay_weights),
                'temporal_decay/weight_min': min(all_decay_weights),
                'temporal_decay/weight_max': max(all_decay_weights),
                'temporal_decay/avg_sequence_length': avg_sequence_length,
                'temporal_decay/num_sequences': num_sequences,
                'temporal_decay/use_temporal_decay': True,
            }
        else:
            decay_metrics = {
                'temporal_decay/gamma': temporal_decay_gamma,
                'temporal_decay/normalize': temporal_decay_normalize,
                'temporal_decay/weight_mean': 1.0,
                'temporal_decay/use_temporal_decay': True,
            }

        all_metrics.update(decay_metrics)
        if is_main_process():
            print(f"🎯 [创新点2.5-时序衰减] 应用时序衰减, gamma={temporal_decay_gamma}, 平均权重={decay_metrics['temporal_decay/weight_mean']:.4f}, 序列数={decay_metrics.get('temporal_decay/num_sequences', 0)}")

    # 组合所有权重 (时序衰减作为权重因子，不修改优势)
    final_ratio = ratio * contribution_weights * temporal_weights

    # 优势保持原样，不被时序衰减直接修改
    final_advantages = advantages

    # 创新点 2.6: 非对称裁剪 或 创新点 2.4: PTRW
    if use_ptrw:
        # 使用PTRW目标函数
        sequence_advantages = verl_F.masked_mean(final_advantages, response_mask, dim=1)
        sequence_ratio = verl_F.masked_mean(final_ratio, response_mask, dim=1)

        ptrw_loss, ptrw_metrics = apply_ptrw_objective(
            importance_weights=sequence_ratio,
            advantages=sequence_advantages,
            sigma=ptrw_sigma,
        )
        all_metrics.update(ptrw_metrics)

        pg_loss = ptrw_loss.mean()
        pg_clipfrac = torch.tensor(0.0, device=ratio.device)  # PTRW不使用裁剪
        pg_clipfrac_lower = torch.tensor(0.0, device=ratio.device)

        if is_main_process():
            print(f"🎯 [创新点2.4-PTRW] 应用概率性信任区域, sigma={ptrw_sigma}")

    elif use_asymmetric_clipping:
        # 使用非对称裁剪
        sequence_advantages = verl_F.masked_mean(final_advantages, response_mask, dim=1)
        sequence_ratio = verl_F.masked_mean(final_ratio, response_mask, dim=1)

        clipped_ratio, asym_metrics = apply_asymmetric_clipping(
            importance_weights=sequence_ratio,
            advantages=sequence_advantages,
            clip_ratio_pos=clip_ratio_pos,
            clip_ratio_neg=clip_ratio_neg,
        )
        all_metrics.update(asym_metrics)

        # 扩展回token级
        clipped_ratio_expanded = clipped_ratio.unsqueeze(1).expand(-1, seq_len)

        pg_losses = -final_advantages * clipped_ratio_expanded
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

        # 计算裁剪统计
        pg_clipfrac = verl_F.masked_mean((torch.abs(sequence_ratio - clipped_ratio) > 1e-6).float().unsqueeze(1).expand(-1, seq_len), response_mask)
        pg_clipfrac_lower = torch.tensor(0.0, device=ratio.device)

        if is_main_process():
            print(f"🎯 [创新点2.6-非对称裁剪] 应用非对称裁剪, pos={clip_ratio_pos}, neg={clip_ratio_neg}")

    else:
        # 使用标准PPO裁剪
        pg_losses1 = -final_advantages * final_ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -final_advantages * torch.clamp(final_ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

        pg_losses3 = -final_advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (final_advantages < 0).float(), response_mask)

        pg_losses = torch.where(final_advantages < 0, clip_pg_losses2, clip_pg_losses1)
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # 添加基础指标
    all_metrics.update({
        'innovation/final_ratio_mean': (final_ratio * response_mask).mean().item(),
        'innovation/final_ratio_std': (final_ratio * response_mask).std().item(),
        'innovation/final_advantages_mean': (final_advantages * response_mask).mean().item(),
        'innovation/final_advantages_std': (final_advantages * response_mask).std().item(),
    })

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, all_metrics


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
