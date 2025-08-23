"""
HVR核心算法实现

实现基于EndoRM思想的内生奖励机制：
1. ERVF (熵正则化价值函数) - 改进的内生价值估计
2. HVR (后见之明价值重塑) - 稀疏奖励指导的价值重塑
3. HVR策略损失 - 基于内生奖励的策略优化
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class HVRMetrics:
    """HVR训练指标"""
    # ERVF相关指标
    ervf_value_mean: float = 0.0
    ervf_value_std: float = 0.0
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    
    # HVR奖励指标
    hvr_reward_mean: float = 0.0
    hvr_reward_std: float = 0.0
    hvr_reward_min: float = 0.0
    hvr_reward_max: float = 0.0
    
    # 稀疏奖励指标
    r_final_mean: float = 0.0
    r_final_std: float = 0.0
    r_final_distribution: Dict[float, int] = None
    
    # 价值重塑指标
    value_reshaping_ratio: float = 0.0
    target_value_mean: float = 0.0
    
    # 处理统计
    total_sequences: int = 0
    successful_hvr_count: int = 0
    success_rate: float = 0.0


def calculate_ervf_value(
    logits: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> Tuple[float, float]:
    """
    计算熵正则化价值函数 (ERVF)
    
    基于EndoRM思想，但加入熵惩罚来减少决策不确定性
    
    Args:
        logits: [vocab_size] - 模型在当前状态的logits输出
        alpha: float - 温度系数 α
        beta: float - 熵惩罚权重 β
    
    Returns:
        tuple: (v_ervf, entropy) - 熵正则化价值和策略熵
    """
    # 1. 计算LSE价值 V_endo (EndoRM基础价值)
    v_endo = alpha * torch.logsumexp(logits / alpha, dim=0)
    
    # 2. 计算策略熵 H
    probs = torch.softmax(logits, dim=0)
    # 避免log(0)，添加小的epsilon
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon))
    
    # 3. 计算最终的熵正则化价值
    v_ervf = v_endo - beta * entropy
    
    return v_ervf.item(), entropy.item()


def calculate_hvr_rewards(
    response_logits: torch.Tensor,
    response_ids: torch.Tensor,
    R_final: float,
    alpha: float = 1.0,
    beta: float = 0.1,
    lambda_hvr: float = 0.5,
) -> Tuple[torch.Tensor, HVRMetrics]:
    """
    计算HVR稠密奖励 (Hindsight Value Reshaping)
    
    结合稀疏奖励R_final和ERVF价值函数，生成稠密的过程性奖励
    
    Args:
        response_logits: [seq_len, vocab_size] - 每步的logits
        response_ids: [seq_len] - 实际生成的token序列
        R_final: float - 最终稀疏奖励 (来自logic_rl)
        alpha: float - 温度系数
        beta: float - 熵惩罚权重
        lambda_hvr: float - HVR混合因子 ∈ [0, 1]
    
    Returns:
        tuple: (hvr_rewards, metrics) - 稠密奖励序列和指标
    """
    seq_len, vocab_size = response_logits.shape
    device = response_logits.device
    
    # 1. 计算价值轨迹 V_ervf_list (长度为L+1)
    v_ervf_list = []
    entropy_list = []
    
    # 计算L个状态的价值
    for t in range(seq_len):
        logits_t = response_logits[t]  # [vocab_size]
        v_ervf_t, entropy_t = calculate_ervf_value(logits_t, alpha, beta)
        v_ervf_list.append(v_ervf_t)
        entropy_list.append(entropy_t)
    
    # 添加终止状态价值 V(s_{L+1}) = 0
    v_ervf_list.append(0.0)
    
    # 2. 计算重塑目标 V_target
    v_max = max(v_ervf_list[:-1])  # 排除终止状态
    v_min = min(v_ervf_list[:-1])
    
    # 将R_final归一化到[0,1] (适应logic_rl的奖励范围[-3,3])
    p = (R_final + 3.0) / 6.0  # R_final ∈ [-3,3] -> p ∈ [0,1]
    p = max(0.0, min(1.0, p))  # 确保在[0,1]范围内
    v_target = (1 - p) * v_min + p * v_max
    
    # 3. 计算重塑后的价值轨迹 V_hvr_list
    v_hvr_list = []
    for v in v_ervf_list:
        v_hvr = (1 - lambda_hvr) * v + lambda_hvr * v_target
        v_hvr_list.append(v_hvr)
    
    # 4. 计算稠密奖励 r_hvr_list
    r_hvr_list = []
    
    for t in range(seq_len):
        # 获取当前步的log概率
        logits_t = response_logits[t]  # [vocab_size]
        token_id_t = response_ids[t].item()
        
        # 使用数值稳定的log_softmax
        log_probs_t = torch.log_softmax(logits_t, dim=0)
        log_prob_t = log_probs_t[token_id_t].item()
        
        # HVR奖励公式: r_hvr_t = α * log_prob_t + V_hvr[t] - V_hvr[t+1]
        r_hvr_t = alpha * log_prob_t + v_hvr_list[t] - v_hvr_list[t + 1]
        r_hvr_list.append(r_hvr_t)
    
    # 5. 将最终奖励加到最后一步
    r_hvr_list[-1] += R_final
    
    # 转换为tensor
    hvr_rewards = torch.tensor(r_hvr_list, dtype=torch.float32, device=device)
    
    # 6. 计算指标
    metrics = HVRMetrics(
        ervf_value_mean=np.mean(v_ervf_list[:-1]),
        ervf_value_std=np.std(v_ervf_list[:-1]),
        entropy_mean=np.mean(entropy_list),
        entropy_std=np.std(entropy_list),
        hvr_reward_mean=hvr_rewards.mean().item(),
        hvr_reward_std=hvr_rewards.std().item(),
        hvr_reward_min=hvr_rewards.min().item(),
        hvr_reward_max=hvr_rewards.max().item(),
        r_final_mean=R_final,
        r_final_std=0.0,  # 单个序列
        value_reshaping_ratio=lambda_hvr,
        target_value_mean=v_target,
        total_sequences=1,
        successful_hvr_count=1,
        success_rate=1.0,
    )
    
    return hvr_rewards, metrics


def hvr_policy_loss(
    log_probs: torch.Tensor,
    hvr_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算基于HVR奖励的策略损失
    
    直接使用HVR生成的稠密奖励，无需额外的优势估计
    
    Args:
        log_probs: [batch_size, seq_len] - 策略的log概率
        hvr_rewards: [batch_size, seq_len] - HVR生成的稠密奖励
        response_mask: [batch_size, seq_len] - 有效token mask
        cliprange: float - PPO裁剪范围
        loss_agg_mode: str - 损失聚合模式
    
    Returns:
        tuple: (policy_loss, metrics) - 策略损失和指标
    """
    # 使用HVR奖励作为"优势"
    advantages = hvr_rewards
    
    # 简化的策略损失 (类似PPO，但使用HVR奖励)
    # 这里我们直接使用奖励作为梯度信号
    policy_loss = -(log_probs * advantages * response_mask).sum()
    
    # 归一化
    if loss_agg_mode == "token-mean":
        policy_loss = policy_loss / response_mask.sum()
    elif loss_agg_mode == "sequence-mean":
        policy_loss = policy_loss / response_mask.shape[0]
    
    # 计算指标
    metrics = {
        "hvr_policy_loss": policy_loss.item(),
        "hvr_advantages_mean": advantages[response_mask > 0].mean().item(),
        "hvr_advantages_std": advantages[response_mask > 0].std().item(),
        "hvr_log_probs_mean": log_probs[response_mask > 0].mean().item(),
    }
    
    return policy_loss, metrics


def aggregate_hvr_metrics(metrics_list: List[HVRMetrics]) -> Dict[str, float]:
    """聚合多个序列的HVR指标"""
    if not metrics_list:
        return {}
    
    # 聚合数值指标
    aggregated = {
        "hvr/ervf_value_mean": np.mean([m.ervf_value_mean for m in metrics_list]),
        "hvr/ervf_value_std": np.mean([m.ervf_value_std for m in metrics_list]),
        "hvr/entropy_mean": np.mean([m.entropy_mean for m in metrics_list]),
        "hvr/entropy_std": np.mean([m.entropy_std for m in metrics_list]),
        "hvr/hvr_reward_mean": np.mean([m.hvr_reward_mean for m in metrics_list]),
        "hvr/hvr_reward_std": np.mean([m.hvr_reward_std for m in metrics_list]),
        "hvr/hvr_reward_min": min([m.hvr_reward_min for m in metrics_list]),
        "hvr/hvr_reward_max": max([m.hvr_reward_max for m in metrics_list]),
        "hvr/r_final_mean": np.mean([m.r_final_mean for m in metrics_list]),
        "hvr/r_final_std": np.std([m.r_final_mean for m in metrics_list]),
        "hvr/value_reshaping_ratio": metrics_list[0].value_reshaping_ratio,  # 超参数
        "hvr/target_value_mean": np.mean([m.target_value_mean for m in metrics_list]),
        "hvr/total_sequences": sum([m.total_sequences for m in metrics_list]),
        "hvr/successful_hvr_count": sum([m.successful_hvr_count for m in metrics_list]),
        "hvr/success_rate": np.mean([m.success_rate for m in metrics_list]),
    }
    
    # 聚合R_final分布
    r_final_values = [m.r_final_mean for m in metrics_list]
    unique_values, counts = np.unique(r_final_values, return_counts=True)
    for val, count in zip(unique_values, counts):
        aggregated[f"hvr/r_final_dist_{val}"] = count
    
    return aggregated
