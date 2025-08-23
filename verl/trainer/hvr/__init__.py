"""
HVR (Hindsight Value Reshaping) 内生奖励训练系统

基于EndoRM思想的内生奖励机制，使用模型自身的logits计算价值函数，
结合稀疏奖励进行后见之明的价值重塑，生成稠密的过程性奖励。

核心组件：
- ERVF (熵正则化价值函数): 基于logits的内生价值估计
- HVR (后见之明价值重塑): 结合稀疏奖励的价值轨迹重塑
- HVR训练器: 无需critic的简化训练流程
"""

from .hvr_core_algos import (
    calculate_ervf_value,
    calculate_hvr_rewards,
    hvr_policy_loss,
    HVRMetrics
)

from .hvr_trainer import HVRTrainer

__all__ = [
    "calculate_ervf_value",
    "calculate_hvr_rewards", 
    "hvr_policy_loss",
    "HVRMetrics",
    "HVRTrainer"
]
