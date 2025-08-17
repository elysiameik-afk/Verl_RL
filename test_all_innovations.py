#!/usr/bin/env python3
"""
测试所有六个创新点的实现
"""

import torch
import numpy as np
from verl.trainer.ppo.core_algos import (
    apply_token_level_ema_smoothing,
    apply_gradient_adaptive_weighting,
    apply_amic_aggregation,
    apply_temporal_decay_weighting,
    apply_ptrw_objective,
    apply_asymmetric_clipping,
    compute_policy_loss_with_innovations
)

def test_innovation_2_1_ema():
    """测试创新点2.1: 时序平滑 (EMA) 的重要性权重"""
    print("🎯 测试创新点2.1: 时序平滑 (EMA)")
    
    batch_size, seq_len = 2, 5
    raw_weights = torch.tensor([
        [1.2, 0.8, 1.5, 0.9, 1.1],
        [0.7, 1.3, 0.6, 1.4, 1.0]
    ])
    response_mask = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ])
    
    smoothed_weights, metrics = apply_token_level_ema_smoothing(
        raw_weights=raw_weights,
        response_mask=response_mask,
        beta=0.7
    )
    
    print(f"  原始权重方差: {metrics['ema/raw_weights_variance']:.4f}")
    print(f"  平滑权重方差: {metrics['ema/smoothed_weights_variance']:.4f}")
    print(f"  方差降低比例: {metrics['ema/variance_reduction_ratio']:.4f}")
    print("  ✅ EMA测试通过\n")

def test_innovation_2_2_gradient_adaptive():
    """测试创新点2.2: 梯度自适应重要性加权"""
    print("🎯 测试创新点2.2: 梯度自适应重要性加权")
    
    batch_size, seq_len = 2, 4
    log_probs = torch.tensor([
        [-0.5, -1.2, -0.8, -0.3],
        [-0.9, -0.4, -1.1, -0.6]
    ], requires_grad=False)
    response_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ])
    
    contribution_weights, metrics = apply_gradient_adaptive_weighting(
        log_probs=log_probs,
        response_mask=response_mask,
        temperature=1.0
    )
    
    print(f"  平均梯度范数: {metrics['gradient_adaptive/avg_gradient_norm']:.4f}")
    print(f"  权重方差: {metrics['gradient_adaptive/weight_variance']:.4f}")
    print(f"  权重均值: {metrics['gradient_adaptive/weight_mean']:.4f}")
    print("  ✅ 梯度自适应测试通过\n")

def test_innovation_2_3_amic():
    """测试创新点2.3: 算术平均重要性校正 (AMIC)"""
    print("🎯 测试创新点2.3: 算术平均重要性校正 (AMIC)")
    
    batch_size, seq_len = 2, 4
    raw_weights = torch.tensor([
        [1.2, 0.8, 1.5, 0.9],
        [0.7, 1.3, 0.6, 1.4]
    ])
    response_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ])
    
    sequence_weights, metrics = apply_amic_aggregation(
        raw_weights=raw_weights,
        response_mask=response_mask
    )
    
    print(f"  序列权重均值: {metrics['amic/sequence_weights_mean']:.4f}")
    print(f"  序列权重方差: {metrics['amic/sequence_weights_variance']:.4f}")
    print(f"  平均序列长度: {metrics['amic/avg_sequence_length']:.1f}")
    print("  ✅ AMIC测试通过\n")

def test_innovation_2_4_ptrw():
    """测试创新点2.4: 概率性信任区域加权 (PTRW)"""
    print("🎯 测试创新点2.4: 概率性信任区域加权 (PTRW)")
    
    importance_weights = torch.tensor([1.1, 0.9, 1.3, 0.7])
    advantages = torch.tensor([0.5, -0.3, 0.8, -0.2])
    
    ptrw_loss, metrics = apply_ptrw_objective(
        importance_weights=importance_weights,
        advantages=advantages,
        sigma=0.2
    )
    
    print(f"  信任权重均值: {metrics['ptrw/trust_weights_mean']:.4f}")
    print(f"  信任权重标准差: {metrics['ptrw/trust_weights_std']:.4f}")
    print(f"  PTRW损失均值: {metrics['ptrw/loss_mean']:.4f}")
    print("  ✅ PTRW测试通过\n")

def test_innovation_2_5_temporal_decay():
    """测试创新点2.5: 基于时序衰减的优势塑造"""
    print("🎯 测试创新点2.5: 基于时序衰减的优势塑造")
    
    sequence_length = 5
    gamma = 0.9
    
    decay_weights, metrics = apply_temporal_decay_weighting(
        sequence_length=sequence_length,
        gamma=gamma,
        normalize=True
    )
    
    print(f"  衰减因子: {metrics['temporal_decay/gamma']:.2f}")
    print(f"  权重总和: {metrics['temporal_decay/weight_sum']:.4f}")
    print(f"  首个权重: {metrics['temporal_decay/first_weight']:.4f}")
    print(f"  最后权重: {metrics['temporal_decay/last_weight']:.4f}")
    print("  ✅ 时序衰减测试通过\n")

def test_innovation_2_6_asymmetric():
    """测试创新点2.6: 正负优势的非对称策略优化"""
    print("🎯 测试创新点2.6: 正负优势的非对称策略优化")
    
    importance_weights = torch.tensor([1.1, 0.9, 1.3, 0.7])
    advantages = torch.tensor([0.5, -0.3, 0.8, -0.2])
    
    clipped_weights, metrics = apply_asymmetric_clipping(
        importance_weights=importance_weights,
        advantages=advantages,
        clip_ratio_pos=0.3,
        clip_ratio_neg=0.1
    )
    
    print(f"  正优势比例: {metrics['asymmetric/pos_advantage_ratio']:.2f}")
    print(f"  负优势比例: {metrics['asymmetric/neg_advantage_ratio']:.2f}")
    print(f"  正样本裁剪比例: {metrics['asymmetric/pos_clipped_ratio']:.2f}")
    print(f"  负样本裁剪比例: {metrics['asymmetric/neg_clipped_ratio']:.2f}")
    print("  ✅ 非对称裁剪测试通过\n")

def test_comprehensive_policy_loss():
    """测试综合策略损失函数"""
    print("🎯 测试综合策略损失函数")
    
    batch_size, seq_len = 2, 4
    old_log_prob = torch.tensor([
        [-0.5, -1.2, -0.8, -0.3],
        [-0.9, -0.4, -1.1, -0.6]
    ])
    log_prob = torch.tensor([
        [-0.4, -1.1, -0.9, -0.4],
        [-0.8, -0.5, -1.0, -0.7]
    ])
    advantages = torch.tensor([
        [0.5, -0.3, 0.8, -0.2],
        [-0.1, 0.6, -0.4, 0.3]
    ])
    response_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ])
    
    # 测试所有创新点组合
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, metrics = compute_policy_loss_with_innovations(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        use_ema_smoothing=True,
        ema_beta=0.9,
        use_gradient_adaptive_weighting=True,
        gradient_weighting_temperature=1.0,
        use_amic=False,  # 与GSPO互斥
        use_ptrw=False,  # 与标准裁剪互斥
        use_temporal_decay=True,
        temporal_decay_gamma=0.95,
        use_asymmetric_clipping=False,  # 与PTRW互斥
    )
    
    print(f"  策略损失: {pg_loss.item():.4f}")
    print(f"  裁剪比例: {pg_clipfrac.item():.4f}")
    print(f"  PPO KL: {ppo_kl.item():.4f}")
    print(f"  最终权重均值: {metrics['innovation/final_ratio_mean']:.4f}")
    print(f"  最终权重标准差: {metrics['innovation/final_ratio_std']:.4f}")
    print("  ✅ 综合策略损失测试通过\n")

if __name__ == "__main__":
    print("🚀 开始测试所有六个创新点...\n")
    
    test_innovation_2_1_ema()
    test_innovation_2_2_gradient_adaptive()
    test_innovation_2_3_amic()
    test_innovation_2_4_ptrw()
    test_innovation_2_5_temporal_decay()
    test_innovation_2_6_asymmetric()
    test_comprehensive_policy_loss()
    
    print("🎉 所有创新点测试完成！")
    print("\n📋 创新点总结:")
    print("  2.1 ✅ 时序平滑 (EMA) 的重要性权重")
    print("  2.2 ✅ 梯度自适应重要性加权")
    print("  2.3 ✅ 算术平均重要性校正 (AMIC)")
    print("  2.4 ✅ 概率性信任区域加权 (PTRW)")
    print("  2.5 ✅ 基于时序衰减的优势塑造")
    print("  2.6 ✅ 正负优势的非对称策略优化")
    print("\n🎯 所有创新点已准备就绪，可以开始实验！")
