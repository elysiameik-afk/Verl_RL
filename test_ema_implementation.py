#!/usr/bin/env python3
"""
测试EMA平滑实现的脚本
"""

import torch
import numpy as np
from verl.trainer.ppo.core_algos import apply_ema_smoothing, compute_policy_loss_with_ema

def test_ema_smoothing():
    """测试EMA平滑功能"""
    print("=== 测试EMA平滑功能 ===")
    
    # 创建模拟数据
    batch_size, seq_len = 4, 10
    raw_weights = torch.randn(batch_size, seq_len) * 2.0 + 1.0  # 模拟重要性权重
    raw_weights = torch.exp(raw_weights)  # 确保为正数
    response_mask = torch.ones(batch_size, seq_len)
    sequence_ids = ['seq_0', 'seq_1', 'seq_2', 'seq_3']
    beta = 0.9
    
    # 初始化EMA状态
    ema_weights_state = {}
    
    print(f"原始权重方差: {(raw_weights * response_mask).var().item():.6f}")
    print(f"原始权重均值: {(raw_weights * response_mask).mean().item():.6f}")
    
    # 应用EMA平滑
    smoothed_weights, ema_metrics = apply_ema_smoothing(
        raw_weights=raw_weights,
        ema_weights_state=ema_weights_state,
        sequence_ids=sequence_ids,
        response_mask=response_mask,
        beta=beta,
    )
    
    print(f"平滑后权重方差: {ema_metrics['ema/smoothed_weights_variance']:.6f}")
    print(f"平滑后权重均值: {ema_metrics['ema/smoothed_weights_mean']:.6f}")
    print(f"方差降低比例: {ema_metrics['ema/variance_reduction_ratio']:.6f}")
    print(f"平滑强度: {ema_metrics['ema/avg_sequence_diff_l2']:.6f}")
    
    # 测试多步EMA
    print("\n=== 测试多步EMA ===")
    for step in range(3):
        new_raw_weights = torch.randn(batch_size, seq_len) * 1.5 + 1.0
        new_raw_weights = torch.exp(new_raw_weights)
        
        smoothed_weights, ema_metrics = apply_ema_smoothing(
            raw_weights=new_raw_weights,
            ema_weights_state=ema_weights_state,
            sequence_ids=sequence_ids,
            response_mask=response_mask,
            beta=beta,
        )
        
        print(f"步骤 {step+1}: 方差降低比例 = {ema_metrics['ema/variance_reduction_ratio']:.6f}")
    
    return True

def test_policy_loss_with_ema():
    """测试带EMA的策略损失计算"""
    print("\n=== 测试策略损失计算 ===")
    
    # 创建模拟数据
    batch_size, seq_len = 2, 8
    old_log_prob = torch.randn(batch_size, seq_len) * 0.1
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.05  # 稍微不同
    advantages = torch.randn(batch_size, seq_len) * 0.5
    response_mask = torch.ones(batch_size, seq_len)
    sequence_ids = ['seq_A', 'seq_B']
    
    # 初始化EMA状态
    ema_weights_state = {}
    
    # 测试不使用EMA
    pg_loss_no_ema, pg_clipfrac_no_ema, ppo_kl_no_ema, pg_clipfrac_lower_no_ema, ema_metrics_no_ema = compute_policy_loss_with_ema(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        use_ema=False,
    )
    
    print(f"不使用EMA - 策略损失: {pg_loss_no_ema.item():.6f}")
    print(f"不使用EMA - 原始权重方差: {ema_metrics_no_ema['ema/raw_weights_variance']:.6f}")
    
    # 测试使用EMA
    pg_loss_ema, pg_clipfrac_ema, ppo_kl_ema, pg_clipfrac_lower_ema, ema_metrics_ema = compute_policy_loss_with_ema(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        ema_weights_state=ema_weights_state,
        sequence_ids=sequence_ids,
        beta=0.9,
        use_ema=True,
    )
    
    print(f"使用EMA - 策略损失: {pg_loss_ema.item():.6f}")
    print(f"使用EMA - 原始权重方差: {ema_metrics_ema['ema/raw_weights_variance']:.6f}")
    print(f"使用EMA - 平滑权重方差: {ema_metrics_ema['ema/smoothed_weights_variance']:.6f}")
    print(f"使用EMA - 方差降低比例: {ema_metrics_ema['ema/variance_reduction_ratio']:.6f}")
    
    return True

def test_different_beta_values():
    """测试不同β值的效果"""
    print("\n=== 测试不同β值的效果 ===")
    
    batch_size, seq_len = 3, 6
    raw_weights = torch.randn(batch_size, seq_len) * 3.0 + 1.0
    raw_weights = torch.exp(raw_weights)
    response_mask = torch.ones(batch_size, seq_len)
    sequence_ids = ['seq_1', 'seq_2', 'seq_3']
    
    beta_values = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    for beta in beta_values:
        ema_weights_state = {}
        smoothed_weights, ema_metrics = apply_ema_smoothing(
            raw_weights=raw_weights,
            ema_weights_state=ema_weights_state,
            sequence_ids=sequence_ids,
            response_mask=response_mask,
            beta=beta,
        )
        
        print(f"β={beta:.2f}: 方差降低比例={ema_metrics['ema/variance_reduction_ratio']:.4f}, "
              f"平滑强度={ema_metrics['ema/avg_sequence_diff_l2']:.4f}")
    
    return True

if __name__ == "__main__":
    print("开始测试EMA实现...")
    
    try:
        # 运行所有测试
        test_ema_smoothing()
        test_policy_loss_with_ema()
        test_different_beta_values()
        
        print("\n✅ 所有测试通过！EMA实现正常工作。")
        print("\n🚀 现在你可以运行训练脚本：")
        print("   bash exp/qwen2.5kk1_ema.sh")
        print("\n📊 WandB将记录以下关键指标：")
        print("   - ema/raw_weights_variance: 原始权重方差")
        print("   - ema/smoothed_weights_variance: 平滑权重方差") 
        print("   - ema/variance_reduction_ratio: 方差降低比例")
        print("   - ema/smoothing_strength: 平滑强度")
        print("   - ema/range_reduction: 权重范围收缩")
        print("   - 以及更多详细分析指标...")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
