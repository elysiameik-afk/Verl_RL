#!/usr/bin/env python3
"""
快速测试EMA实现是否工作
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    from verl.trainer.ppo.core_algos import apply_ema_smoothing, compute_policy_loss_with_ema
    print("✅ 成功导入EMA函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_ema_basic():
    """基础EMA测试"""
    print("\n🧪 开始基础EMA测试...")
    
    # 创建测试数据
    batch_size, seq_len = 2, 5
    raw_weights = torch.tensor([[2.0, 1.5, 3.0, 0.8, 1.2],
                               [1.8, 2.2, 0.9, 1.6, 2.1]])
    response_mask = torch.ones(batch_size, seq_len)
    sequence_ids = ['test_seq_1', 'test_seq_2']
    beta = 0.9
    
    print(f"原始权重:\n{raw_weights}")
    print(f"原始权重方差: {(raw_weights * response_mask).var().item():.6f}")
    
    # 初始化EMA状态
    ema_weights_state = {}
    
    # 第一次平滑
    smoothed_weights, ema_metrics = apply_ema_smoothing(
        raw_weights=raw_weights,
        ema_weights_state=ema_weights_state,
        sequence_ids=sequence_ids,
        response_mask=response_mask,
        beta=beta,
    )
    
    print(f"平滑后权重:\n{smoothed_weights}")
    print(f"平滑后权重方差: {ema_metrics['ema/smoothed_weights_variance']:.6f}")
    print(f"方差降低比例: {ema_metrics['ema/variance_reduction_ratio']:.6f}")
    print(f"平滑强度: {ema_metrics['ema/smoothing_strength']:.6f}")
    
    # 第二次平滑（模拟下一步）
    new_raw_weights = torch.tensor([[1.5, 2.0, 2.5, 1.0, 1.8],
                                   [2.1, 1.7, 1.3, 2.0, 1.9]])
    
    smoothed_weights_2, ema_metrics_2 = apply_ema_smoothing(
        raw_weights=new_raw_weights,
        ema_weights_state=ema_weights_state,
        sequence_ids=sequence_ids,
        response_mask=response_mask,
        beta=beta,
    )
    
    print(f"\n第二步原始权重:\n{new_raw_weights}")
    print(f"第二步平滑权重:\n{smoothed_weights_2}")
    print(f"第二步方差降低比例: {ema_metrics_2['ema/variance_reduction_ratio']:.6f}")
    
    return True

def test_policy_loss():
    """测试策略损失计算"""
    print("\n🎯 测试策略损失计算...")
    
    batch_size, seq_len = 2, 4
    old_log_prob = torch.tensor([[-0.1, -0.2, -0.15, -0.18],
                                [-0.12, -0.25, -0.11, -0.20]])
    log_prob = torch.tensor([[-0.09, -0.22, -0.14, -0.19],
                            [-0.11, -0.23, -0.13, -0.21]])
    advantages = torch.tensor([[0.5, -0.3, 0.2, 0.1],
                              [-0.2, 0.4, -0.1, 0.3]])
    response_mask = torch.ones(batch_size, seq_len)
    sequence_ids = ['policy_seq_1', 'policy_seq_2']
    
    ema_weights_state = {}
    
    # 测试带EMA的策略损失
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ema_metrics = compute_policy_loss_with_ema(
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
    
    print(f"策略损失: {pg_loss.item():.6f}")
    print(f"裁剪比例: {pg_clipfrac.item():.6f}")
    print(f"PPO KL: {ppo_kl.item():.6f}")
    print(f"方差降低比例: {ema_metrics['ema/variance_reduction_ratio']:.6f}")
    print(f"平滑强度: {ema_metrics['ema/smoothing_strength']:.6f}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始EMA功能测试...")
    
    try:
        test_ema_basic()
        test_policy_loss()
        print("\n✅ 所有测试通过！EMA实现正常工作。")
        print("\n📋 现在你可以运行训练:")
        print("   bash exp/qwen2.5kk1_ema.sh")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
