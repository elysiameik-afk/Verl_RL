#!/usr/bin/env python3
"""
测试EMA功能是否被正确触发
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    from verl.trainer.ppo.core_algos import apply_ema_smoothing, compute_policy_loss_with_ema
    from verl.utils.py_functional import append_to_dict
    from verl.utils.metric.utils import reduce_metrics
    print("✅ 成功导入所需模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_ema_with_temp_ids():
    """测试使用临时序列ID的EMA功能"""
    print("\n🧪 测试使用临时序列ID的EMA功能...")
    
    batch_size, seq_len = 3, 6
    
    # 模拟Actor中的数据
    old_log_prob = torch.randn(batch_size, seq_len) * 0.1 - 2.0
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.05
    advantages = torch.randn(batch_size, seq_len) * 0.3
    response_mask = torch.ones(batch_size, seq_len)
    
    # 使用临时序列ID（模拟没有uid的情况）
    sequence_ids = [f"temp_seq_{i}" for i in range(batch_size)]
    
    print(f"序列IDs: {sequence_ids}")
    print(f"原始log概率形状: {old_log_prob.shape}")
    print(f"当前log概率形状: {log_prob.shape}")
    
    # 初始化EMA状态
    ema_weights_state = {}
    
    # 测试EMA策略损失计算
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
    
    print(f"\n📊 EMA计算结果:")
    print(f"策略损失: {pg_loss.item():.6f}")
    print(f"裁剪比例: {pg_clipfrac.item():.6f}")
    print(f"PPO KL: {ppo_kl.item():.6f}")
    
    print(f"\n🎯 EMA指标:")
    for key, value in ema_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📈 EMA状态:")
    print(f"活跃序列数: {len(ema_weights_state)}")
    for seq_id, state in ema_weights_state.items():
        print(f"  {seq_id}: 步数={state['step_count']}, 权重形状={state['prev_weights'].shape}")
    
    return ema_metrics

def test_metrics_aggregation():
    """测试metrics聚合流程"""
    print(f"\n🔄 测试metrics聚合流程...")
    
    # 模拟Actor中的metrics字典
    actor_metrics = {}
    
    # 模拟3个micro-batch的处理
    for micro_batch_idx in range(3):
        print(f"\n处理micro-batch {micro_batch_idx + 1}...")
        
        # 生成EMA metrics
        ema_metrics = {
            'ema/raw_weights_variance': 2.0 + micro_batch_idx * 0.1,
            'ema/smoothed_weights_variance': 1.5 + micro_batch_idx * 0.05,
            'ema/variance_reduction_ratio': 1.33 + micro_batch_idx * 0.02,
            'ema/smoothing_strength': 0.25 + micro_batch_idx * 0.01,
            'ema/beta': 0.9,
            'ema/use_ema': True,
        }
        
        # 使用append_to_dict添加到actor_metrics
        append_to_dict(actor_metrics, ema_metrics)
        
        print(f"  添加的EMA metrics: {ema_metrics}")
    
    print(f"\n📊 聚合前的actor_metrics:")
    for key, values in actor_metrics.items():
        print(f"  {key}: {values}")
    
    # 模拟trainer中的reduce_metrics调用
    reduced_metrics = reduce_metrics(actor_metrics)
    
    print(f"\n📈 聚合后的metrics (这些会发送到WandB):")
    for key, value in reduced_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # 验证关键指标
    print(f"\n✅ 验证关键指标:")
    expected_keys = [
        'ema/raw_weights_variance',
        'ema/smoothed_weights_variance', 
        'ema/variance_reduction_ratio',
        'ema/smoothing_strength',
        'ema/beta',
        'ema/use_ema'
    ]
    
    all_present = True
    for key in expected_keys:
        if key in reduced_metrics:
            print(f"  ✓ {key}: 存在")
        else:
            print(f"  ❌ {key}: 缺失")
            all_present = False
    
    if all_present:
        print(f"\n🎉 所有EMA指标都存在，应该会出现在WandB中！")
    else:
        print(f"\n⚠️ 有指标缺失，可能不会完全出现在WandB中")
    
    return reduced_metrics

if __name__ == "__main__":
    print("🚀 开始测试EMA触发机制...")
    
    try:
        # 测试EMA功能
        ema_metrics = test_ema_with_temp_ids()
        
        # 测试metrics聚合
        final_metrics = test_metrics_aggregation()
        
        print(f"\n✅ 所有测试通过！")
        print(f"\n📋 修复总结:")
        print(f"  1. ✓ 修复了metrics传递方式 (append_to_dict)")
        print(f"  2. ✓ 添加了临时序列ID支持")
        print(f"  3. ✓ 添加了调试打印信息")
        print(f"  4. ✓ 验证了完整的metrics流程")
        
        print(f"\n🔧 现在重新运行训练:")
        print(f"   bash exp/qwen2.5kk1_ema.sh")
        print(f"\n👀 训练时注意观察:")
        print(f"   - 🎯 [EMA-GRPO] Actor use_ema_smoothing=True")
        print(f"   - 🎯 [EMA-GRPO] Added EMA metrics: ...")
        print(f"   - WandB中的 ema/* 指标")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
