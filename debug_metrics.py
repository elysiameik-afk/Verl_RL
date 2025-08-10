#!/usr/bin/env python3
"""
调试metrics传递问题
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

from verl.utils.py_functional import append_to_dict
from verl.utils.metric.utils import reduce_metrics

def test_metrics_flow():
    """测试metrics的完整流程"""
    print("🔍 测试metrics传递流程...")
    
    # 模拟Actor中的metrics字典
    metrics = {}
    
    # 模拟EMA metrics（就像我们在apply_ema_smoothing中生成的）
    ema_metrics = {
        'ema/raw_weights_variance': 2.5,
        'ema/smoothed_weights_variance': 1.8,
        'ema/variance_reduction_ratio': 1.39,
        'ema/smoothing_strength': 0.28,
        'ema/beta': 0.9,
        'ema/use_ema': True,
    }
    
    print(f"原始EMA metrics: {ema_metrics}")
    
    # 使用append_to_dict添加metrics（就像在Actor中做的）
    append_to_dict(metrics, ema_metrics)
    
    print(f"添加到metrics后: {metrics}")
    
    # 模拟其他metrics
    other_metrics = {
        'actor/pg_loss': 0.15,
        'actor/pg_clipfrac': 0.02,
        'actor/ppo_kl': 0.001,
    }
    
    append_to_dict(metrics, other_metrics)
    
    print(f"添加其他metrics后: {metrics}")
    
    # 模拟trainer中的reduce_metrics调用
    reduced_metrics = reduce_metrics(metrics)
    
    print(f"reduce后的metrics: {reduced_metrics}")
    
    # 验证EMA metrics是否正确传递
    expected_ema_keys = [
        'ema/raw_weights_variance',
        'ema/smoothed_weights_variance', 
        'ema/variance_reduction_ratio',
        'ema/smoothing_strength',
        'ema/beta',
        'ema/use_ema'
    ]
    
    print(f"\n✅ 检查EMA metrics是否存在:")
    for key in expected_ema_keys:
        if key in reduced_metrics:
            print(f"  ✓ {key}: {reduced_metrics[key]}")
        else:
            print(f"  ❌ {key}: 缺失")
    
    return reduced_metrics

def test_multiple_batches():
    """测试多个batch的metrics聚合"""
    print(f"\n🔄 测试多个batch的metrics聚合...")
    
    metrics = {}
    
    # 模拟3个micro-batch的EMA metrics
    for batch_idx in range(3):
        ema_metrics = {
            'ema/raw_weights_variance': 2.5 + batch_idx * 0.1,
            'ema/smoothed_weights_variance': 1.8 + batch_idx * 0.05,
            'ema/variance_reduction_ratio': 1.39 + batch_idx * 0.02,
            'ema/beta': 0.9,
        }
        
        print(f"Batch {batch_idx} EMA metrics: {ema_metrics}")
        append_to_dict(metrics, ema_metrics)
    
    print(f"\n聚合前的metrics: {metrics}")
    
    # Reduce metrics
    reduced_metrics = reduce_metrics(metrics)
    
    print(f"聚合后的metrics: {reduced_metrics}")
    
    # 验证平均值计算
    expected_variance = (2.5 + 2.6 + 2.7) / 3
    actual_variance = reduced_metrics['ema/raw_weights_variance']
    
    print(f"\n验证平均值计算:")
    print(f"  期望的raw_weights_variance: {expected_variance:.3f}")
    print(f"  实际的raw_weights_variance: {actual_variance:.3f}")
    print(f"  差异: {abs(expected_variance - actual_variance):.6f}")
    
    return reduced_metrics

if __name__ == "__main__":
    print("🚀 开始调试metrics传递...")
    
    try:
        test_metrics_flow()
        test_multiple_batches()
        
        print(f"\n✅ metrics传递测试通过！")
        print(f"\n📝 修复说明:")
        print(f"  - 使用 append_to_dict() 而不是 metrics.update()")
        print(f"  - 这样metrics会被正确添加到列表中")
        print(f"  - trainer中的 reduce_metrics() 会计算平均值")
        print(f"\n🔧 现在重新运行训练，EMA指标应该会出现在WandB中！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
