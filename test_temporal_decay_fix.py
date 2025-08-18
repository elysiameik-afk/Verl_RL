#!/usr/bin/env python3
"""
测试时序衰减权重的修复效果
"""

import torch
import numpy as np
from verl.trainer.ppo.core_algos import apply_temporal_decay_weighting

def test_temporal_decay_weights():
    """测试不同参数下的时序衰减权重"""
    
    print("🔧 测试时序衰减权重修复效果\n")
    
    # 测试参数
    sequence_length = 5
    gammas = [0.8, 0.9, 0.95, 0.99]
    
    for gamma in gammas:
        print(f"📊 γ = {gamma}")
        
        # 测试归一化版本
        decay_weights_norm, metrics_norm = apply_temporal_decay_weighting(
            sequence_length=sequence_length,
            gamma=gamma,
            normalize=True
        )
        
        # 测试非归一化版本
        decay_weights_raw, metrics_raw = apply_temporal_decay_weighting(
            sequence_length=sequence_length,
            gamma=gamma,
            normalize=False
        )
        
        print(f"  归一化版本:")
        print(f"    权重: {decay_weights_norm.tolist()}")
        print(f"    总和: {metrics_norm['temporal_decay/weight_sum']:.4f}")
        print(f"    均值: {metrics_norm['temporal_decay/weight_mean']:.4f}")
        print(f"    标准差: {metrics_norm['temporal_decay/weight_std']:.4f}")
        
        print(f"  非归一化版本:")
        print(f"    权重: {decay_weights_raw.tolist()}")
        print(f"    总和: {metrics_raw['temporal_decay/weight_sum']:.4f}")
        print(f"    均值: {metrics_raw['temporal_decay/weight_mean']:.4f}")
        print(f"    标准差: {metrics_raw['temporal_decay/weight_std']:.4f}")
        
        print()

def test_expected_behavior():
    """测试预期的衰减行为"""
    print("🎯 验证预期的衰减行为\n")
    
    # 对于γ=0.8, 序列长度=5的情况
    gamma = 0.8
    sequence_length = 5
    
    # 手动计算预期值
    expected_raw = [gamma**i for i in range(sequence_length)]
    expected_sum = sum(expected_raw)
    expected_normalized = [w/expected_sum for w in expected_raw]
    
    print(f"手动计算 (γ={gamma}, 长度={sequence_length}):")
    print(f"  原始权重: {expected_raw}")
    print(f"  归一化权重: {expected_normalized}")
    print(f"  归一化后均值: {np.mean(expected_normalized):.4f}")
    print()
    
    # 使用我们的函数计算
    decay_weights, metrics = apply_temporal_decay_weighting(
        sequence_length=sequence_length,
        gamma=gamma,
        normalize=True
    )
    
    print(f"函数计算结果:")
    print(f"  权重: {decay_weights.tolist()}")
    print(f"  均值: {metrics['temporal_decay/weight_mean']:.4f}")
    print()
    
    # 验证是否一致
    if np.allclose(decay_weights.numpy(), expected_normalized, atol=1e-6):
        print("✅ 计算结果正确！")
    else:
        print("❌ 计算结果有误！")

if __name__ == "__main__":
    test_temporal_decay_weights()
    test_expected_behavior()
    
    print("📋 总结:")
    print("  - 修复后，归一化版本的权重总和=1，均值<1")
    print("  - γ值越小，衰减效果越明显，均值越小")
    print("  - 第一个token权重最大，后续token权重递减")
    print("  - 现在应该能在WandB中看到正确的衰减效果了！")
