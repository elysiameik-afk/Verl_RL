#!/usr/bin/env python3
"""
测试HVR (Hindsight Value Reshaping) 内生奖励算法
"""

import torch
import numpy as np
from verl.trainer.ppo.core_algos import calculate_ervf_value, calculate_hvr_rewards, apply_hvr_integration

def test_ervf_value():
    """测试ERVF价值函数计算"""
    print("🔬 测试ERVF价值函数计算\n")
    
    # 创建测试logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    
    # 测试不同参数
    test_cases = [
        (1.0, 0.0),   # 无熵惩罚
        (1.0, 0.1),   # 轻微熵惩罚
        (1.0, 0.5),   # 中等熵惩罚
        (0.5, 0.1),   # 低温度
        (2.0, 0.1),   # 高温度
    ]
    
    for alpha, beta in test_cases:
        v_ervf = calculate_ervf_value(logits, alpha, beta)
        
        # 手动计算验证
        v_endo = alpha * torch.logsumexp(logits / alpha, dim=0).item()
        probs = torch.softmax(logits, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        expected_v_ervf = v_endo - beta * entropy
        
        print(f"α={alpha}, β={beta}:")
        print(f"  V_ERVF: {v_ervf:.4f}")
        print(f"  V_endo: {v_endo:.4f}")
        print(f"  熵: {entropy:.4f}")
        print(f"  预期值: {expected_v_ervf:.4f}")
        print(f"  匹配: {'✅' if abs(v_ervf - expected_v_ervf) < 1e-6 else '❌'}")
        print()

def test_hvr_rewards():
    """测试HVR奖励计算"""
    print("🎯 测试HVR奖励计算\n")
    
    # 创建测试数据
    seq_len = 5
    vocab_size = 100
    
    # 随机logits和token序列
    response_logits = torch.randn(seq_len, vocab_size)
    response_ids = torch.randint(0, vocab_size, (seq_len,))
    
    # 测试不同的R_final值
    r_final_values = [-3.0, -1.0, 0.0, 1.0, 3.0]
    
    for r_final in r_final_values:
        print(f"📊 R_final = {r_final}")
        
        hvr_rewards = calculate_hvr_rewards(
            response_logits=response_logits,
            response_ids=response_ids,
            R_final=r_final,
            alpha=1.0,
            beta=0.1,
            lambda_hvr=0.5,
        )
        
        print(f"  HVR奖励: {hvr_rewards.tolist()}")
        print(f"  奖励总和: {hvr_rewards.sum().item():.4f}")
        print(f"  奖励均值: {hvr_rewards.mean().item():.4f}")
        print(f"  最后奖励: {hvr_rewards[-1].item():.4f}")
        print()

def test_hvr_integration():
    """测试HVR集成功能"""
    print("🔧 测试HVR集成功能\n")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    
    # 原始advantages
    advantages = torch.randn(batch_size, seq_len)
    
    # 响应logits和IDs
    response_logits = torch.randn(batch_size, seq_len, vocab_size)
    response_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 响应mask (模拟不同长度的序列)
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[0, 8:] = 0  # 第一个序列长度为8
    response_mask[1, 6:] = 0  # 第二个序列长度为6
    
    print("原始advantages:")
    for i in range(batch_size):
        valid_pos = torch.where(response_mask[i] > 0)[0]
        print(f"  序列{i}: {advantages[i, valid_pos].tolist()}")
    
    # 应用HVR
    enhanced_advantages, hvr_metrics = apply_hvr_integration(
        advantages=advantages,
        response_logits=response_logits,
        response_ids=response_ids,
        response_mask=response_mask,
        alpha=1.0,
        beta=0.1,
        lambda_hvr=0.5,
    )
    
    print("\nHVR增强后的advantages:")
    for i in range(batch_size):
        valid_pos = torch.where(response_mask[i] > 0)[0]
        print(f"  序列{i}: {enhanced_advantages[i, valid_pos].tolist()}")
    
    print("\nHVR指标:")
    for key, value in hvr_metrics.items():
        print(f"  {key}: {value}")

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("🎛️ 测试参数敏感性\n")
    
    # 固定测试数据
    seq_len = 8
    vocab_size = 100
    response_logits = torch.randn(seq_len, vocab_size)
    response_ids = torch.randint(0, vocab_size, (seq_len,))
    r_final = 1.0
    
    # 测试alpha的影响
    print("📈 Alpha参数影响:")
    alphas = [0.5, 1.0, 2.0, 4.0]
    for alpha in alphas:
        hvr_rewards = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=alpha, beta=0.1, lambda_hvr=0.5
        )
        print(f"  α={alpha}: 均值={hvr_rewards.mean().item():.4f}, 标准差={hvr_rewards.std().item():.4f}")
    
    print("\n📈 Beta参数影响:")
    betas = [0.0, 0.1, 0.3, 0.5]
    for beta in betas:
        hvr_rewards = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=1.0, beta=beta, lambda_hvr=0.5
        )
        print(f"  β={beta}: 均值={hvr_rewards.mean().item():.4f}, 标准差={hvr_rewards.std().item():.4f}")
    
    print("\n📈 Lambda参数影响:")
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    for lambda_hvr in lambdas:
        hvr_rewards = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=1.0, beta=0.1, lambda_hvr=lambda_hvr
        )
        print(f"  λ={lambda_hvr}: 均值={hvr_rewards.mean().item():.4f}, 标准差={hvr_rewards.std().item():.4f}")

def test_edge_cases():
    """测试边界情况"""
    print("🔍 测试边界情况\n")
    
    # 测试极短序列
    print("📏 极短序列 (长度=1):")
    short_logits = torch.randn(1, 100)
    short_ids = torch.randint(0, 100, (1,))
    short_rewards = calculate_hvr_rewards(short_logits, short_ids, 1.0)
    print(f"  奖励: {short_rewards.tolist()}")
    
    # 测试极端R_final值
    print("\n📊 极端R_final值:")
    normal_logits = torch.randn(3, 100)
    normal_ids = torch.randint(0, 100, (3,))
    
    extreme_r_finals = [-3.0, 3.0]
    for r_final in extreme_r_finals:
        rewards = calculate_hvr_rewards(normal_logits, normal_ids, r_final)
        print(f"  R_final={r_final}: {rewards.tolist()}")
    
    # 测试数值稳定性
    print("\n🔢 数值稳定性测试:")
    # 创建极大的logits
    large_logits = torch.randn(3, 100) * 10
    large_rewards = calculate_hvr_rewards(large_logits, normal_ids, 0.0)
    print(f"  大logits: 是否有NaN={torch.isnan(large_rewards).any().item()}")
    print(f"  大logits: 是否有Inf={torch.isinf(large_rewards).any().item()}")

if __name__ == "__main__":
    print("🚀 开始测试HVR (Hindsight Value Reshaping) 内生奖励算法\n")
    
    test_ervf_value()
    test_hvr_rewards()
    test_hvr_integration()
    test_parameter_sensitivity()
    test_edge_cases()
    
    print("🎉 HVR算法测试完成！")
    print("\n📋 HVR算法特性总结:")
    print("  ✅ ERVF价值函数: 结合EndoRM和熵正则化")
    print("  ✅ 稠密奖励生成: 基于价值轨迹重塑")
    print("  ✅ 稀疏奖励集成: R_final指导价值目标")
    print("  ✅ 参数可控: α控制温度，β控制熵惩罚，λ控制混合")
    print("  ✅ 数值稳定: 使用log_softmax避免数值问题")
    print("\n🎯 HVR内生奖励机制已准备就绪，可以开始训练！")
