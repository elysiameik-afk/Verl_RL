#!/usr/bin/env python3
"""
测试HVR Manager的集成实现

验证HVR Logic RL Manager的正确性和完整性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from verl.trainer.ppo.core_algos import (
    calculate_ervf_value,
    calculate_hvr_rewards_for_group,
    aggregate_hvr_metrics_dict
)

# 测试HVR Manager
def test_hvr_manager():
    """测试HVR Manager的基本功能"""
    print("🎯 测试HVR Manager\n")

    try:
        from verl.trainer.ppo.reward_manager.hvr_logic_rl_reward import HVRLogicRLRewardManager

        # 创建HVR Manager
        manager = HVRLogicRLRewardManager(
            tokenizer=None,  # 简化测试
            num_examine=1,
            hvr_alpha=1.0,
            hvr_beta=0.1,
            hvr_lambda=0.5
        )

        print("✅ HVR Manager创建成功")
        print(f"   参数: α={manager.hvr_alpha}, β={manager.hvr_beta}, λ={manager.hvr_lambda}")

        return True

    except Exception as e:
        print(f"❌ HVR Manager测试失败: {e}")
        return False

def test_ervf_value_function():
    """测试ERVF价值函数"""
    print("🔬 测试ERVF (熵正则化价值函数)\n")
    
    # 创建测试logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    
    # 测试不同参数组合
    test_cases = [
        (1.0, 0.0, "无熵惩罚"),
        (1.0, 0.1, "轻微熵惩罚"),
        (1.0, 0.5, "中等熵惩罚"),
        (0.5, 0.1, "低温度"),
        (2.0, 0.1, "高温度"),
    ]
    
    for alpha, beta, description in test_cases:
        v_ervf, entropy = calculate_ervf_value(logits, alpha, beta)
        
        # 手动验证计算
        v_endo = alpha * torch.logsumexp(logits / alpha, dim=0).item()
        expected_v_ervf = v_endo - beta * entropy
        
        print(f"📊 {description} (α={alpha}, β={beta}):")
        print(f"  V_ERVF: {v_ervf:.4f}")
        print(f"  V_endo: {v_endo:.4f}")
        print(f"  熵: {entropy:.4f}")
        print(f"  验证: {'✅' if abs(v_ervf - expected_v_ervf) < 1e-6 else '❌'}")
        print()

def test_hvr_group_processing():
    """测试HVR组处理 (GRPO集成版本)"""
    print("🎯 测试HVR组处理 (GRPO集成版本)\n")

    # 创建组数据 (模拟GRPO的一个组)
    group_size = 4
    group_data = []

    # 测试logic_rl的典型奖励值
    logic_rl_rewards = [-3, -1, 1, 3]

    for i in range(group_size):
        seq_len = np.random.randint(8, 16)  # 随机序列长度
        vocab_size = 1000

        # 创建单个序列数据
        logits = torch.randn(seq_len, vocab_size)
        ids = torch.randint(0, vocab_size, (seq_len,))
        r_final = logic_rl_rewards[i]  # 使用不同的奖励

        group_data.append({
            'logits': logits,
            'ids': ids,
            'r_final': r_final
        })

    print(f"📊 组数据: {group_size} 个序列")
    print(f"   稀疏奖励: {[d['r_final'] for d in group_data]}")

    # 计算HVR组回报
    group_returns, hvr_metrics = calculate_hvr_rewards_for_group(
        group_data=group_data,
        alpha=1.0,
        beta=0.1,
        lambda_hvr=0.5
    )

    # 计算GRPO优势
    mean_return = sum(group_returns) / len(group_returns)
    grpo_advantages = [ret - mean_return for ret in group_returns]

    print(f"\n✅ HVR组处理结果:")
    print(f"   组回报: {[f'{ret:.4f}' for ret in group_returns]}")
    print(f"   平均回报: {mean_return:.4f}")
    print(f"   GRPO优势: {[f'{adv:.4f}' for adv in grpo_advantages]}")

    # 聚合指标
    aggregated_metrics = aggregate_hvr_metrics_dict(hvr_metrics)
    print(f"\n📊 HVR指标:")
    for key, value in aggregated_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print()

def test_hvr_parameter_sensitivity():
    """测试HVR参数敏感性"""
    print("🎛️ 测试HVR参数敏感性\n")
    
    # 固定测试数据
    seq_len = 10
    vocab_size = 50
    response_logits = torch.randn(seq_len, vocab_size)
    response_ids = torch.randint(0, vocab_size, (seq_len,))
    r_final = 1.0
    
    # 测试α参数影响
    print("📈 α (温度系数) 参数影响:")
    alphas = [0.5, 1.0, 2.0, 4.0]
    # 需要使用组数据格式
    group_data = [{'logits': response_logits, 'ids': response_ids, 'r_final': r_final}]

    for alpha in alphas:
        group_returns, metrics = calculate_hvr_rewards_for_group(
            group_data, alpha=alpha, beta=0.1, lambda_hvr=0.5
        )
        print(f"  α={alpha}: 组回报={group_returns[0]:.4f}")
    
    print("\n📈 β (熵惩罚) 参数影响:")
    betas = [0.0, 0.05, 0.1, 0.2, 0.5]
    for beta in betas:
        group_returns, metrics = calculate_hvr_rewards_for_group(
            group_data, alpha=1.0, beta=beta, lambda_hvr=0.5
        )
        print(f"  β={beta}: 组回报={group_returns[0]:.4f}")

    print("\n📈 λ (混合因子) 参数影响:")
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    for lambda_hvr in lambdas:
        group_returns, metrics = calculate_hvr_rewards_for_group(
            group_data, alpha=1.0, beta=0.1, lambda_hvr=lambda_hvr
        )
        print(f"  λ={lambda_hvr}: 组回报={group_returns[0]:.4f}")

# 移除不再需要的测试函数

def test_edge_cases():
    """测试边界情况"""
    print("\n🔍 测试边界情况\n")
    
    # 测试极短序列
    print("📏 极短序列 (长度=1):")
    short_group_data = [{'logits': torch.randn(1, 100), 'ids': torch.randint(0, 100, (1,)), 'r_final': 1.0}]
    short_returns, short_metrics = calculate_hvr_rewards_for_group(short_group_data, 1.0, 0.1, 0.5)
    print(f"  组回报: {short_returns}")
    print(f"  成功率: {short_metrics.get('successful_count', 0) / short_metrics.get('total_count', 1)}")

    # 测试极端R_final值
    print("\n📊 极端R_final值:")
    normal_logits = torch.randn(3, 100)
    normal_ids = torch.randint(0, 100, (3,))

    extreme_r_finals = [-3.0, 3.0]
    for r_final in extreme_r_finals:
        test_group_data = [{'logits': normal_logits, 'ids': normal_ids, 'r_final': r_final}]
        returns, _ = calculate_hvr_rewards_for_group(test_group_data, 1.0, 0.1, 0.5)
        print(f"  R_final={r_final}: 组回报={returns[0]:.3f}")

    # 测试数值稳定性
    print("\n🔢 数值稳定性测试:")
    large_logits = torch.randn(3, 100) * 10  # 大logits
    large_group_data = [{'logits': large_logits, 'ids': normal_ids, 'r_final': 0.0}]
    large_returns, _ = calculate_hvr_rewards_for_group(large_group_data, 1.0, 0.1, 0.5)
    print(f"  大logits: NaN检查={'❌' if any(np.isnan(large_returns)) else '✅'}")
    print(f"  大logits: Inf检查={'❌' if any(np.isinf(large_returns)) else '✅'}")

def visualize_hvr_comparison():
    """可视化HVR vs 标准方法对比"""
    try:
        print("\n📊 生成HVR对比可视化...\n")
        
        seq_len = 20
        vocab_size = 100
        response_logits = torch.randn(seq_len, vocab_size)
        response_ids = torch.randint(0, vocab_size, (seq_len,))
        
        # 不同λ值的HVR奖励
        lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        plt.figure(figsize=(12, 8))
        
        test_group_data = [{'logits': response_logits, 'ids': response_ids, 'r_final': 1.0}]

        for i, lambda_hvr in enumerate(lambdas):
            group_returns, _ = calculate_hvr_rewards_for_group(
                test_group_data, alpha=1.0, beta=0.1, lambda_hvr=lambda_hvr
            )

            plt.subplot(2, 3, i + 1)
            plt.bar(0, group_returns[0])
            plt.title(f'λ = {lambda_hvr}')
            plt.xlabel('序列')
            plt.ylabel('组回报')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hvr_comparison.png', dpi=150, bbox_inches='tight')
        print("  可视化图表已保存为 hvr_comparison.png")
        
    except ImportError:
        print("  matplotlib未安装，跳过可视化")

if __name__ == "__main__":
    print("🚀 开始测试HVR纯净算法实现\n")
    
    test_ervf_value_function()
    test_hvr_group_processing()
    test_hvr_parameter_sensitivity()
    test_edge_cases()
    visualize_hvr_comparison()
    
    print("🎉 HVR-GRPO集成算法测试完成！")
    print("\n📋 HVR在GRPO框架中的特性总结:")
    print("  ✅ ERVF价值函数: 基于logits的内生价值 + 熵正则化")
    print("  ✅ HVR奖励重塑: 稀疏奖励指导的价值轨迹重塑")
    print("  ✅ GRPO组间投票: 保留组内相对优势计算")
    print("  ✅ 无需critic: 完全基于模型自身的内生价值估计")
    print("  ✅ Logic RL兼容: 支持{-3,-1,0,1,3}等稀疏奖励")
    print("  ✅ 参数可控: α控制温度，β控制熵惩罚，λ控制重塑强度")
    print("  ✅ 数值稳定: 使用log_softmax等稳定计算")
    print("\n🎯 HVR-GRPO集成内生奖励系统已准备就绪！")
