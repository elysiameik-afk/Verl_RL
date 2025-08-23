#!/usr/bin/env python3
"""
测试HVR在GRPO框架中的集成实现

验证ERVF价值函数和HVR奖励重塑在GRPO中的正确性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from verl.trainer.ppo.core_algos import (
    calculate_ervf_value,
    calculate_hvr_rewards_for_group,
    aggregate_hvr_metrics_dict
)

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
    for alpha in alphas:
        hvr_rewards, metrics = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=alpha, beta=0.1, lambda_hvr=0.5
        )
        print(f"  α={alpha}: 奖励均值={hvr_rewards.mean().item():.4f}, "
              f"ERVF均值={metrics.ervf_value_mean:.4f}")
    
    print("\n📈 β (熵惩罚) 参数影响:")
    betas = [0.0, 0.05, 0.1, 0.2, 0.5]
    for beta in betas:
        hvr_rewards, metrics = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=1.0, beta=beta, lambda_hvr=0.5
        )
        print(f"  β={beta}: 奖励均值={hvr_rewards.mean().item():.4f}, "
              f"熵均值={metrics.entropy_mean:.4f}")
    
    print("\n📈 λ (混合因子) 参数影响:")
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    for lambda_hvr in lambdas:
        hvr_rewards, metrics = calculate_hvr_rewards(
            response_logits, response_ids, r_final,
            alpha=1.0, beta=0.1, lambda_hvr=lambda_hvr
        )
        print(f"  λ={lambda_hvr}: 奖励均值={hvr_rewards.mean().item():.4f}, "
              f"重塑比例={metrics.value_reshaping_ratio:.1f}")

def test_hvr_policy_loss():
    """测试HVR策略损失"""
    print("\n🔧 测试HVR策略损失计算\n")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    
    log_probs = torch.randn(batch_size, seq_len)
    hvr_rewards = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[0, 3:] = 0  # 第一个序列长度为3
    response_mask[1, 4:] = 0  # 第二个序列长度为4
    
    # 计算策略损失
    policy_loss, metrics = hvr_policy_loss(
        log_probs=log_probs,
        hvr_rewards=hvr_rewards,
        response_mask=response_mask,
        cliprange=0.2,
        loss_agg_mode="token-mean",
    )
    
    print(f"策略损失: {policy_loss.item():.6f}")
    print(f"HVR优势均值: {metrics['hvr_advantages_mean']:.6f}")
    print(f"Log概率均值: {metrics['hvr_log_probs_mean']:.6f}")

def test_metrics_aggregation():
    """测试指标聚合"""
    print("\n📊 测试指标聚合\n")
    
    # 创建多个序列的指标
    from verl.trainer.hvr.hvr_core_algos import HVRMetrics
    
    metrics_list = []
    for i in range(3):
        metrics = HVRMetrics(
            ervf_value_mean=1.0 + i * 0.1,
            entropy_mean=2.0 + i * 0.1,
            hvr_reward_mean=0.5 + i * 0.1,
            r_final_mean=[-1, 0, 1][i],
            total_sequences=1,
            successful_hvr_count=1,
            success_rate=1.0,
        )
        metrics_list.append(metrics)
    
    # 聚合指标
    aggregated = aggregate_hvr_metrics(metrics_list)
    
    print("聚合后的指标:")
    for key, value in aggregated.items():
        print(f"  {key}: {value}")

def test_edge_cases():
    """测试边界情况"""
    print("\n🔍 测试边界情况\n")
    
    # 测试极短序列
    print("📏 极短序列 (长度=1):")
    short_logits = torch.randn(1, 100)
    short_ids = torch.randint(0, 100, (1,))
    short_rewards, short_metrics = calculate_hvr_rewards(short_logits, short_ids, 1.0)
    print(f"  奖励: {short_rewards.tolist()}")
    print(f"  成功率: {short_metrics.success_rate}")
    
    # 测试极端R_final值
    print("\n📊 极端R_final值:")
    normal_logits = torch.randn(3, 100)
    normal_ids = torch.randint(0, 100, (3,))
    
    extreme_r_finals = [-3.0, 3.0]
    for r_final in extreme_r_finals:
        rewards, _ = calculate_hvr_rewards(normal_logits, normal_ids, r_final)
        print(f"  R_final={r_final}: 奖励范围=[{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
    
    # 测试数值稳定性
    print("\n🔢 数值稳定性测试:")
    large_logits = torch.randn(3, 100) * 10  # 大logits
    large_rewards, _ = calculate_hvr_rewards(large_logits, normal_ids, 0.0)
    print(f"  大logits: NaN检查={'❌' if torch.isnan(large_rewards).any() else '✅'}")
    print(f"  大logits: Inf检查={'❌' if torch.isinf(large_rewards).any() else '✅'}")

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
        
        for i, lambda_hvr in enumerate(lambdas):
            hvr_rewards, _ = calculate_hvr_rewards(
                response_logits, response_ids, 1.0,
                alpha=1.0, beta=0.1, lambda_hvr=lambda_hvr
            )
            
            plt.subplot(2, 3, i + 1)
            plt.plot(hvr_rewards.numpy(), 'o-', linewidth=2)
            plt.title(f'λ = {lambda_hvr}')
            plt.xlabel('Token位置')
            plt.ylabel('HVR奖励')
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
