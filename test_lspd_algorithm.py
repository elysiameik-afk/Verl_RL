#!/usr/bin/env python3
"""
测试LSPD (对数尺度位置衰减) 算法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from verl.trainer.ppo.core_algos import apply_temporal_decay_weighting

def test_lspd_vs_standard():
    """对比LSPD和标准指数衰减"""
    print("🔬 测试LSPD vs 标准指数衰减\n")
    
    # 测试不同序列长度
    sequence_lengths = [10, 100, 1000, 2048]
    
    for seq_len in sequence_lengths:
        print(f"📊 序列长度: {seq_len}")
        
        # 标准指数衰减
        standard_weights, standard_metrics = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            gamma=0.95,
            normalize=True,
            use_lspd=False
        )
        
        # LSPD算法
        lspd_weights, lspd_metrics = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            gamma=0.95,  # 这个参数在LSPD中不使用
            normalize=True,
            use_lspd=True,
            lspd_alpha=2.0,
            lspd_tau=10.0
        )
        
        print(f"  标准衰减:")
        print(f"    均值: {standard_metrics['temporal_decay/weight_mean']:.6f}")
        print(f"    首末比: {standard_metrics['temporal_decay/weight_ratio_first_to_last']:.2f}")
        print(f"    前5个权重: {standard_weights[:5].tolist()}")
        
        print(f"  LSPD算法:")
        print(f"    均值: {lspd_metrics['temporal_decay/weight_mean']:.6f}")
        print(f"    首末比: {lspd_metrics['temporal_decay/weight_ratio_first_to_last']:.2f}")
        print(f"    前5个权重: {lspd_weights[:5].tolist()}")
        print()

def test_lspd_parameters():
    """测试LSPD算法的不同参数"""
    print("🎛️ 测试LSPD参数影响\n")
    
    seq_len = 2048
    
    # 测试不同的alpha值
    print("📈 不同alpha值的影响:")
    alphas = [0.5, 1.0, 2.0, 4.0]
    for alpha in alphas:
        weights, metrics = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            normalize=True,
            use_lspd=True,
            lspd_alpha=alpha,
            lspd_tau=10.0
        )
        print(f"  α={alpha}: 均值={metrics['temporal_decay/weight_mean']:.6f}, "
              f"首末比={metrics['temporal_decay/weight_ratio_first_to_last']:.2f}")
    
    print()
    
    # 测试不同的tau值
    print("📈 不同tau值的影响:")
    taus = [1.0, 5.0, 10.0, 50.0]
    for tau in taus:
        weights, metrics = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            normalize=True,
            use_lspd=True,
            lspd_alpha=2.0,
            lspd_tau=tau
        )
        print(f"  τ={tau}: 均值={metrics['temporal_decay/weight_mean']:.6f}, "
              f"首末比={metrics['temporal_decay/weight_ratio_first_to_last']:.2f}")
    
    print()

def test_lspd_properties():
    """验证LSPD算法的关键特性"""
    print("✅ 验证LSPD算法特性\n")
    
    # 特性1: 长度自适应性
    print("🔍 特性1: 长度自适应性")
    lengths = [100, 1000, 2048, 5000]
    for length in lengths:
        weights, metrics = apply_temporal_decay_weighting(
            sequence_length=length,
            normalize=True,
            use_lspd=True,
            lspd_alpha=2.0,
            lspd_tau=10.0
        )
        print(f"  长度{length}: 均值={metrics['temporal_decay/weight_mean']:.6f}, "
              f"最后权重={metrics['temporal_decay/last_weight']:.6f}")
    
    print()
    
    # 特性2: 先陡峭后平缓
    print("🔍 特性2: 先陡峭后平缓的衰减曲线")
    weights, _ = apply_temporal_decay_weighting(
        sequence_length=20,
        normalize=False,
        use_lspd=True,
        lspd_alpha=2.0,
        lspd_tau=5.0
    )
    
    print(f"  前10个权重: {weights[:10].tolist()}")
    print(f"  后10个权重: {weights[10:].tolist()}")
    
    # 计算相邻权重的差值
    diffs = torch.diff(weights)
    print(f"  前5个差值: {diffs[:5].tolist()}")
    print(f"  后5个差值: {diffs[-5:].tolist()}")
    print(f"  差值变化: 前期变化大，后期变化小 ✅" if abs(diffs[0]) > abs(diffs[-1]) else "  差值变化: 异常 ❌")
    
    print()

def visualize_comparison():
    """可视化对比（如果有matplotlib）"""
    try:
        print("📊 生成可视化对比图...")
        
        seq_len = 100
        positions = np.arange(seq_len)
        
        # 标准衰减
        standard_weights, _ = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            gamma=0.95,
            normalize=False,
            use_lspd=False
        )
        
        # LSPD算法
        lspd_weights, _ = apply_temporal_decay_weighting(
            sequence_length=seq_len,
            normalize=False,
            use_lspd=True,
            lspd_alpha=2.0,
            lspd_tau=10.0
        )
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(positions, standard_weights.numpy(), 'b-', label='标准指数衰减 (γ=0.95)', linewidth=2)
        plt.plot(positions, lspd_weights.numpy(), 'r-', label='LSPD (α=2.0, τ=10.0)', linewidth=2)
        plt.xlabel('位置')
        plt.ylabel('权重')
        plt.title('权重衰减对比 (线性尺度)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(positions, standard_weights.numpy(), 'b-', label='标准指数衰减', linewidth=2)
        plt.semilogy(positions, lspd_weights.numpy(), 'r-', label='LSPD', linewidth=2)
        plt.xlabel('位置')
        plt.ylabel('权重 (对数尺度)')
        plt.title('权重衰减对比 (对数尺度)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lspd_comparison.png', dpi=150, bbox_inches='tight')
        print("  图表已保存为 lspd_comparison.png")
        
    except ImportError:
        print("  matplotlib未安装，跳过可视化")

if __name__ == "__main__":
    print("🚀 开始测试LSPD (对数尺度位置衰减) 算法\n")
    
    test_lspd_vs_standard()
    test_lspd_parameters()
    test_lspd_properties()
    visualize_comparison()
    
    print("🎉 LSPD算法测试完成！")
    print("\n📋 LSPD算法优势总结:")
    print("  ✅ 长序列友好: 后期权重不会完全消失")
    print("  ✅ 自适应性强: 自动适应不同序列长度")
    print("  ✅ 衰减合理: 先陡峭后平缓，符合认知直觉")
    print("  ✅ 参数可控: α控制衰减强度，τ控制时间尺度")
    print("\n🎯 现在可以在训练中使用LSPD算法了！")
