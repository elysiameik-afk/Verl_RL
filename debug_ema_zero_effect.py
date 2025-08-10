#!/usr/bin/env python3
"""
调试为什么EMA效果为0
"""

import numpy as np

def simulate_ema_effect():
    """模拟EMA效果"""
    print("🧪 模拟EMA效果...")
    
    # 模拟重要性权重（接近1.0但有变化）
    raw_weights_step1 = np.array([1.05, 0.98, 1.12, 0.95, 1.08])
    raw_weights_step2 = np.array([1.02, 1.05, 0.89, 1.15, 0.92])
    
    beta = 0.9
    
    print(f"步骤1原始权重: {raw_weights_step1}")
    print(f"步骤1权重方差: {np.var(raw_weights_step1):.6f}")
    
    # 第一步：初始化为当前权重（修复后的逻辑）
    prev_ema = raw_weights_step1.copy()  # 现在的初始化方式
    smoothed_step1 = beta * raw_weights_step1 + (1 - beta) * prev_ema
    
    print(f"步骤1平滑权重: {smoothed_step1}")
    print(f"步骤1差异: {np.linalg.norm(raw_weights_step1 - smoothed_step1):.6f}")
    
    # 第二步：使用前一步的平滑权重
    prev_ema = smoothed_step1
    smoothed_step2 = beta * raw_weights_step2 + (1 - beta) * prev_ema
    
    print(f"\n步骤2原始权重: {raw_weights_step2}")
    print(f"步骤2权重方差: {np.var(raw_weights_step2):.6f}")
    print(f"步骤2平滑权重: {smoothed_step2}")
    print(f"步骤2差异: {np.linalg.norm(raw_weights_step2 - smoothed_step2):.6f}")
    
    # 计算方差降低
    raw_var = np.var(raw_weights_step2)
    smoothed_var = np.var(smoothed_step2)
    variance_reduction = raw_var / (smoothed_var + 1e-8)
    smoothing_strength = 1.0 - (smoothed_var / (raw_var + 1e-8))
    
    print(f"\n📊 指标分析:")
    print(f"  variance_reduction_ratio: {variance_reduction:.6f}")
    print(f"  smoothing_strength: {smoothing_strength:.6f}")
    
    return variance_reduction, smoothing_strength

def analyze_zero_effect():
    """分析为什么效果为0"""
    print(f"\n🔍 分析零效果的可能原因:")
    
    # 情况1：权重完全相同
    identical_weights = np.array([1.0, 1.0, 1.0, 1.0])
    print(f"1. 权重完全相同: {identical_weights}")
    print(f"   方差: {np.var(identical_weights):.6f}")
    print(f"   → 如果重要性权重都接近1.0，方差本身就很小")
    
    # 情况2：每次重新初始化
    print(f"\n2. 每次重新初始化的情况:")
    weights = np.array([1.05, 0.98, 1.12, 0.95])
    beta = 0.9
    
    # 每次都用当前权重初始化（第一步没有平滑效果）
    smoothed = beta * weights + (1 - beta) * weights  # prev_ema = weights
    print(f"   原始: {weights}")
    print(f"   平滑: {smoothed}")
    print(f"   差异: {np.linalg.norm(weights - smoothed):.6f}")
    print(f"   → 第一步时差异为0是正常的！")
    
    # 情况3：序列ID一直在变化
    print(f"\n3. 序列ID一直变化的情况:")
    print(f"   如果每个micro-batch的序列ID都不同，")
    print(f"   EMA状态就无法累积，每次都是第一步")
    print(f"   → 这会导致平滑效果始终为0")

if __name__ == "__main__":
    print("🚀 开始分析EMA零效果问题...")
    
    try:
        variance_reduction, smoothing_strength = simulate_ema_effect()
        analyze_zero_effect()
        
        print(f"\n💡 可能的解决方案:")
        print(f"  1. 检查序列ID是否在不同步骤间保持一致")
        print(f"  2. 降低beta值（如0.7）以增加平滑效果")
        print(f"  3. 确认重要性权重确实有变化")
        print(f"  4. 检查EMA状态是否正确累积")
        
        print(f"\n🔧 调试建议:")
        print(f"  运行训练并观察调试输出中的:")
        print(f"  - seq_id是否保持一致")
        print(f"  - step_count是否递增")
        print(f"  - raw_weights是否有变化")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
