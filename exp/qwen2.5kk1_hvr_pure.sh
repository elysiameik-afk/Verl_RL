#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "🎯 开始HVR纯净测试: 内生奖励机制 (Hindsight Value Reshaping)..."
echo "🎯 [HVR特性] ERVF熵正则化价值函数 + 后见之明价值重塑"
echo "🎯 [HVR优势] 无需critic网络，基于模型自身logits的内生价值估计"

# 使用简化的HVR启动脚本
python3 run_hvr_simple.py
# 所有配置都在run_hvr_simple.py中定义

echo "🎉 HVR纯净测试完成！"
echo "📊 [HVR结果] 请查看WandB中的HVR专用指标："
echo "  - hvr/ervf_value_mean: ERVF价值函数均值"
echo "  - hvr/entropy_mean: 策略熵均值"
echo "  - hvr/hvr_reward_mean: HVR奖励均值"
echo "  - hvr/r_final_distribution: 稀疏奖励分布"
echo "  - hvr/success_rate: HVR处理成功率"
