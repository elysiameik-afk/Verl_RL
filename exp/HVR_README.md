# HVR (Hybrid Value Reshaping) 实验指南

## 概述

本目录包含HVR (混合价值重塑) 创新点的实验脚本和说明。HVR是一个基于GRPO框架的新奖励模块，通过融合模型内生价值和外部稀疏奖励，生成稠密、连续、高分辨率的总回报信号。

## 核心创新

### 1. ERVF (熵正则化价值函数)
- 基于logits计算模型的内生价值
- 公式：`V_ervf = V_endo - β * H`
- 其中：`V_endo = α * logsumexp(logits/α)`

### 2. HVR (混合价值重塑)
- 使用外部奖励指导内生价值轨迹重塑
- 公式：`V_hvr = (1-λ) * V_ervf + λ * V_target`
- 分解为稠密奖励：`r_hvr_t = α * log_prob_t + V_hvr[t] - V_hvr[t+1]`

### 3. Z-score归一化
- 序列级和组级双重归一化
- 保持数值稳定性，控制奖励范围在[-6, 6]

## 实验脚本

### HVR实验
```bash
bash exp/qwen2.5kk1_hvr.sh
```

**关键配置：**
- `reward_model.reward_manager=hvr_logic_rl`
- `reward_model.alpha=1.0` (温度系数)
- `reward_model.beta=0.1` (熵惩罚权重)
- `reward_model.lambda_hvr=0.5` (重塑强度)
- `reward_model.use_zscore=True` (启用Z-score归一化)
- `reward_model.target_scale=3.0` (目标标准差)

### Baseline对比实验
```bash
bash exp/qwen2.5kk1_baseline.sh
```

**关键配置：**
- `reward_model.reward_manager=logic_rl` (原始LogicRL)
- 其他配置与HVR实验相同，确保公平对比

## 监控指标

### WandB记录的关键指标

#### HVR模块内部指标 (最重要)
- `rewards/v_ervf_mean`: 内生价值V_ervf的平均值
- `rewards/v_target_mean`: 目标价值V_target的平均值  
- `rewards/v_hvr_mean`: 重塑后价值V_hvr的平均值

#### 奖励信号质量
- `rewards/raw_return_mean`: 归一化前原始总回报均值
- `rewards/raw_return_std`: 归一化前原始总回报标准差
- `rewards/final_return_mean`: 最终精炼后总回报均值
- `rewards/r_final_mean`: 外部稀疏奖励R_final均值
- `rewards/external_score_mean`: 原始外部分数均值 (与其他算法对比的关键指标)

### 论文图表建议

1. **内生价值演化图**: `v_ervf_mean` vs 训练步数
2. **目标价值对比图**: `v_target_mean` vs `v_ervf_mean` vs 训练步数
3. **重塑效果图**: `v_hvr_mean` 介于前两者之间的演化
4. **奖励稳定性图**: `raw_return_std` vs 训练步数
5. **最终性能图**: `external_score_mean` vs 训练步数 (核心结果)

## 实验流程

### 1. 环境准备
确保已安装所有依赖，特别是修改后的verl库。

### 2. 运行实验
```bash
# 先运行baseline获得对比基线
bash exp/qwen2.5kk1_baseline.sh

# 再运行HVR实验
bash exp/qwen2.5kk1_hvr.sh
```

### 3. 监控训练
- 观察控制台输出的HVR指标
- 在WandB中实时监控关键指标
- 特别关注`external_score_mean`的提升

### 4. 结果分析
- 对比HVR vs Baseline的`external_score_mean`
- 分析内生价值的演化趋势
- 验证奖励信号的稳定性

## 超参数调优建议

### 核心超参数
- `alpha`: 控制内生价值的温度，建议范围[0.5, 2.0]
- `beta`: 控制熵惩罚强度，建议范围[0.05, 0.2]  
- `lambda_hvr`: 控制重塑强度，建议范围[0.3, 0.7]

### 数值稳定性
- `use_zscore=True`: 强烈建议保持开启
- `target_scale=3.0`: 可根据实际情况调整到[2.0, 5.0]

## 故障排除

### 常见问题
1. **未找到logits数据**: 确保actor支持`return_logits=True`
2. **数值异常**: 检查Z-score归一化是否正常工作
3. **性能下降**: 尝试调小`lambda_hvr`或`beta`

### 调试信息
HVR会在控制台输出详细的调试信息：
- `🎯 [HVR初始化]`: 确认参数设置
- `🎯 [HVR指标]`: 实时监控关键数值
- `⚠️ [HVR警告]`: 异常情况提醒

## 预期结果

### 成功指标
1. `external_score_mean` 相比baseline有显著提升
2. `v_ervf_mean` 随训练步数稳步上升
3. `raw_return_std` 保持在合理范围内
4. 训练过程稳定，无数值异常

### 论文贡献
- 证明内生价值函数的有效性
- 展示混合价值重塑的优势
- 提供稠密奖励信号的新方法
- 在KK任务上获得SOTA结果
