# EMA-GRPO 设置指南

## 🛠️ 配置文件更新

我已经在 `verl/trainer/config/ppo_trainer.yaml` 中添加了EMA相关配置：

```yaml
# 在 actor_rollout_ref.actor 部分添加了:
use_ema_smoothing: false  # 是否启用EMA平滑
ema_beta: 0.9            # EMA平滑因子β
```

## 🚀 运行步骤

### 1. 快速测试EMA功能
```bash
python quick_test_ema.py
```

### 2. 运行EMA训练
```bash
bash exp/qwen2.5kk1_ema.sh
```

### 3. 运行对比实验
```bash
bash exp/run_beta_comparison.sh
```

## 🔧 配置说明

现在你可以在训练脚本中使用以下配置：

```bash
# 启用EMA平滑
actor_rollout_ref.actor.use_ema_smoothing=True

# 设置平滑因子（推荐0.9）
actor_rollout_ref.actor.ema_beta=0.9
```

## 📊 预期效果

运行后你应该在WandB中看到这些指标：

### 核心指标
- `ema/raw_weights_variance` - 原始权重方差
- `ema/smoothed_weights_variance` - 平滑权重方差  
- `ema/variance_reduction_ratio` - 方差降低比例
- `ema/smoothing_strength` - 平滑强度

### 分析指标
- `ema/range_reduction` - 权重范围收缩
- `ema/weights_diff_l2` - L2差异
- `ema/relative_change_mean` - 相对变化

## 🐛 故障排除

### 如果遇到配置错误
1. 确认 `verl/trainer/config/ppo_trainer.yaml` 已更新
2. 检查配置语法是否正确
3. 运行 `python quick_test_ema.py` 验证功能

### 如果导入错误
确保你在正确的目录下运行，并且已经安装了所有依赖。

## 🎯 实验建议

1. **基线对比**: 先运行不启用EMA的版本作为基线
2. **β值测试**: 测试不同的β值 (0.5, 0.7, 0.9, 0.95, 0.99)
3. **方差分析**: 重点关注方差降低效果
4. **收敛分析**: 对比收敛速度和稳定性

现在配置已经正确设置，你可以开始实验了！🚀
