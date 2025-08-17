# GRPO 六大创新点实现指南

## 📋 创新点概览

本项目实现了六个针对GRPO算法的创新点，分为三个独立的研究模块：

### 模块A: 重要性权重计算 (方差缩减)
- **创新点 2.1**: 时序平滑 (EMA) 的重要性权重
- **创新点 2.3**: 算术平均重要性校正 (AMIC)

### 模块B: 信用分配 (Credit Assignment)
- **创新点 2.2**: 梯度自适应重要性加权
- **创新点 2.5**: 基于时序衰减的优势塑造

### 模块C: 信任区域目标函数 (Trust Region)
- **创新点 2.4**: 概率性信任区域加权 (PTRW)
- **创新点 2.6**: 正负优势的非对称策略优化

## 🚀 快速开始

### 1. 测试所有创新点
```bash
python test_all_innovations.py
```

### 2. 单独测试每个创新点

#### 创新点 2.1: EMA时序平滑
```bash
bash exp/qwen2.5kk1_token_ema.sh
```

#### 创新点 2.2: 梯度自适应加权
```bash
bash exp/qwen2.5kk1_innovation_2_2.sh
```

#### 创新点 2.3: AMIC算术平均
```bash
bash exp/qwen2.5kk1_innovation_2_3.sh
```

#### 创新点 2.4: PTRW概率性信任区域
```bash
bash exp/qwen2.5kk1_innovation_2_4.sh
```

#### 创新点 2.5: 时序衰减
```bash
bash exp/qwen2.5kk1_innovation_2_5.sh
```

#### 创新点 2.6: 非对称裁剪
```bash
bash exp/qwen2.5kk1_innovation_2_6.sh
```

### 3. 组合多个创新点
```bash
bash exp/qwen2.5kk1_all_innovations.sh
```

## ⚙️ 配置参数

### 创新点 2.1: EMA时序平滑
```yaml
use_ema_smoothing: True
ema_beta: 0.9  # 平滑因子 β ∈ (0, 1]
```

### 创新点 2.2: 梯度自适应加权
```yaml
use_gradient_adaptive_weighting: True
gradient_weighting_temperature: 1.0  # softmax温度参数
```

### 创新点 2.3: AMIC算术平均
```yaml
use_amic: True  # 与GSPO互斥
```

### 创新点 2.4: PTRW概率性信任区域
```yaml
use_ptrw: True  # 与标准裁剪互斥
ptrw_sigma: 0.2  # 高斯信任区域宽度
```

### 创新点 2.5: 时序衰减
```yaml
use_temporal_decay: True
temporal_decay_gamma: 0.95  # 衰减因子 γ ∈ (0, 1]
temporal_decay_normalize: True  # 是否归一化
```

### 创新点 2.6: 非对称裁剪
```yaml
use_asymmetric_clipping: True  # 与PTRW互斥
clip_ratio_pos: 0.3  # 正优势裁剪范围
clip_ratio_neg: 0.1  # 负优势裁剪范围
```

## 📊 记录的指标

### EMA指标 (创新点2.1)
- `ema/variance_reduction_ratio`: 方差降低比例
- `ema/raw_weights_variance`: 原始权重方差
- `ema/smoothed_weights_variance`: 平滑权重方差

### 梯度自适应指标 (创新点2.2)
- `gradient_adaptive/avg_gradient_norm`: 平均梯度范数
- `gradient_adaptive/weight_variance`: 权重方差

### AMIC指标 (创新点2.3)
- `amic/sequence_weights_variance`: 序列权重方差
- `amic/avg_sequence_length`: 平均序列长度

### PTRW指标 (创新点2.4)
- `ptrw/trust_weights_mean`: 信任权重均值
- `ptrw/loss_mean`: PTRW损失均值

### 时序衰减指标 (创新点2.5)
- `temporal_decay/weight_sum`: 权重总和
- `temporal_decay/first_weight`: 首个权重
- `temporal_decay/last_weight`: 最后权重

### 非对称裁剪指标 (创新点2.6)
- `asymmetric/pos_advantage_ratio`: 正优势比例
- `asymmetric/neg_advantage_ratio`: 负优势比例

## 🔬 实验建议

### 单一创新点实验
每个创新点都可以独立测试，建议按以下顺序：
1. 先测试创新点2.1 (EMA) - 最稳定
2. 再测试创新点2.3 (AMIC) - 与GSPO对比
3. 测试其他创新点

### 组合实验
推荐的组合方案：
- **保守组合**: EMA + 时序衰减
- **激进组合**: AMIC + 梯度自适应 + PTRW
- **平衡组合**: EMA + 梯度自适应 + 非对称裁剪

### 互斥关系
注意以下创新点不能同时启用：
- AMIC vs GSPO (模块A内互斥)
- PTRW vs 非对称裁剪 (模块C内互斥)

## 📈 论文写作支持

所有指标都会自动记录到WandB，支持：
- 方差降低效果图
- 不同β值对比
- 训练稳定性分析
- 性能提升统计

## 🛠️ 故障排除

### 常见问题
1. **梯度计算错误**: 梯度自适应加权使用简化版本，基于对数概率绝对值
2. **分布式训练**: 所有打印输出只在主进程执行
3. **内存使用**: 大部分创新点内存开销很小

### 调试模式
运行测试脚本查看详细输出：
```bash
python test_all_innovations.py
```

## 📝 引用

如果使用了这些创新点，请在论文中引用相应的理论基础和实现细节。
