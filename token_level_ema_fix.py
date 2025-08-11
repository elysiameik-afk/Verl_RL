#!/usr/bin/env python3
"""
正确的Token级EMA实现
"""

import torch
import numpy as np

def apply_token_level_ema_smoothing(
    raw_weights: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """
    Apply token-level EMA smoothing within each sequence.
    
    核心创新：对同一序列内的token权重进行时序平滑
    w'[i,t] = β × w[i,t] + (1-β) × w'[i,t-1]
    
    Args:
        raw_weights: [batch_size, seq_len] - 原始重要性权重 w[i,t]
        response_mask: [batch_size, seq_len] - 有效token的mask
        beta: float - EMA平滑因子 (0 < β ≤ 1)
    
    Returns:
        tuple: (smoothed_weights, ema_metrics)
    """
    batch_size, seq_len = raw_weights.shape
    smoothed_weights = torch.zeros_like(raw_weights)
    
    sequence_variance_reductions = []
    sequence_smoothing_effects = []
    
    # 对每个序列应用token级EMA
    for i in range(batch_size):
        sequence_mask = response_mask[i]  # [seq_len]
        sequence_weights = raw_weights[i]  # [seq_len]
        smoothed_sequence = torch.zeros_like(sequence_weights)
        
        # 找到有效token位置
        valid_positions = torch.where(sequence_mask > 0)[0]
        if len(valid_positions) == 0:
            smoothed_weights[i] = sequence_weights
            continue
            
        # 初始化第一个有效token: w'[i,0] = w[i,0]
        first_pos = valid_positions[0].item()
        smoothed_sequence[first_pos] = sequence_weights[first_pos]
        
        # 对后续有效token应用EMA: w'[i,t] = β × w[i,t] + (1-β) × w'[i,t-1]
        prev_smoothed = smoothed_sequence[first_pos]
        for pos_idx in range(1, len(valid_positions)):
            t = valid_positions[pos_idx].item()
            current_raw = sequence_weights[t]
            current_smoothed = beta * current_raw + (1 - beta) * prev_smoothed
            smoothed_sequence[t] = current_smoothed
            prev_smoothed = current_smoothed
        
        # 复制无效token
        invalid_mask = sequence_mask == 0
        smoothed_sequence[invalid_mask] = sequence_weights[invalid_mask]
        
        smoothed_weights[i] = smoothed_sequence
        
        # 计算每个序列的指标
        if len(valid_positions) > 1:
            valid_raw = sequence_weights[valid_positions]
            valid_smoothed = smoothed_sequence[valid_positions]
            
            raw_var = valid_raw.var()
            smoothed_var = valid_smoothed.var()
            
            if raw_var > 1e-8:
                var_reduction = raw_var / (smoothed_var + 1e-8)
                sequence_variance_reductions.append(var_reduction.item())
            
            smoothing_effect = torch.norm(valid_raw - valid_smoothed).item()
            sequence_smoothing_effects.append(smoothing_effect)
    
    # 计算整体指标
    raw_variance = (raw_weights * response_mask).var()
    smoothed_variance = (smoothed_weights * response_mask).var()
    overall_variance_reduction = raw_variance / (smoothed_variance + 1e-8)
    
    # 编译最终指标
    ema_metrics = {
        # 核心方差指标
        'ema/raw_weights_variance': raw_variance.item(),
        'ema/smoothed_weights_variance': smoothed_variance.item(),
        'ema/variance_reduction_ratio': overall_variance_reduction.item(),
        
        # 平滑效果指标
        'ema/avg_sequence_variance_reduction': np.mean(sequence_variance_reductions) if sequence_variance_reductions else 1.0,
        'ema/avg_smoothing_effect': np.mean(sequence_smoothing_effects) if sequence_smoothing_effects else 0.0,
        
        # 基础统计
        'ema/raw_weights_mean': (raw_weights * response_mask).mean().item(),
        'ema/smoothed_weights_mean': (smoothed_weights * response_mask).mean().item(),
        'ema/raw_weights_std': (raw_weights * response_mask).std().item(),
        'ema/smoothed_weights_std': (smoothed_weights * response_mask).std().item(),
        
        # 配置信息
        'ema/beta': beta,
        'ema/use_ema': True,
        'ema/processed_sequences': batch_size,
        'ema/total_valid_tokens': response_mask.sum().item(),
    }
    
    # 调试输出
    if torch.distributed.get_rank() == 0 and batch_size > 0:
        i = 0
        valid_mask = response_mask[i] > 0
        if valid_mask.sum() > 1:
            raw_seq = raw_weights[i][valid_mask]
            smoothed_seq = smoothed_weights[i][valid_mask]
            print(f"🔍 [TOKEN-EMA] 序列{i} (有效token数: {valid_mask.sum().item()}):")
            print(f"  原始权重前5个: {raw_seq[:5].tolist()}")
            print(f"  平滑权重前5个: {smoothed_seq[:5].tolist()}")
            print(f"  方差变化: {raw_seq.var().item():.6f} → {smoothed_seq.var().item():.6f}")
            print(f"  平滑强度: {torch.norm(raw_seq - smoothed_seq).item():.6f}")
    
    return smoothed_weights, ema_metrics

def test_token_ema():
    """测试token级EMA"""
    print("🧪 测试Token级EMA...")
    
    # 创建有变化的权重序列
    batch_size, seq_len = 2, 6
    raw_weights = torch.tensor([
        [1.5, 0.8, 1.2, 0.9, 1.1, 0.7],  # 序列1：有明显变化
        [1.0, 1.3, 0.6, 1.4, 0.8, 1.2],  # 序列2：有明显变化
    ])
    response_mask = torch.ones(batch_size, seq_len)
    beta = 0.7  # 较低的beta以看到更明显效果
    
    print(f"原始权重:")
    print(f"  序列1: {raw_weights[0].tolist()}")
    print(f"  序列2: {raw_weights[1].tolist()}")
    print(f"  序列1方差: {raw_weights[0].var().item():.6f}")
    print(f"  序列2方差: {raw_weights[1].var().item():.6f}")
    
    smoothed_weights, metrics = apply_token_level_ema_smoothing(
        raw_weights=raw_weights,
        response_mask=response_mask,
        beta=beta,
    )
    
    print(f"\n平滑后权重:")
    print(f"  序列1: {smoothed_weights[0].tolist()}")
    print(f"  序列2: {smoothed_weights[1].tolist()}")
    print(f"  序列1方差: {smoothed_weights[0].var().item():.6f}")
    print(f"  序列2方差: {smoothed_weights[1].var().item():.6f}")
    
    print(f"\nEMA指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    return smoothed_weights, metrics

if __name__ == "__main__":
    test_token_ema()
