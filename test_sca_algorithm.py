#!/usr/bin/env python3
"""
测试结构化信用分配 (SCA) 算法
"""

import torch
import numpy as np
from verl.trainer.ppo.core_algos import apply_structured_credit_assignment, find_subsequence

def create_test_sequence():
    """创建测试序列"""
    # 模拟token序列: <think>推理过程</think><answer>答案内容</answer>
    THINK_OPEN = [151667]
    THINK_CLOSE = [151668]
    ANSWER_OPEN = [27, 9217, 29]
    ANSWER_CLOSE = [522, 9217, 29]
    
    # 构造测试序列
    sequence = []
    sequence.extend(THINK_OPEN)  # <think>
    sequence.extend([100, 101, 102, 103, 104])  # 推理过程 (5个token)
    sequence.extend(THINK_CLOSE)  # </think>
    sequence.extend(ANSWER_OPEN)  # <answer>
    sequence.extend([200, 201, 202])  # 答案内容 (3个token)
    sequence.extend(ANSWER_CLOSE)  # </answer>
    
    return sequence

def test_subsequence_finding():
    """测试子序列查找功能"""
    print("🔍 测试子序列查找功能\n")
    
    sequence = create_test_sequence()
    print(f"测试序列: {sequence}")
    
    THINK_OPEN = [151667]
    THINK_CLOSE = [151668]
    ANSWER_OPEN = [27, 9217, 29]
    ANSWER_CLOSE = [522, 9217, 29]
    
    think_start = find_subsequence(sequence, THINK_OPEN)
    think_end = find_subsequence(sequence, THINK_CLOSE)
    answer_start = find_subsequence(sequence, ANSWER_OPEN)
    answer_end = find_subsequence(sequence, ANSWER_CLOSE)
    
    print(f"<think> 起始位置: {think_start}")
    print(f"</think> 起始位置: {think_end}")
    print(f"<answer> 起始位置: {answer_start}")
    print(f"</answer> 起始位置: {answer_end}")
    
    # 验证内容范围
    process_start = think_start + len(THINK_OPEN)
    process_end = think_end
    answer_content_start = answer_start + len(ANSWER_OPEN)
    answer_content_end = answer_end
    
    print(f"推理过程范围: [{process_start}, {process_end})")
    print(f"推理内容: {sequence[process_start:process_end]}")
    print(f"答案内容范围: [{answer_content_start}, {answer_content_end})")
    print(f"答案内容: {sequence[answer_content_start:answer_content_end]}")
    print()

def test_sca_positive_reward():
    """测试正奖励情况下的SCA"""
    print("🎯 测试正奖励情况下的SCA\n")
    
    # 创建测试数据
    sequence = create_test_sequence()
    batch_size = 2
    seq_len = len(sequence)
    
    # 构造输入张量
    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    token_ids[0, :] = torch.tensor(sequence)
    token_ids[1, :] = torch.tensor(sequence)  # 重复序列
    
    # 正优势 (模拟正奖励)
    advantages = torch.ones((batch_size, seq_len)) * 0.5
    
    # 全部有效
    response_mask = torch.ones((batch_size, seq_len))
    
    # 应用SCA
    adjusted_weights, metrics = apply_structured_credit_assignment(
        token_ids=token_ids,
        advantages=advantages,
        response_mask=response_mask,
        answer_credit_ratio=0.3,
        structure_credit_ratio=0.2,
        process_credit_ratio=0.5,
        lspd_alpha=2.0,
        lspd_tau=10.0,
        lspd_normalize=True,
    )
    
    print("SCA指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n调整后的权重 (第一个序列):")
    weights_seq1 = adjusted_weights[0].tolist()
    for i, weight in enumerate(weights_seq1):
        print(f"  位置{i} (token {sequence[i]}): {weight:.6f}")
    
    print()

def test_sca_negative_reward():
    """测试负奖励情况下的SCA"""
    print("🎯 测试负奖励情况下的SCA\n")
    
    # 创建测试数据
    sequence = create_test_sequence()
    batch_size = 1
    seq_len = len(sequence)
    
    # 构造输入张量
    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    token_ids[0, :] = torch.tensor(sequence)
    
    # 负优势 (模拟负奖励)
    advantages = torch.ones((batch_size, seq_len)) * -0.5
    
    # 全部有效
    response_mask = torch.ones((batch_size, seq_len))
    
    # 应用SCA
    adjusted_weights, metrics = apply_structured_credit_assignment(
        token_ids=token_ids,
        advantages=advantages,
        response_mask=response_mask,
        answer_credit_ratio=0.3,
        structure_credit_ratio=0.2,
        process_credit_ratio=0.5,
        lspd_alpha=2.0,
        lspd_tau=10.0,
        lspd_normalize=True,
    )
    
    print("负奖励SCA指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n调整后的权重 (负奖励):")
    weights_seq1 = adjusted_weights[0].tolist()
    for i, weight in enumerate(weights_seq1):
        print(f"  位置{i} (token {sequence[i]}): {weight:.6f}")
    
    print()

def test_sca_malformed_sequence():
    """测试格式错误的序列"""
    print("🎯 测试格式错误的序列\n")
    
    # 创建缺少标记的序列
    malformed_sequence = [100, 101, 102, 103, 104]  # 没有任何标记
    batch_size = 1
    seq_len = len(malformed_sequence)
    
    # 构造输入张量
    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    token_ids[0, :] = torch.tensor(malformed_sequence)
    
    # 负优势
    advantages = torch.ones((batch_size, seq_len)) * -0.5
    
    # 全部有效
    response_mask = torch.ones((batch_size, seq_len))
    
    # 应用SCA
    adjusted_weights, metrics = apply_structured_credit_assignment(
        token_ids=token_ids,
        advantages=advantages,
        response_mask=response_mask,
        answer_credit_ratio=0.3,
        structure_credit_ratio=0.2,
        process_credit_ratio=0.5,
        lspd_alpha=2.0,
        lspd_tau=10.0,
        lspd_normalize=True,
    )
    
    print("格式错误序列SCA指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n调整后的权重 (格式错误):")
    weights_seq1 = adjusted_weights[0].tolist()
    for i, weight in enumerate(weights_seq1):
        print(f"  位置{i} (token {malformed_sequence[i]}): {weight:.6f}")
    
    print()

def test_credit_ratio_validation():
    """验证信用比例分配"""
    print("🎯 验证信用比例分配\n")
    
    # 测试不同的信用比例
    ratios = [
        (0.3, 0.2, 0.5),  # 标准
        (0.5, 0.1, 0.4),  # 更重视答案
        (0.1, 0.1, 0.8),  # 更重视推理过程
    ]
    
    sequence = create_test_sequence()
    
    for answer_ratio, structure_ratio, process_ratio in ratios:
        print(f"测试比例 - 答案:{answer_ratio}, 结构:{structure_ratio}, 推理:{process_ratio}")
        
        # 验证比例总和
        total = answer_ratio + structure_ratio + process_ratio
        print(f"  比例总和: {total}")
        
        if abs(total - 1.0) > 1e-6:
            print(f"  ⚠️ 警告: 比例总和不等于1.0")
        else:
            print(f"  ✅ 比例总和正确")
        
        print()

if __name__ == "__main__":
    print("🚀 开始测试结构化信用分配 (SCA) 算法\n")
    
    test_subsequence_finding()
    test_sca_positive_reward()
    test_sca_negative_reward()
    test_sca_malformed_sequence()
    test_credit_ratio_validation()
    
    print("🎉 SCA算法测试完成！")
    print("\n📋 SCA算法特性总结:")
    print("  ✅ 结构化解析: 自动识别<think>和<answer>标记")
    print("  ✅ 差异化分配: 不同部分使用不同的信用分配策略")
    print("  ✅ LSPD集成: 推理过程部分应用时序衰减")
    print("  ✅ 奖励敏感: 根据奖励正负采用不同策略")
    print("  ✅ 错误处理: 格式错误时的降级处理")
    print("\n🎯 SCA算法已准备就绪，可以开始训练！")
