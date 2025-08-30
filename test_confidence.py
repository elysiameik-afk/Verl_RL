#!/usr/bin/env python3
"""
测试自信度计算功能的简单脚本
"""

import torch
import torch.nn as nn

def test_confidence_methods():
    """直接测试自信度计算方法"""

    # 模拟DataParallelPPOActor的自信度计算方法
    class ConfidenceCalculator:
        def __init__(self):
            self.lgc_window_size = 256
            self.lgc_avg_pool = torch.nn.AvgPool1d(kernel_size=self.lgc_window_size, stride=1)

        def _compute_token_confidence_from_logits(self, logits: torch.Tensor, sampled_tokens: torch.Tensor, top_k: int = 20) -> torch.Tensor:
            """计算token级别的置信度"""
            # 计算log概率
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # 获取top-k的值和索引
            top_k_values, top_k_indices = torch.topk(log_probs, k=top_k, dim=-1)

            # 找到实际采样token在top-k中的位置
            sampled_tokens_expanded = sampled_tokens.unsqueeze(-1)
            mask = (top_k_indices == sampled_tokens_expanded)

            # 创建排除采样token的mask
            exclude_mask = ~mask

            # 计算排除采样token后的平均log概率
            masked_values = top_k_values.masked_fill(mask, float('-inf'))

            # 计算有效token的数量（应该是top_k-1）
            valid_count = exclude_mask.sum(dim=-1, keepdim=True).float()

            # 计算平均值，排除-inf的值
            valid_values = masked_values.masked_fill(masked_values == float('-inf'), 0.0)
            confidence = valid_values.sum(dim=-1) / valid_count.squeeze(-1)

            # 取反得到置信度分数（越高越自信）
            return -confidence

        def _compute_lgc_from_token_confidence(self, token_confidence: torch.Tensor) -> torch.Tensor:
            """从token置信度计算序列级别的LGC分数"""
            batch_size, response_len = token_confidence.shape

            # 边缘情况：如果序列长度小于窗口大小，直接计算平均
            if response_len < self.lgc_window_size:
                return token_confidence.mean(dim=-1)

            # 计算组置信度：使用滑动窗口
            token_confidence_expanded = token_confidence.unsqueeze(1)
            group_confidence = self.lgc_avg_pool(token_confidence_expanded)
            group_confidence = group_confidence.squeeze(1)

            # 计算LGC：取最小值（最低组置信度）
            lgc_scores = torch.min(group_confidence, dim=-1).values

            return lgc_scores

    calculator = ConfidenceCalculator()

    print("🧪 开始测试自信度计算...")

    # 创建测试数据
    batch_size = 2
    response_len = 10  # 短序列测试
    vocab_size = 1000

    print("\n1. 测试token置信度计算:")
    logits = torch.randn(batch_size, response_len, vocab_size)
    sampled_tokens = torch.randint(0, vocab_size, (batch_size, response_len))

    token_conf = calculator._compute_token_confidence_from_logits(logits, sampled_tokens)
    print(f"   token confidence shape: {token_conf.shape}")
    print(f"   token confidence values: {token_conf}")

    print("\n2. 测试LGC计算 (短序列):")
    lgc_scores = calculator._compute_lgc_from_token_confidence(token_conf)
    print(f"   LGC scores shape: {lgc_scores.shape}")
    print(f"   LGC scores values: {lgc_scores}")

    print("\n3. 测试LGC计算 (长序列):")
    # 测试长序列（超过窗口大小）
    long_response_len = 300
    long_token_conf = torch.randn(batch_size, long_response_len)
    long_lgc_scores = calculator._compute_lgc_from_token_confidence(long_token_conf)
    print(f"   长序列 LGC scores shape: {long_lgc_scores.shape}")
    print(f"   长序列 LGC scores values: {long_lgc_scores}")

    print("\n4. 验证DeepConf逻辑:")
    # 创建一个简单的例子来验证DeepConf逻辑
    simple_logits = torch.tensor([[[10.0, 1.0, 0.5, 0.1, 0.0]]])  # (1, 1, 5)
    simple_sampled = torch.tensor([[0]])  # 采样了第0个token（概率最高的）

    simple_conf = calculator._compute_token_confidence_from_logits(simple_logits, simple_sampled, top_k=5)
    print(f"   简单例子置信度: {simple_conf}")
    print(f"   (应该是排除最高概率token后，剩余4个token的平均log概率的负值)")

    print("\n✅ 自信度计算测试完成!")

if __name__ == "__main__":
    test_confidence_methods()
