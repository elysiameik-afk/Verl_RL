#!/usr/bin/env python3
"""
HVR功能测试脚本

测试HVR Logic RL Reward Manager的基本功能
"""

import torch
import numpy as np
from verl import DataProto
from verl.workers.reward_manager.hvr_logic_rl_reward import (
    HVRLogicRLRewardManager, 
    calculate_ervf_value, 
    calculate_hvr_rewards_for_group
)

def test_ervf_value():
    """测试ERVF价值函数计算"""
    print("🧪 测试ERVF价值函数...")
    
    # 创建测试logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    
    # 测试不同参数
    alpha, beta = 1.0, 0.1
    v_ervf = calculate_ervf_value(logits, alpha, beta)
    
    print(f"✅ ERVF值计算成功: {v_ervf:.4f}")
    assert isinstance(v_ervf, float), "ERVF值应该是float类型"
    assert not np.isnan(v_ervf), "ERVF值不应该是NaN"
    assert not np.isinf(v_ervf), "ERVF值不应该是无穷大"


def test_hvr_group_calculation():
    """测试HVR组级计算"""
    print("🧪 测试HVR组级计算...")
    
    # 创建测试数据
    seq_len, vocab_size = 10, 1000
    group_size = 4
    
    group_data = []
    for i in range(group_size):
        logits = torch.randn(seq_len, vocab_size)
        response_ids = torch.randint(0, vocab_size, (seq_len,))
        r_final = np.random.choice([-3, -1, 1, 3])  # 模拟外部奖励
        
        group_data.append({
            'logits': logits,
            'ids': response_ids,
            'r_final': r_final,
            'external_score': r_final
        })
    
    # 计算HVR奖励
    hvr_returns, metrics = calculate_hvr_rewards_for_group(
        group_data, alpha=1.0, beta=0.1, lambda_hvr=0.5
    )
    
    print(f"✅ HVR组级计算成功")
    print(f"   - 返回{len(hvr_returns)}个奖励值")
    print(f"   - 奖励范围: [{min(hvr_returns):.2f}, {max(hvr_returns):.2f}]")
    print(f"   - 指标数量: {len(metrics)}")
    
    # 验证结果
    assert len(hvr_returns) == group_size, f"应该返回{group_size}个奖励值"
    assert all(-6.1 <= r <= 6.1 for r in hvr_returns), "奖励值应该在[-6,6]范围内"
    assert 'v_ervf_mean' in metrics, "应该包含v_ervf_mean指标"
    assert 'external_score_mean' in metrics, "应该包含external_score_mean指标"


def test_hvr_manager_basic():
    """测试HVR Manager基本功能"""
    print("🧪 测试HVR Manager基本功能...")
    
    try:
        # 这个测试需要tokenizer，可能会失败，但至少测试导入
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # 创建HVR Manager
        hvr_manager = HVRLogicRLRewardManager(
            tokenizer=tokenizer,
            num_examine=5,
            alpha=1.0,
            beta=0.1,
            lambda_hvr=0.5
        )
        
        print("✅ HVR Manager创建成功")
        print(f"   - alpha: {hvr_manager.alpha}")
        print(f"   - beta: {hvr_manager.beta}")
        print(f"   - lambda_hvr: {hvr_manager.lambda_hvr}")
        
    except Exception as e:
        print(f"⚠️  HVR Manager测试跳过 (需要tokenizer): {e}")


def test_numerical_stability():
    """测试数值稳定性"""
    print("🧪 测试数值稳定性...")
    
    # 测试极端logits值
    extreme_logits = torch.tensor([100.0, -100.0, 0.0] * 100)
    
    try:
        v_ervf = calculate_ervf_value(extreme_logits, 1.0, 0.1)
        print(f"✅ 极端值处理成功: {v_ervf:.4f}")
        assert not np.isnan(v_ervf), "极端值不应该产生NaN"
        assert not np.isinf(v_ervf), "极端值不应该产生无穷大"
    except Exception as e:
        print(f"❌ 数值稳定性测试失败: {e}")
        raise


def main():
    """运行所有测试"""
    print("🚀 开始HVR功能测试...\n")
    
    try:
        test_ervf_value()
        print()
        
        test_hvr_group_calculation()
        print()
        
        test_hvr_manager_basic()
        print()
        
        test_numerical_stability()
        print()
        
        print("🎉 所有测试通过！HVR功能正常工作。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
