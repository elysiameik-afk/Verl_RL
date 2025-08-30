#!/usr/bin/env python3
"""
测试分布式修复的简单脚本
"""

import torch

def test_distributed_checks():
    """测试分布式检查逻辑"""
    
    print("🧪 测试分布式检查逻辑...")
    
    # 测试未初始化的情况
    print(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
    
    # 测试我们的检查逻辑
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    print(f"should_print (未初始化): {should_print}")
    
    if should_print:
        print("✅ 在未初始化分布式的情况下，可以正常打印")
    
    # 测试rank获取逻辑
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"✅ 安全获取rank: {rank}")
    except Exception as e:
        print(f"❌ 获取rank失败: {e}")
    
    print("✅ 分布式检查逻辑测试完成!")

if __name__ == "__main__":
    test_distributed_checks()
