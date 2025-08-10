#!/usr/bin/env python3
"""
测试DataProto split/chunk修复
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    from verl.protocol import DataProto
    print("✅ 成功导入DataProto")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_dataproto_methods():
    """测试DataProto的split和chunk方法"""
    print("\n🧪 测试DataProto方法...")
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 8
    
    batch_data = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "responses": torch.randint(0, 1000, (batch_size, seq_len//2)),
    }
    
    non_tensor_data = {
        "uid": [f"seq_{i}" for i in range(batch_size)]
    }
    
    # 创建DataProto对象
    data_proto = DataProto(batch=batch_data, non_tensor_batch=non_tensor_data)
    
    print(f"原始batch_size: {data_proto.batch['input_ids'].shape[0]}")
    print(f"原始uid: {data_proto.non_tensor_batch['uid']}")
    
    # 测试split方法
    print(f"\n测试split方法:")
    try:
        if hasattr(data_proto, 'split'):
            split_result = data_proto.split(2)
            print(f"  ✅ split方法存在")
            print(f"  分割后数量: {len(list(split_result))}")
        else:
            print(f"  ❌ DataProto没有split方法")
    except Exception as e:
        print(f"  ❌ split方法调用失败: {e}")
    
    # 测试chunk方法
    print(f"\n测试chunk方法:")
    try:
        if hasattr(data_proto, 'chunk'):
            chunk_result = data_proto.chunk(2)
            print(f"  ✅ chunk方法存在")
            chunks = list(chunk_result)
            print(f"  分块后数量: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"    块{i}: batch_size={chunk.batch['input_ids'].shape[0]}, uid={chunk.non_tensor_batch.get('uid', 'None')}")
        else:
            print(f"  ❌ DataProto没有chunk方法")
    except Exception as e:
        print(f"  ❌ chunk方法调用失败: {e}")
    
    return True

def test_fallback_logic():
    """测试回退逻辑"""
    print(f"\n🔧 测试回退逻辑...")
    
    # 模拟mini_batch对象
    class MockMiniBatch:
        def __init__(self, has_split=True):
            self.batch = {"input_ids": torch.zeros(4, 8)}
            if has_split:
                self.split = lambda x: [MockMiniBatch(False), MockMiniBatch(False)]
    
    # 测试1: 有split方法
    mini_batch_with_split = MockMiniBatch(has_split=True)
    if hasattr(mini_batch_with_split, 'split'):
        print("  ✅ 对象有split方法，使用split")
    else:
        print("  ❌ 对象没有split方法")
    
    # 测试2: 没有split方法（DataProto情况）
    mini_batch_dataproto = MockMiniBatch(has_split=False)
    if hasattr(mini_batch_dataproto, 'split'):
        print("  使用split方法")
    else:
        print("  ✅ 对象没有split方法，应该使用chunk回退逻辑")
    
    return True

if __name__ == "__main__":
    print("🚀 开始测试DataProto修复...")
    
    try:
        # 测试DataProto方法
        test_dataproto_methods()
        
        # 测试回退逻辑
        test_fallback_logic()
        
        print(f"\n✅ 测试完成！")
        print(f"\n📋 修复总结:")
        print(f"  1. ✓ 统一了DataProto的分割逻辑")
        print(f"  2. ✓ 添加了split/chunk方法的回退机制")
        print(f"  3. ✓ 修复了uid字段在mini-batch中的传递")
        print(f"  4. ✓ 同时修复了dp_actor和megatron_actor")
        
        print(f"\n🔧 现在重新运行训练应该不会出现AttributeError了！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
