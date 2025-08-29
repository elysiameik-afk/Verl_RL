#!/usr/bin/env python3
"""
测试HVR配置是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import tempfile

def test_hvr_config():
    """测试HVR配置加载"""
    print("🧪 测试HVR配置加载...")
    
    try:
        # 获取配置目录的绝对路径
        config_dir = os.path.abspath("verl/trainer/config")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # 测试基础配置
            cfg = compose(config_name="ppo_trainer")
            
            print("✅ 基础配置加载成功")
            print(f"   - reward_manager: {cfg.reward_model.reward_manager}")
            print(f"   - alpha: {cfg.reward_model.alpha}")
            print(f"   - beta: {cfg.reward_model.beta}")
            print(f"   - lambda_hvr: {cfg.reward_model.lambda_hvr}")
            print(f"   - use_zscore: {cfg.reward_model.use_zscore}")
            print(f"   - target_scale: {cfg.reward_model.target_scale}")
            
            # 测试HVR配置覆盖
            cfg_hvr = compose(
                config_name="ppo_trainer",
                overrides=[
                    "reward_model.reward_manager=hvr_logic_rl",
                    "reward_model.alpha=1.5",
                    "reward_model.beta=0.2",
                    "reward_model.lambda_hvr=0.7",
                    "reward_model.use_zscore=false",
                    "reward_model.target_scale=4.0"
                ]
            )
            
            print("\n✅ HVR配置覆盖成功")
            print(f"   - reward_manager: {cfg_hvr.reward_model.reward_manager}")
            print(f"   - alpha: {cfg_hvr.reward_model.alpha}")
            print(f"   - beta: {cfg_hvr.reward_model.beta}")
            print(f"   - lambda_hvr: {cfg_hvr.reward_model.lambda_hvr}")
            print(f"   - use_zscore: {cfg_hvr.reward_model.use_zscore}")
            print(f"   - target_scale: {cfg_hvr.reward_model.target_scale}")
            
            return True
            
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hvr_manager_creation():
    """测试HVR Manager创建"""
    print("\n🧪 测试HVR Manager创建...")
    
    try:
        from verl.workers.reward_manager.hvr_logic_rl_reward import HVRLogicRLRewardManager
        
        # 模拟配置参数
        config_params = {
            'alpha': 1.5,
            'beta': 0.2,
            'lambda_hvr': 0.7,
            'use_zscore': False,
            'target_scale': 4.0
        }
        
        # 创建一个简单的tokenizer mock
        class MockTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return "mock response"
        
        tokenizer = MockTokenizer()
        
        # 创建HVR Manager
        hvr_manager = HVRLogicRLRewardManager(
            tokenizer=tokenizer,
            num_examine=5,
            **config_params
        )
        
        print("✅ HVR Manager创建成功")
        print(f"   - alpha: {hvr_manager.alpha}")
        print(f"   - beta: {hvr_manager.beta}")
        print(f"   - lambda_hvr: {hvr_manager.lambda_hvr}")
        print(f"   - use_zscore: {hvr_manager.use_zscore}")
        print(f"   - target_scale: {hvr_manager.target_scale}")
        
        return True
        
    except Exception as e:
        print(f"❌ HVR Manager创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("🚀 开始HVR配置测试...\n")
    
    success = True
    
    # 测试配置加载
    if not test_hvr_config():
        success = False
    
    # 测试Manager创建
    if not test_hvr_manager_creation():
        success = False
    
    if success:
        print("\n🎉 所有配置测试通过！HVR配置正确。")
    else:
        print("\n❌ 配置测试失败，请检查配置文件。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
