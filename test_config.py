#!/usr/bin/env python3
"""
测试配置文件是否正确加载EMA参数
"""

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def test_config(cfg: DictConfig):
    print("✅ 配置加载成功!")
    print(f"use_ema_smoothing: {cfg.actor_rollout_ref.actor.use_ema_smoothing}")
    print(f"ema_beta: {cfg.actor_rollout_ref.actor.ema_beta}")
    
    # 测试覆盖配置
    print("\n🔧 测试配置覆盖...")
    print(f"原始 use_ema_smoothing: {cfg.actor_rollout_ref.actor.use_ema_smoothing}")
    
    return cfg

if __name__ == "__main__":
    test_config()
