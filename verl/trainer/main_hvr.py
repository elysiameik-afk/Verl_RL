"""
HVR训练主程序

基于内生奖励机制的强化学习训练，使用模型自身的logits计算价值函数，
结合稀疏奖励进行后见之明的价值重塑。

使用方法:
python -m verl.trainer.main_hvr --config-path=... --config-name=...
"""

import os
import sys
import yaml
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.hvr.hvr_trainer import HVRTrainer
from verl.trainer.ppo.core_algos import is_main_process


@ray.remote
class HVRTaskRunner:
    """HVR训练任务运行器"""
    
    def run(self, config: DictConfig):
        """运行HVR训练"""
        if is_main_process():
            print("🎯 [HVR] 启动HVR内生奖励训练")
            print("🎯 [HVR特性] ERVF价值函数 + 后见之明价值重塑")
            print(f"🎯 [HVR参数] α={config.actor_rollout_ref.actor.get('hvr_alpha', 1.0)}")
            print(f"🎯 [HVR参数] β={config.actor_rollout_ref.actor.get('hvr_beta', 0.1)}")
            print(f"🎯 [HVR参数] λ={config.actor_rollout_ref.actor.get('hvr_lambda', 0.5)}")
        
        # 创建HVR训练器
        trainer = HVRTrainer(config)
        
        # 初始化workers
        trainer.init_workers()
        
        # 开始训练
        trainer.fit()
        
        if is_main_process():
            print("🎉 [HVR] HVR训练任务完成！")


def run_hvr(config: DictConfig):
    """运行HVR训练"""
    # 验证HVR配置
    _validate_hvr_config(config)
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init()
    
    # 创建并运行HVR任务
    runner = HVRTaskRunner.remote()
    ray.get(runner.run.remote(config))


def _validate_hvr_config(config: DictConfig):
    """验证HVR配置"""
    if is_main_process():
        print("🔍 [HVR] 验证配置...")
    
    # 检查必需的HVR参数
    actor_config = config.actor_rollout_ref.actor
    
    # 确保关闭remove_padding (HVR需要完整logits)
    if actor_config.get("use_remove_padding", True):
        if is_main_process():
            print("⚠️ [HVR] 自动关闭use_remove_padding (HVR需要完整logits)")
        actor_config.use_remove_padding = False
    
    # 设置默认HVR参数
    hvr_defaults = {
        "hvr_alpha": 1.0,
        "hvr_beta": 0.1, 
        "hvr_lambda": 0.5,
        "hvr_cliprange": 0.2,
    }
    
    for key, default_value in hvr_defaults.items():
        if key not in actor_config:
            actor_config[key] = default_value
            if is_main_process():
                print(f"🔧 [HVR] 设置默认参数 {key}={default_value}")
    
    # 验证参数范围
    assert 0 < actor_config.hvr_alpha <= 10, f"hvr_alpha应在(0,10]范围内，当前值: {actor_config.hvr_alpha}"
    assert 0 <= actor_config.hvr_beta <= 1, f"hvr_beta应在[0,1]范围内，当前值: {actor_config.hvr_beta}"
    assert 0 <= actor_config.hvr_lambda <= 1, f"hvr_lambda应在[0,1]范围内，当前值: {actor_config.hvr_lambda}"
    
    # 确保使用支持的奖励管理器
    supported_reward_managers = ["logic_rl", "juben_reward", "prime"]
    reward_manager = config.reward_model.reward_manager
    if reward_manager not in supported_reward_managers:
        if is_main_process():
            print(f"⚠️ [HVR] 奖励管理器 {reward_manager} 可能不兼容，推荐使用: {supported_reward_managers}")
    
    if is_main_process():
        print("✅ [HVR] 配置验证完成")
        print(f"🎯 [HVR] 训练参数总结:")
        print(f"    α (温度系数): {actor_config.hvr_alpha}")
        print(f"    β (熵惩罚): {actor_config.hvr_beta}")
        print(f"    λ (混合因子): {actor_config.hvr_lambda}")
        print(f"    奖励管理器: {reward_manager}")


def load_hvr_config():
    """加载HVR配置文件"""
    # 获取配置文件路径
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    config_path = os.path.join(config_dir, "hvr_trainer.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"HVR配置文件未找到: {config_path}")

    # 加载YAML配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 转换为OmegaConf
    config = OmegaConf.create(config_dict)
    return config


def main():
    """HVR训练主函数"""
    if is_main_process():
        print("🚀 [HVR] 启动HVR内生奖励训练系统")

    try:
        # 加载配置
        config = load_hvr_config()

        if is_main_process():
            print("📋 [HVR] 配置概览:")
            print(OmegaConf.to_yaml(config))

        # 运行HVR训练
        run_hvr(config)

    except Exception as e:
        if is_main_process():
            print(f"❌ [HVR] 训练失败: {e}")
        raise
    finally:
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
