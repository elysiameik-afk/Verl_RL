#!/usr/bin/env python3
"""
简化的HVR训练启动脚本

直接使用Python字典配置，避免复杂的Hydra配置系统
"""

import os
import sys
import ray
from omegaconf import OmegaConf

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from verl.trainer.hvr.hvr_trainer import HVRTrainer
from verl.trainer.ppo.core_algos import is_main_process


def create_hvr_config():
    """创建HVR配置"""
    config = {
        # 训练器配置
        "trainer": {
            "project_name": "HVR-IntrinsicReward",
            "experiment_name": "HVR_ERVF_Pure_Test",
            "nnodes": 1,
            "n_gpus_per_node": 1,
            "total_epochs": 8,
            "save_freq": 2,
            "test_freq": 1,
            "logger": ["wandb"],
            "default_local_dir": "/root/autodl-tmp/myverl/ckpts/HVR_Pure",
            "critic_warmup": 0,
        },
        
        # Actor配置
        "actor_rollout_ref": {
            "actor": {
                "optim": {
                    "lr": 3e-6,
                    "weight_decay": 0.01,
                },
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 4,
                "use_remove_padding": False,  # HVR必需
                
                # HVR核心参数
                "hvr_alpha": 1.0,
                "hvr_beta": 0.1,
                "hvr_lambda": 0.5,
                "hvr_cliprange": 0.2,
                
                # FSDP配置
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
                "grad_clip": 1.0,
                
                # 禁用其他创新点
                "use_ema_smoothing": False,
                "use_gradient_adaptive_weighting": False,
                "use_amic": False,
                "use_ptrw": False,
                "use_temporal_decay": False,
                "use_sca": False,
                "use_asymmetric_clipping": False,
            },
            
            # 模型配置
            "model": {
                "path": "/root/autodl-tmp/myverl/mymodels/qwen3-0.6b",
                "use_remove_padding": False,
                "enable_gradient_checkpointing": True,
            },
            
            # Rollout配置
            "rollout": {
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 8,
                "max_num_batched_tokens": 16384,
                "tensor_model_parallel_size": 1,
            },
            
            # Reference配置
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {
                    "param_offload": False,
                },
            },
        },
        
        # 数据配置
        "data": {
            "train_files": "/root/autodl-tmp/myverl/data/kk/4ppl_few/train.parquet",
            "val_files": "/root/autodl-tmp/myverl/data/kk/4ppl_few/test.parquet",
            "train_batch_size": 16,
            "val_batch_size": 8,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        
        # 奖励模型配置
        "reward_model": {
            "reward_manager": "logic_rl",
        },
        
        # 算法配置
        "algorithm": {
            "adv_estimator": "hvr",
            "kl_ctrl": {
                "kl_coef": 0.05,
            },
        },
    }
    
    return OmegaConf.create(config)


def validate_hvr_config(config):
    """验证HVR配置"""
    if is_main_process():
        print("🔍 [HVR] 验证配置...")
    
    # 检查必需的路径
    model_path = config.actor_rollout_ref.model.path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    train_files = config.data.train_files
    if not os.path.exists(train_files):
        raise FileNotFoundError(f"训练数据不存在: {train_files}")
    
    # 验证HVR参数
    actor_config = config.actor_rollout_ref.actor
    assert 0 < actor_config.hvr_alpha <= 10, f"hvr_alpha应在(0,10]范围内"
    assert 0 <= actor_config.hvr_beta <= 1, f"hvr_beta应在[0,1]范围内"
    assert 0 <= actor_config.hvr_lambda <= 1, f"hvr_lambda应在[0,1]范围内"
    
    if is_main_process():
        print("✅ [HVR] 配置验证完成")


def run_hvr_training():
    """运行HVR训练"""
    if is_main_process():
        print("🎯 [HVR] 启动HVR内生奖励训练")
        print("🎯 [HVR特性] ERVF价值函数 + 后见之明价值重塑")
    
    try:
        # 创建配置
        config = create_hvr_config()
        
        # 验证配置
        validate_hvr_config(config)
        
        if is_main_process():
            print(f"🎯 [HVR参数] α={config.actor_rollout_ref.actor.hvr_alpha}")
            print(f"🎯 [HVR参数] β={config.actor_rollout_ref.actor.hvr_beta}")
            print(f"🎯 [HVR参数] λ={config.actor_rollout_ref.actor.hvr_lambda}")
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init()
        
        # 创建HVR训练器
        trainer = HVRTrainer(config)
        
        # 初始化workers
        trainer.init_workers()
        
        # 开始训练
        trainer.fit()
        
        if is_main_process():
            print("🎉 [HVR] HVR训练完成！")
            
    except Exception as e:
        if is_main_process():
            print(f"❌ [HVR] 训练失败: {e}")
        raise
    finally:
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # 设置环境变量
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
    print("🚀 [HVR] 启动HVR纯净内生奖励训练系统")
    print("🎯 [HVR优势] 无需critic网络，基于模型自身logits的内生价值估计")
    
    run_hvr_training()
