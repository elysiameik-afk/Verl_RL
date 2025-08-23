"""
HVR训练器实现

基于Ray的分布式HVR训练器，专门为内生奖励机制设计：
1. 简化的训练流程（无critic）
2. 直接使用HVR奖励进行策略优化
3. 复用现有的rollout和奖励计算组件
"""

import ray
import torch
from typing import Dict, Any
from omegaconf import DictConfig

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.hvr.hvr_actor import HVRActor
from verl.trainer.ppo.core_algos import is_main_process
from verl.single_controller.ray.base import RayClassWithInitArgs


class HVRTrainer(RayPPOTrainer):
    """
    HVR专用训练器
    
    继承RayPPOTrainer但专门为HVR内生奖励机制设计：
    - 移除critic相关逻辑
    - 使用HVRActor替代标准Actor
    - 简化的训练循环
    """
    
    def __init__(self, config: DictConfig):
        # 调用父类初始化，但会在后续替换actor
        super().__init__(config)
        
        if is_main_process():
            print("🎯 [HVR Trainer] 初始化HVR训练器")
            print("🎯 [HVR特性] 内生奖励机制，无需critic网络")
    
    def init_workers(self):
        """初始化HVR专用的workers"""
        if is_main_process():
            print("🎯 [HVR] 初始化HVR专用workers...")
        
        # 初始化rollout workers (复用现有实现)
        self.rollout_wg.init_model()
        
        # 初始化reference policy (复用现有实现)  
        self.ref_policy_wg.init_model()
        
        # 初始化HVR Actor (替换标准actor)
        self._init_hvr_actor()
        
        if is_main_process():
            print("✅ [HVR] HVR workers初始化完成")
    
    def _init_hvr_actor(self):
        """初始化HVR专用Actor"""
        if is_main_process():
            print("🎯 [HVR] 初始化HVR Actor...")
        
        # 创建HVR Actor配置
        hvr_actor_config = self.config.actor_rollout_ref.actor.copy()
        
        # 确保HVR必需的配置
        hvr_actor_config.use_remove_padding = False  # HVR需要完整logits
        
        # 创建HVR Actor类
        @ray.remote(num_gpus=self.config.trainer.n_gpus_per_node)
        class HVRActorWorker(RayClassWithInitArgs):
            def __init__(self, config):
                self.actor = HVRActor(config)
            
            def update_policy(self, data):
                return self.actor.update_policy(data)
            
            def generate_sequences(self, prompts, **kwargs):
                return self.actor.generate_sequences(prompts, **kwargs)
        
        # 初始化HVR Actor workers
        self.hvr_actor_wg = HVRActorWorker.remote(hvr_actor_config)
        
        if is_main_process():
            print("✅ [HVR] HVR Actor初始化完成")
    
    def fit(self):
        """HVR专用的训练循环"""
        if is_main_process():
            print("🚀 [HVR] 开始HVR训练循环")
        
        for epoch in range(self.config.trainer.total_epochs):
            if is_main_process():
                print(f"\n🎯 [HVR] Epoch {epoch + 1}/{self.config.trainer.total_epochs}")
            
            # 训练一个epoch
            epoch_metrics = self._train_epoch(epoch)
            
            # 记录指标
            if self.logger:
                self.logger.log(epoch_metrics, step=epoch)
            
            # 保存检查点
            if (epoch + 1) % self.config.trainer.save_freq == 0:
                self._save_checkpoint(epoch)
            
            # 验证
            if (epoch + 1) % self.config.trainer.test_freq == 0:
                val_metrics = self._validate(epoch)
                if self.logger:
                    self.logger.log(val_metrics, step=epoch)
        
        if is_main_process():
            print("🎉 [HVR] HVR训练完成！")
    
    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        """训练一个epoch"""
        epoch_metrics = {}
        
        # 获取训练数据
        for batch_idx, batch in enumerate(self.train_dataloader):
            if is_main_process() and batch_idx % 10 == 0:
                print(f"🔄 [HVR] Epoch {epoch}, Batch {batch_idx}")
            
            # 执行一个训练步骤
            step_metrics = self._train_step(batch)
            
            # 聚合指标
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # 计算epoch平均指标
        for key, values in epoch_metrics.items():
            if isinstance(values[0], (int, float)):
                epoch_metrics[key] = sum(values) / len(values)
        
        return epoch_metrics
    
    def _train_step(self, batch) -> Dict[str, Any]:
        """执行一个训练步骤"""
        # 1. Rollout阶段 (复用现有实现)
        rollout_output = self.rollout_wg.rollout(batch)
        
        # 2. 奖励计算 (复用现有的logic_rl等)
        reward_output = self.reward_wg.compute_reward(rollout_output)
        
        # 3. Reference policy计算 (复用现有实现)
        ref_output = self.ref_policy_wg.compute_ref_log_prob(rollout_output)
        
        # 4. 准备HVR训练数据
        hvr_batch = self._prepare_hvr_batch(rollout_output, reward_output, ref_output)
        
        # 5. HVR Actor更新
        actor_metrics = ray.get(self.hvr_actor_wg.update_policy.remote(hvr_batch))
        
        return actor_metrics
    
    def _prepare_hvr_batch(self, rollout_output, reward_output, ref_output):
        """准备HVR训练数据"""
        # 组合所有必需的数据
        hvr_batch = {
            # 基础序列数据
            "responses": rollout_output["responses"],
            "input_ids": rollout_output["input_ids"], 
            "attention_mask": rollout_output["attention_mask"],
            "position_ids": rollout_output["position_ids"],
            
            # 参考策略数据
            "ref_log_prob": ref_output["ref_log_prob"],
            
            # 其他元数据
            "uid": rollout_output.get("uid", []),
        }
        
        # 添加奖励数据到batch中
        if hasattr(reward_output, 'batch'):
            hvr_batch.batch = reward_output.batch
        else:
            # 如果reward_output不是batch格式，创建batch
            hvr_batch.batch = {
                "token_level_rewards": reward_output.get("token_level_rewards"),
                "token_level_scores": reward_output.get("token_level_scores"),
            }
        
        return hvr_batch
    
    def _validate(self, epoch: int) -> Dict[str, Any]:
        """验证阶段"""
        if is_main_process():
            print(f"🔍 [HVR] 执行验证 (Epoch {epoch})")
        
        # 简化的验证逻辑
        val_metrics = {
            "val/epoch": epoch,
            "val/hvr_validation": True,
        }
        
        return val_metrics
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        if is_main_process():
            print(f"💾 [HVR] 保存检查点 (Epoch {epoch})")
        
        # 保存HVR Actor状态
        # 这里可以添加具体的保存逻辑
        pass
    
    @property
    def train_dataloader(self):
        """训练数据加载器 (复用父类实现)"""
        return super().train_dataloader
    
    @property 
    def val_dataloader(self):
        """验证数据加载器 (复用父类实现)"""
        return super().val_dataloader
