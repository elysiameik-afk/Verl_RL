"""
HVR专用Actor实现

基于DataParallel Actor，但专门为HVR内生奖励机制设计：
1. 保存完整的logits用于ERVF计算
2. 直接使用HVR奖励进行策略训练
3. 移除critic相关逻辑，简化训练流程
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.hvr.hvr_core_algos import (
    calculate_hvr_rewards, 
    hvr_policy_loss,
    aggregate_hvr_metrics,
    HVRMetrics
)
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.core_algos import is_main_process
from verl.utils.torch_functional import logprobs_from_logits


class HVRActor(DataParallelPPOActor):
    """
    HVR专用Actor
    
    继承DataParallelPPOActor但专门为HVR设计：
    - 保存logits用于内生价值计算
    - 使用HVR奖励替代GRPO优势估计
    - 简化的训练流程（无critic）
    """
    
    def __init__(self, config, device_name="cuda"):
        super().__init__(config, device_name)
        
        # HVR专用配置
        self.hvr_alpha = self.config.get("hvr_alpha", 1.0)
        self.hvr_beta = self.config.get("hvr_beta", 0.1)
        self.hvr_lambda = self.config.get("hvr_lambda", 0.5)
        self.hvr_cliprange = self.config.get("hvr_cliprange", 0.2)
        
        if is_main_process():
            print(f"🎯 [HVR Actor] 初始化完成")
            print(f"🎯 [HVR参数] α={self.hvr_alpha}, β={self.hvr_beta}, λ={self.hvr_lambda}")
    
    def _forward_micro_batch_with_logits_hvr(
        self, 
        micro_batch, 
        temperature, 
        calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        HVR专用的前向传播，返回logits用于内生价值计算
        
        Returns:
            entropy: [batch_size, response_len] 
            log_probs: [batch_size, response_len]
            logits: [batch_size, response_len, vocab_size] - 用于HVR
        """
        response_length = micro_batch["responses"].size(-1)
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            
            # HVR要求非remove_padding模式以获取完整logits
            if self.use_remove_padding:
                raise ValueError("HVR Actor不支持remove_padding模式，请设置use_remove_padding=False")
            
            # 前向传播获取logits
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            
            # 处理logits和概率
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # [batch_size, response_length, vocab_size]
            
            # 计算log概率
            log_probs = logprobs_from_logits(logits, micro_batch["responses"])
            
            # 计算熵（如果需要）
            entropy = None
            if calculate_entropy:
                entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            
            return entropy, log_probs, logits
    
    def extract_sparse_rewards(self, data) -> torch.Tensor:
        """
        从数据中提取稀疏奖励R_final
        
        Args:
            data: 训练数据批次
            
        Returns:
            sparse_rewards: [batch_size] - 每个序列的稀疏奖励
        """
        # 尝试从不同字段获取原始奖励
        reward_fields = ["token_level_rewards", "token_level_scores", "rewards"]
        
        for field in reward_fields:
            if hasattr(data, 'batch') and field in data.batch:
                reward_tensor = data.batch[field]  # [batch_size, seq_len]
                
                # 提取每个序列最后一个非零位置的奖励
                sparse_rewards = []
                for i in range(reward_tensor.shape[0]):
                    nonzero_indices = torch.nonzero(reward_tensor[i]).flatten()
                    if len(nonzero_indices) > 0:
                        # 取最后一个非零位置的奖励作为R_final
                        last_reward_pos = nonzero_indices[-1]
                        r_final = reward_tensor[i, last_reward_pos].item()
                        sparse_rewards.append(r_final)
                    else:
                        # 如果没有非零奖励，使用0
                        sparse_rewards.append(0.0)
                
                return torch.tensor(sparse_rewards, device=reward_tensor.device)
        
        # 如果找不到奖励字段，返回零奖励
        batch_size = data["responses"].shape[0]
        if is_main_process():
            print("⚠️ [HVR] 未找到稀疏奖励字段，使用零奖励")
        return torch.zeros(batch_size, device=data["responses"].device)
    
    def update_policy(self, data) -> Dict[str, Any]:
        """
        HVR专用的策略更新
        
        使用内生奖励机制，无需critic和GAE优势估计
        """
        metrics = {}
        
        # 基础数据
        responses = data["responses"]
        response_mask = data["attention_mask"][:, -responses.size(-1):]
        batch_size = responses.shape[0]
        
        # 提取稀疏奖励
        sparse_rewards = self.extract_sparse_rewards(data)
        
        if is_main_process():
            print(f"🎯 [HVR] 处理批次: batch_size={batch_size}")
            print(f"🎯 [HVR] 稀疏奖励分布: {sparse_rewards.tolist()}")
        
        # 前向传播获取logits
        entropy, log_prob, response_logits = self._forward_micro_batch_with_logits_hvr(
            micro_batch=data, 
            temperature=1.0,  # HVR使用固定温度
            calculate_entropy=True
        )
        
        # 计算HVR奖励
        hvr_rewards_batch = []
        hvr_metrics_list = []
        
        for i in range(batch_size):
            # 获取有效位置
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) == 0:
                continue
            
            # 获取当前序列的数据
            valid_logits = response_logits[i, valid_positions]  # [valid_len, vocab_size]
            valid_ids = responses[i, valid_positions]  # [valid_len]
            r_final = sparse_rewards[i].item()
            
            try:
                # 计算HVR奖励
                hvr_rewards, hvr_metrics = calculate_hvr_rewards(
                    response_logits=valid_logits,
                    response_ids=valid_ids,
                    R_final=r_final,
                    alpha=self.hvr_alpha,
                    beta=self.hvr_beta,
                    lambda_hvr=self.hvr_lambda,
                )
                
                # 存储结果
                hvr_rewards_full = torch.zeros_like(response_mask[i], dtype=torch.float32)
                hvr_rewards_full[valid_positions] = hvr_rewards
                hvr_rewards_batch.append(hvr_rewards_full)
                hvr_metrics_list.append(hvr_metrics)
                
            except Exception as e:
                if is_main_process():
                    print(f"⚠️ [HVR] 序列{i}计算失败: {e}")
                # 使用零奖励作为fallback
                hvr_rewards_batch.append(torch.zeros_like(response_mask[i], dtype=torch.float32))
        
        if not hvr_rewards_batch:
            if is_main_process():
                print("⚠️ [HVR] 没有成功计算的HVR奖励")
            return {"loss": torch.tensor(0.0)}
        
        # 转换为批次张量
        hvr_rewards_tensor = torch.stack(hvr_rewards_batch)
        
        # 计算策略损失
        policy_loss, policy_metrics = hvr_policy_loss(
            log_probs=log_prob,
            hvr_rewards=hvr_rewards_tensor,
            response_mask=response_mask,
            cliprange=self.hvr_cliprange,
            loss_agg_mode="token-mean",
        )
        
        # 聚合HVR指标
        hvr_aggregated_metrics = aggregate_hvr_metrics(hvr_metrics_list)
        
        # 合并所有指标
        metrics.update(policy_metrics)
        metrics.update(hvr_aggregated_metrics)
        
        # 添加基础指标
        metrics.update({
            "hvr/batch_size": batch_size,
            "hvr/sparse_reward_mean": sparse_rewards.mean().item(),
            "hvr/sparse_reward_std": sparse_rewards.std().item(),
            "hvr/alpha": self.hvr_alpha,
            "hvr/beta": self.hvr_beta,
            "hvr/lambda": self.hvr_lambda,
        })
        
        if is_main_process():
            print(f"🎯 [HVR] 策略损失: {policy_loss.item():.6f}")
            print(f"🎯 [HVR] HVR奖励均值: {hvr_rewards_tensor.mean().item():.6f}")
            print(f"🎯 [HVR] 成功率: {hvr_aggregated_metrics.get('hvr/success_rate', 0):.2f}")
        
        # 反向传播
        self.actor_module.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪
        if hasattr(self, 'grad_clip') and self.grad_clip > 0:
            if isinstance(self.actor_module, FSDP):
                self.actor_module.clip_grad_norm_(self.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.grad_clip)
        
        # 优化器步骤
        self.actor_optim.step()
        
        metrics["loss"] = policy_loss.item()
        return metrics

    def generate_sequences(self, prompts, **generation_kwargs):
        """
        生成序列（复用父类实现）
        """
        return super().generate_sequences(prompts, **generation_kwargs)
