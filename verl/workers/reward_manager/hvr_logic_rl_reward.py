"""
HVR Logic RL Reward Manager

基于HVR (Hindsight Value Reshaping) 的奖励管理器，集成ERVF价值函数和后见之明价值重塑。
继承自LogicRLRewardManager，在获取稀疏奖励后应用HVR算法，输出重塑后的奖励。

核心创新：
1. ERVF (熵正则化价值函数) - 基于logits的内生价值估计
2. HVR (后见之明价值重塑) - 稀疏奖励指导的价值轨迹重塑
3. GRPO组间投票 - 保持组内相对优势计算的稳定性
"""

import torch
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager.logic_rl_reward import LogicRLRewardManager, _select_rm_score_fn
from verl.trainer.ppo.core_algos import (
    calculate_ervf_value,
    calculate_hvr_rewards_for_group,
    aggregate_hvr_metrics_dict,
    is_main_process
)
from verl.workers.reward_manager.registry import register


@register("hvr_logic_rl")
class HVRLogicRLRewardManager(LogicRLRewardManager):
    """
    HVR Logic RL奖励管理器

    在LogicRL基础上集成HVR内生奖励机制：
    1. 获取稀疏奖励R_final (复用LogicRL的compute_score)
    2. 应用HVR奖励重塑 (ERVF + 价值重塑)
    3. 计算GRPO组间优势 (保持稳定性)
    4. 输出包含HVR信息的奖励张量
    """

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", **kwargs):
        super().__init__(tokenizer, num_examine, reward_fn_key, **kwargs)

        # HVR参数配置 (从kwargs中获取)
        self.hvr_alpha = kwargs.get("hvr_alpha", 1.0)      # 温度系数
        self.hvr_beta = kwargs.get("hvr_beta", 0.1)        # 熵惩罚权重
        self.hvr_lambda = kwargs.get("hvr_lambda", 0.5)    # HVR混合因子

        # 指标记录
        self.hvr_metrics_history = []

        if is_main_process():
            print("🎯 [HVR Manager] 初始化HVR Logic RL奖励管理器")
            print(f"🎯 [HVR参数] α={self.hvr_alpha}, β={self.hvr_beta}, λ={self.hvr_lambda}")
            print("🎯 [HVR特性] ERVF价值函数 + 后见之明价值重塑 + GRPO组间投票")
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        HVR奖励管理器的主要接口

        工作流程：
        1. 调用父类获取稀疏奖励
        2. 检查是否有logits用于HVR计算
        3. 如果有logits，应用HVR重塑；否则回退到原始LogicRL
        4. 返回奖励张量和额外信息
        """
        if is_main_process():
            print("🎯 [HVR Manager] 开始HVR奖励计算")

        try:
            # 1. 首先调用父类获取基础奖励
            if "rm_scores" in data.batch.keys():
                # 已经有预计算的奖励，直接使用
                base_reward_tensor = data.batch["rm_scores"]
                reward_extra_info = {}

                if is_main_process():
                    print("🔍 [HVR Manager] 使用预计算的rm_scores")
            else:
                # 需要计算奖励
                base_result = super().__call__(data, return_dict=True)
                base_reward_tensor = base_result["reward_tensor"]
                reward_extra_info = base_result.get("reward_extra_info", {})

                if is_main_process():
                    print("🔍 [HVR Manager] 计算了新的奖励")

            # 2. 检查是否有rollout_log_probs用于HVR计算
            if "rollout_log_probs" in data.batch:
                rollout_log_probs = data.batch["rollout_log_probs"]

                if is_main_process():
                    print(f"🔍 [HVR Manager] 找到rollout_log_probs，形状: {rollout_log_probs.shape}")
                    print("🎯 [HVR Manager] 开始HVR重塑")

                # 3. 应用HVR重塑 (使用log_probs而不是logits)
                hvr_reward_tensor, hvr_extra_info = self._apply_hvr_to_rewards_with_logprobs(
                    data=data,
                    base_reward_tensor=base_reward_tensor,
                    rollout_log_probs=rollout_log_probs
                )

                # 4. 合并额外信息
                reward_extra_info.update(hvr_extra_info)

                if return_dict:
                    return {
                        "reward_tensor": hvr_reward_tensor,
                        "reward_extra_info": reward_extra_info
                    }
                else:
                    return hvr_reward_tensor

            else:
                if is_main_process():
                    print("⚠️ [HVR Manager] 未找到rollout_log_probs，回退到原始LogicRL")

                # 回退到原始LogicRL (确保列表格式)
                batch_size = base_reward_tensor.shape[0]
                reward_extra_info["hvr_applied"] = [False] * batch_size
                reward_extra_info["hvr_fallback_reason"] = ["no_rollout_log_probs"] * batch_size

                if return_dict:
                    return {
                        "reward_tensor": base_reward_tensor,
                        "reward_extra_info": reward_extra_info
                    }
                else:
                    return base_reward_tensor

        except Exception as e:
            if is_main_process():
                print(f"❌ [HVR Manager] HVR计算失败: {e}")
                print("   回退到原始LogicRL")

            # 完全回退到父类
            return super().__call__(data, return_dict)

    def _apply_hvr_to_rewards(self, data, base_reward_tensor, logits):
        """
        应用HVR重塑到奖励张量

        Args:
            data: DataProto对象
            base_reward_tensor: [batch_size, seq_len] 基础奖励张量
            logits: [batch_size, seq_len, vocab_size] logits张量

        Returns:
            tuple: (hvr_reward_tensor, hvr_extra_info)
        """
        # 1. 提取稀疏奖励
        sparse_rewards = self._extract_sparse_rewards_from_tensor(base_reward_tensor)

        if is_main_process():
            print(f"🔍 [HVR Manager] 稀疏奖励分布: {dict(zip(*np.unique(sparse_rewards, return_counts=True)))}")

        # 2. 准备组数据
        group_data = self._prepare_group_data_from_batch(data, logits, sparse_rewards)

        # 3. 计算HVR组回报
        group_returns, hvr_metrics = calculate_hvr_rewards_for_group(
            group_data=group_data,
            alpha=self.hvr_alpha,
            beta=self.hvr_beta,
            lambda_hvr=self.hvr_lambda
        )

        # 4. 计算GRPO组间优势
        mean_return = sum(group_returns) / len(group_returns)
        grpo_advantages = [ret - mean_return for ret in group_returns]

        # 5. 创建HVR奖励张量 (将序列级优势分配到token级)
        hvr_reward_tensor = torch.zeros_like(base_reward_tensor, dtype=torch.float32)

        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        # 获取response部分的mask
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]

        for i, seq_advantage in enumerate(grpo_advantages):
            # 将序列级优势分配给所有有效token
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) > 0:
                hvr_reward_tensor[i, valid_positions] = seq_advantage

        # 6. 聚合HVR指标
        aggregated_metrics = aggregate_hvr_metrics_dict(hvr_metrics)

        # 7. 构建额外信息 (确保所有值都是列表格式)
        batch_size = len(group_returns)
        hvr_extra_info = {
            "hvr_applied": [True] * batch_size,
            "hvr_group_return_mean": [mean_return] * batch_size,
            "hvr_group_return_std": [np.std(group_returns)] * batch_size,
            "hvr_grpo_advantage_mean": [np.mean(grpo_advantages)] * batch_size,
            "hvr_grpo_advantage_std": [np.std(grpo_advantages)] * batch_size,
            "hvr_sparse_rewards": sparse_rewards,  # 已经是列表
            "hvr_alpha": [self.hvr_alpha] * batch_size,
            "hvr_beta": [self.hvr_beta] * batch_size,
            "hvr_lambda": [self.hvr_lambda] * batch_size,
        }

        # 8. 添加HVR指标
        hvr_extra_info.update(aggregated_metrics)

        # 9. 记录指标历史
        self.hvr_metrics_history.append(aggregated_metrics)

        if is_main_process():
            print(f"✅ [HVR Manager] HVR重塑完成")
            print(f"   组平均回报: {mean_return:.4f}")
            print(f"   GRPO优势范围: [{min(grpo_advantages):.4f}, {max(grpo_advantages):.4f}]")
            print(f"   HVR成功率: {aggregated_metrics.get('hvr/success_rate', 0):.2f}")

        return hvr_reward_tensor, hvr_extra_info

    def _apply_hvr_to_rewards_with_logprobs(self, data, base_reward_tensor, rollout_log_probs):
        """
        使用rollout_log_probs应用HVR重塑 (适配vLLM rollout输出)

        Args:
            data: DataProto对象
            base_reward_tensor: [batch_size, seq_len] 基础奖励张量
            rollout_log_probs: [batch_size, seq_len] rollout的log概率

        Returns:
            tuple: (hvr_reward_tensor, hvr_extra_info)
        """
        # 1. 提取稀疏奖励
        sparse_rewards = self._extract_sparse_rewards_from_tensor(base_reward_tensor)

        if is_main_process():
            print(f"🔍 [HVR Manager] 稀疏奖励分布: {dict(zip(*np.unique(sparse_rewards, return_counts=True)))}")

        # 2. 准备组数据 (使用log_probs而不是logits)
        group_data = self._prepare_group_data_with_logprobs(data, rollout_log_probs, sparse_rewards)

        # 3. 计算HVR组回报 (使用简化的HVR算法)
        group_returns, hvr_metrics = self._calculate_hvr_returns_from_logprobs(
            group_data=group_data,
            alpha=self.hvr_alpha,
            beta=self.hvr_beta,
            lambda_hvr=self.hvr_lambda
        )

        # 4. 计算GRPO组间优势
        mean_return = sum(group_returns) / len(group_returns)
        grpo_advantages = [ret - mean_return for ret in group_returns]

        # 5. 创建HVR奖励张量
        hvr_reward_tensor = torch.zeros_like(base_reward_tensor, dtype=torch.float32)

        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        # 获取response部分的mask
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]

        for i, seq_advantage in enumerate(grpo_advantages):
            # 将序列级优势分配给所有有效token
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) > 0:
                hvr_reward_tensor[i, valid_positions] = seq_advantage

        # 6. 聚合HVR指标 (使用安全的聚合方式)
        try:
            aggregated_metrics = aggregate_hvr_metrics_dict(hvr_metrics)
        except Exception as e:
            if is_main_process():
                print(f"⚠️ [HVR Manager] 指标聚合失败: {e}，使用简化指标")
            # 使用简化的指标格式
            aggregated_metrics = {
                'hvr/success_rate': hvr_metrics['successful_count'] / hvr_metrics['total_count'],
                'hvr/total_sequences': hvr_metrics['total_count'],
                'hvr/successful_sequences': hvr_metrics['successful_count'],
            }

        # 7. 构建额外信息 (确保所有值都是列表格式)
        batch_size = len(group_returns)
        hvr_extra_info = {
            "hvr_applied": [True] * batch_size,
            "hvr_method": ["logprobs_based"] * batch_size,
            "hvr_group_return_mean": [mean_return] * batch_size,
            "hvr_group_return_std": [np.std(group_returns)] * batch_size,
            "hvr_grpo_advantage_mean": [np.mean(grpo_advantages)] * batch_size,
            "hvr_grpo_advantage_std": [np.std(grpo_advantages)] * batch_size,
            "hvr_sparse_rewards": sparse_rewards,  # 已经是列表
            "hvr_alpha": [self.hvr_alpha] * batch_size,
            "hvr_beta": [self.hvr_beta] * batch_size,
            "hvr_lambda": [self.hvr_lambda] * batch_size,
        }

        # 8. 添加HVR指标
        hvr_extra_info.update(aggregated_metrics)

        # 9. 记录指标历史
        self.hvr_metrics_history.append(aggregated_metrics)

        if is_main_process():
            print(f"✅ [HVR Manager] HVR重塑完成 (基于log_probs)")
            print(f"   组平均回报: {mean_return:.4f}")
            print(f"   GRPO优势范围: [{min(grpo_advantages):.4f}, {max(grpo_advantages):.4f}]")
            print(f"   HVR成功率: {aggregated_metrics.get('hvr/success_rate', 0):.2f}")

        return hvr_reward_tensor, hvr_extra_info

    def _prepare_group_data_with_logprobs(self, data, rollout_log_probs, sparse_rewards):
        """使用log_probs准备HVR组数据"""
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        # 获取response部分的数据
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]
        response_log_probs = rollout_log_probs  # [batch_size, response_len]

        group_data = []

        for i in range(len(sparse_rewards)):
            # 获取有效位置
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) == 0:
                if is_main_process():
                    print(f"⚠️ [HVR Manager] 序列{i}没有有效token，跳过")
                continue

            # 提取有效的log_probs和token IDs
            valid_log_probs = response_log_probs[i, valid_positions]  # [valid_len]
            valid_ids = responses[i, valid_positions]  # [valid_len]
            r_final = sparse_rewards[i]

            group_data.append({
                'log_probs': valid_log_probs,
                'ids': valid_ids,
                'r_final': r_final
            })

            if is_main_process() and i == 0:
                print(f"🔍 [HVR Manager] 序列{i}: 有效长度={len(valid_positions)}, R_final={r_final}")

        return group_data

    def _calculate_hvr_returns_from_logprobs(self, group_data, alpha, beta, lambda_hvr):
        """
        基于log_probs计算HVR回报 (简化版本)

        由于没有原始logits，我们使用简化的HVR算法：
        1. 使用log_probs作为价值的代理
        2. 应用价值重塑
        3. 计算总回报
        """
        group_returns = []
        hvr_metrics = {
            'ervf_values': [],      # 兼容aggregate_hvr_metrics_dict
            'entropies': [],        # 兼容aggregate_hvr_metrics_dict
            'hvr_rewards': [],      # 兼容aggregate_hvr_metrics_dict
            'r_finals': [],
            'v_targets': [],
            'sequence_lengths': [],
            'successful_count': 0,
            'total_count': len(group_data)
        }

        if is_main_process():
            print(f"🎯 [HVR Manager] 处理组数据: {len(group_data)} 个序列 (基于log_probs)")

        for seq_idx, d in enumerate(group_data):
            try:
                # 提取数据
                log_probs = d['log_probs']  # [sequence_length]
                ids = d['ids']              # [sequence_length]
                r_final = d['r_final']      # scalar

                sequence_length = log_probs.shape[0]

                if is_main_process() and seq_idx == 0:
                    print(f"🔍 [HVR Manager] 序列{seq_idx}: 长度={sequence_length}, R_final={r_final}")

                # 简化的HVR计算：使用log_probs作为价值代理
                # V_proxy(t) = alpha * log_prob(t) (简化的内生价值)
                v_proxy_list = [alpha * lp.item() for lp in log_probs]
                v_proxy_list.append(0.0)  # 终止状态

                # 价值重塑
                V_max = max(v_proxy_list[:-1])
                V_min = min(v_proxy_list[:-1])

                p = (r_final + 3.0) / 6.0
                p = max(0.0, min(1.0, p))
                V_target = (1.0 - p) * V_min + p * V_max

                # 重塑后价值
                V_hvr_list = [(1.0 - lambda_hvr) * v + lambda_hvr * V_target for v in v_proxy_list]

                # 计算稠密奖励
                r_hvr_list = []
                for t in range(sequence_length):
                    # 简化的HVR奖励：r_hvr_t = alpha * log_prob_t + V_hvr[t] - V_hvr[t+1]
                    r_hvr_t = alpha * log_probs[t].item() + V_hvr_list[t] - V_hvr_list[t + 1]
                    r_hvr_list.append(r_hvr_t)

                # 添加最终奖励
                r_hvr_list[-1] += r_final

                # 计算总回报
                total_return = sum(r_hvr_list)
                group_returns.append(total_return)

                # 收集指标 (兼容aggregate_hvr_metrics_dict格式)
                hvr_metrics['ervf_values'].extend(v_proxy_list[:-1])  # 使用v_proxy作为ERVF代理
                hvr_metrics['entropies'].extend([0.0] * sequence_length)  # 简化版没有熵计算
                hvr_metrics['hvr_rewards'].extend(r_hvr_list)
                hvr_metrics['r_finals'].append(r_final)
                hvr_metrics['v_targets'].append(V_target)
                hvr_metrics['sequence_lengths'].append(sequence_length)
                hvr_metrics['successful_count'] += 1

                if is_main_process() and seq_idx == 0:
                    print(f"✅ [HVR Manager] 序列{seq_idx}: 总回报={total_return:.4f}, V_target={V_target:.4f}")
                    print(f"   HVR奖励范围: [{min(r_hvr_list):.4f}, {max(r_hvr_list):.4f}]")

            except Exception as e:
                if is_main_process():
                    print(f"❌ [HVR Manager] 序列{seq_idx}处理失败: {e}")
                group_returns.append(0.0)

        if is_main_process():
            success_rate = hvr_metrics['successful_count'] / hvr_metrics['total_count']
            print(f"🎯 [HVR Manager] 组处理完成: 成功率={success_rate:.2f}, 平均回报={np.mean(group_returns):.4f}")

        return group_returns, hvr_metrics

    def _extract_sparse_rewards_from_tensor(self, reward_tensor):
        """从奖励张量中提取稀疏奖励"""
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

        return sparse_rewards

    def _prepare_group_data_from_batch(self, data, logits, sparse_rewards):
        """从batch数据准备HVR组数据"""
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        # 获取response部分的数据
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]
        response_logits = logits[:, -response_length:, :]  # [batch_size, response_len, vocab_size]

        group_data = []

        for i in range(len(sparse_rewards)):
            # 获取有效位置
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) == 0:
                if is_main_process():
                    print(f"⚠️ [HVR Manager] 序列{i}没有有效token，跳过")
                continue

            # 提取有效的logits和token IDs
            valid_logits = response_logits[i, valid_positions]  # [valid_len, vocab_size]
            valid_ids = responses[i, valid_positions]  # [valid_len]
            r_final = sparse_rewards[i]

            group_data.append({
                'logits': valid_logits,
                'ids': valid_ids,
                'r_final': r_final
            })

            if is_main_process() and i == 0:
                print(f"🔍 [HVR Manager] 序列{i}: 有效长度={len(valid_positions)}, R_final={r_final}")

        return group_data

    def get_hvr_metrics_summary(self):
        """获取HVR指标摘要 (用于最终分析)"""
        if not self.hvr_metrics_history:
            return {}

        # 聚合所有历史指标
        summary = {}

        # 收集所有数值指标
        all_metrics = defaultdict(list)
        for metrics in self.hvr_metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

        # 计算统计摘要
        for key, values in all_metrics.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)

        summary["hvr_total_batches"] = len(self.hvr_metrics_history)

        return summary
    
    def _extract_logits_from_rollout(self, rollout_output):
        """从rollout输出中提取logits"""
        # 检查可能的logits字段
        logits_fields = ["logits", "response_logits", "output_logits"]
        
        for field in logits_fields:
            if field in rollout_output:
                logits = rollout_output[field]
                if is_main_process():
                    print(f"🔍 [HVR Manager] 找到logits字段: {field}, 形状: {logits.shape}")
                return logits
        
        # 如果没有找到logits，抛出错误
        available_fields = list(rollout_output.keys())
        raise ValueError(f"未找到logits字段。可用字段: {available_fields}")
    
    def _extract_sparse_rewards(self, sparse_reward_result):
        """从LogicRL结果中提取稀疏奖励列表"""
        # 检查可能的奖励字段
        reward_fields = ["token_level_rewards", "token_level_scores", "rewards"]
        
        for field in reward_fields:
            if field in sparse_reward_result:
                reward_tensor = sparse_reward_result[field]  # [batch_size, seq_len]
                
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
                
                if is_main_process():
                    print(f"🔍 [HVR Manager] 从{field}提取稀疏奖励: {sparse_rewards}")
                
                return sparse_rewards
        
        # 如果没有找到奖励字段，使用零奖励
        batch_size = len(rollout_output.get("responses", [1]))
        if is_main_process():
            print(f"⚠️ [HVR Manager] 未找到稀疏奖励字段，使用零奖励")
            print(f"   可用字段: {list(sparse_reward_result.keys())}")
        
        return [0.0] * batch_size

    def _apply_hvr_reshaping(self, rollout_output, logits, sparse_rewards):
        """
        应用HVR奖励重塑的核心方法

        Args:
            rollout_output: rollout输出数据
            logits: [batch_size, seq_len, vocab_size] - 模型logits
            sparse_rewards: List[float] - 稀疏奖励R_final列表

        Returns:
            tuple: (hvr_advantages, hvr_metrics)
        """
        if is_main_process():
            print(f"🎯 [HVR Manager] 开始HVR重塑, 组大小: {len(sparse_rewards)}")

        # 1. 准备组数据
        group_data = self._prepare_group_data(rollout_output, logits, sparse_rewards)

        # 2. 计算HVR组回报
        group_returns, hvr_metrics = calculate_hvr_rewards_for_group(
            group_data=group_data,
            alpha=self.hvr_alpha,
            beta=self.hvr_beta,
            lambda_hvr=self.hvr_lambda
        )

        # 3. 计算GRPO组间优势 (保持GRPO的核心稳定性)
        mean_return = sum(group_returns) / len(group_returns)
        grpo_advantages = [ret - mean_return for ret in group_returns]

        # 4. 将序列级优势扩展到token级
        responses = rollout_output["responses"]
        attention_mask = rollout_output["attention_mask"]

        # 获取response部分的mask
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]

        # 创建token级优势张量
        hvr_advantages = torch.zeros_like(response_mask, dtype=torch.float32)

        for i, seq_advantage in enumerate(grpo_advantages):
            # 将序列级优势分配给所有有效token
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) > 0:
                hvr_advantages[i, valid_positions] = seq_advantage

        # 5. 聚合HVR指标
        aggregated_metrics = aggregate_hvr_metrics_dict(hvr_metrics)

        # 6. 添加GRPO和管理器级别的指标
        aggregated_metrics.update({
            'hvr/group_return_mean': mean_return,
            'hvr/group_return_std': np.std(group_returns),
            'hvr/grpo_advantage_mean': np.mean(grpo_advantages),
            'hvr/grpo_advantage_std': np.std(grpo_advantages),
            'hvr/manager_alpha': self.hvr_alpha,
            'hvr/manager_beta': self.hvr_beta,
            'hvr/manager_lambda': self.hvr_lambda,
            'hvr/token_advantage_mean': hvr_advantages[response_mask > 0].mean().item(),
            'hvr/token_advantage_std': hvr_advantages[response_mask > 0].std().item(),
        })

        if is_main_process():
            print(f"✅ [HVR Manager] HVR重塑完成")
            print(f"   组平均回报: {mean_return:.4f}")
            print(f"   GRPO优势范围: [{min(grpo_advantages):.4f}, {max(grpo_advantages):.4f}]")
            print(f"   Token优势范围: [{hvr_advantages[response_mask > 0].min().item():.4f}, {hvr_advantages[response_mask > 0].max().item():.4f}]")

        return hvr_advantages, aggregated_metrics

    def _prepare_group_data(self, rollout_output, logits, sparse_rewards):
        """为HVR准备组数据"""
        responses = rollout_output["responses"]
        attention_mask = rollout_output["attention_mask"]

        # 获取response部分的数据
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]
        response_logits = logits[:, -response_length:, :]  # [batch_size, response_len, vocab_size]

        group_data = []

        for i in range(len(sparse_rewards)):
            # 获取有效位置
            valid_positions = torch.where(response_mask[i] > 0)[0]
            if len(valid_positions) == 0:
                if is_main_process():
                    print(f"⚠️ [HVR Manager] 序列{i}没有有效token，跳过")
                continue

            # 提取有效的logits和token IDs
            valid_logits = response_logits[i, valid_positions]  # [valid_len, vocab_size]
            valid_ids = responses[i, valid_positions]  # [valid_len]
            r_final = sparse_rewards[i]

            group_data.append({
                'logits': valid_logits,
                'ids': valid_ids,
                'r_final': r_final
            })

            if is_main_process() and i == 0:
                print(f"🔍 [HVR Manager] 序列{i}: 有效长度={len(valid_positions)}, R_final={r_final}")

        return group_data

    def get_hvr_metrics_summary(self):
        """获取HVR指标摘要 (用于最终分析)"""
        if not self.hvr_metrics_history:
            return {}

        # 聚合所有历史指标
        summary = {}

        # 收集所有数值指标
        all_metrics = {}
        for metrics in self.hvr_metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # 计算统计摘要
        for key, values in all_metrics.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)

        summary["hvr_total_batches"] = len(self.hvr_metrics_history)

        return summary