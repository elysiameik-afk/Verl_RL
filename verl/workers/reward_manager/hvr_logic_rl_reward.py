"""
HVR Logic RL Reward Manager

基于HVR (Hybrid Value Reshaping) 的奖励管理器，集成ERVF价值函数和混合价值重塑。
继承自LogicRLRewardManager，在获取稀疏奖励后应用HVR算法，输出重塑后的奖励。

核心创新：
1. ERVF (熵正则化价值函数) - 基于logits的内生价值估计
2. HVR (混合价值重塑) - 稀疏奖励指导的价值轨迹重塑
3. Z-score归一化 - 保持数值稳定性
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager.logic_rl_reward import LogicRLRewardManager, _select_rm_score_fn
from verl.workers.reward_manager.registry import register


def calculate_ervf_value(logits: torch.Tensor, alpha: float, beta: float) -> float:
    """
    计算基于ERVF的内生价值函数

    Args:
        logits: 一维张量 (vocab_size,) - 模型在某状态的原始logits
        alpha: 温度系数，调节logsumexp平滑度
        beta: 熵惩罚权重，调节不确定性惩罚力度

    Returns:
        V_ervf: 熵正则化后的内生价值
    """
    # 数值稳定化
    logits = torch.clamp(logits, min=-10, max=10)

    # 计算内生价值 V_endo
    V_endo = alpha * torch.logsumexp(logits / alpha, dim=-1)

    # 计算策略熵 H
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    H = -torch.sum(probs * log_probs, dim=-1)

    # 最终熵正则化价值
    V_ervf = V_endo - beta * H

    return V_ervf.item()


def calculate_hvr_rewards_for_group(
    group_data: List[Dict],
    alpha: float = 1.0,
    beta: float = 0.1,
    lambda_hvr: float = 0.5,
    use_zscore: bool = True,
    target_scale: float = 3.0
) -> Tuple[List[float], Dict[str, float]]:
    """
    为一组response计算HVR奖励

    Args:
        group_data: 组内所有response的数据
        alpha, beta, lambda_hvr: HVR算法超参数
        use_zscore: 是否使用Z-score归一化
        target_scale: Z-score后的目标标准差

    Returns:
        group_returns: 每个response的最终奖励
        metrics: 监控指标
    """
    group_returns = []
    metrics = {
        'v_ervf_values': [],
        'v_target_values': [],
        'v_hvr_values': [],
        'raw_returns': [],
        'r_final_values': [],
        'external_scores': []
    }

    for response_data in group_data:
        logits = response_data['logits']  # (seq_len, vocab_size)
        r_final = response_data['r_final']  # 外部稀疏奖励
        response_ids = response_data['ids']  # (seq_len,)
        external_score = response_data.get('external_score', r_final)  # 原始外部分数

        # 1. 计算V_ervf轨迹
        V_ervf_list = []
        for step_logits in logits:
            V_ervf = calculate_ervf_value(step_logits, alpha, beta)
            V_ervf_list.append(V_ervf)

        # 2. 序列级Z-score归一化 (稳定单个序列)
        if len(V_ervf_list) > 1 and use_zscore:
            V_ervf_tensor = torch.tensor(V_ervf_list)
            V_ervf_mean = V_ervf_tensor.mean()
            V_ervf_std = V_ervf_tensor.std() + 1e-8
            V_ervf_normalized = (V_ervf_tensor - V_ervf_mean) / V_ervf_std
        else:
            V_ervf_normalized = torch.tensor(V_ervf_list)

        # 3. 重塑目标 (直接使用外部奖励，不归一化)
        V_target = r_final * 0.3  # 适度缩放外部奖励影响

        # 4. 重塑价值轨迹
        V_hvr_list = [(1.0 - lambda_hvr) * v.item() + lambda_hvr * V_target
                      for v in V_ervf_normalized]

        # 5. 分解为稠密奖励
        r_hvr_list = []
        for t in range(len(V_hvr_list) - 1):
            # 计算当前token的log概率
            step_logits = logits[t]
            token_id = response_ids[t]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            log_prob_t = log_probs[token_id].item()

            # HVR稠密奖励公式
            r_hvr_t = alpha * log_prob_t + V_hvr_list[t] - V_hvr_list[t+1]
            # 控制单步奖励范围
            r_hvr_t = torch.clamp(torch.tensor(r_hvr_t), min=-1.0, max=1.0).item()
            r_hvr_list.append(r_hvr_t)

        # 6. 计算总回报
        total_hvr_reward = sum(r_hvr_list)
        raw_total_return = total_hvr_reward + r_final

        # 收集指标
        metrics['v_ervf_values'].extend(V_ervf_list)
        metrics['v_target_values'].append(V_target)
        metrics['v_hvr_values'].extend(V_hvr_list)
        metrics['raw_returns'].append(raw_total_return)
        metrics['r_final_values'].append(r_final)
        metrics['external_scores'].append(external_score)

        group_returns.append(raw_total_return)

    # 7. 组级Z-score归一化
    if len(group_returns) > 1 and use_zscore:
        returns_tensor = torch.tensor(group_returns)
        returns_mean = returns_tensor.mean()
        returns_std = returns_tensor.std() + 1e-8
        returns_normalized = (returns_tensor - returns_mean) / returns_std

        # 缩放到目标范围
        final_returns = (returns_normalized * target_scale).tolist()

        # 最终范围控制
        final_returns = [torch.clamp(torch.tensor(r), min=-6.0, max=6.0).item()
                        for r in final_returns]
    else:
        final_returns = [torch.clamp(torch.tensor(r), min=-6.0, max=6.0).item()
                        for r in group_returns]

    # 计算汇总指标
    summary_metrics = {
        'v_ervf_mean': np.mean(metrics['v_ervf_values']) if metrics['v_ervf_values'] else 0.0,
        'v_target_mean': np.mean(metrics['v_target_values']) if metrics['v_target_values'] else 0.0,
        'v_hvr_mean': np.mean(metrics['v_hvr_values']) if metrics['v_hvr_values'] else 0.0,
        'raw_return_mean': np.mean(metrics['raw_returns']) if metrics['raw_returns'] else 0.0,
        'raw_return_std': np.std(metrics['raw_returns']) if metrics['raw_returns'] else 0.0,
        'final_return_mean': np.mean(final_returns),
        'r_final_mean': np.mean(metrics['r_final_values']) if metrics['r_final_values'] else 0.0,
        'external_score_mean': np.mean(metrics['external_scores']) if metrics['external_scores'] else 0.0,
    }

    return final_returns, summary_metrics


@register("hvr_logic_rl")
class HVRLogicRLRewardManager(LogicRLRewardManager):
    """
    HVR Logic RL奖励管理器

    在LogicRL基础上集成HVR混合价值重塑机制：
    1. 获取稀疏奖励R_final (复用LogicRL的compute_score)
    2. 应用HVR奖励重塑 (ERVF + 价值重塑)
    3. Z-score归一化保持数值稳定
    4. 输出包含HVR信息的奖励张量
    """

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", **kwargs):
        super().__init__(tokenizer, num_examine, reward_fn_key, **kwargs)

        # HVR超参数 - 从kwargs中获取，提供默认值
        self.alpha = kwargs.get('alpha', 1.0)
        self.beta = kwargs.get('beta', 0.1)
        self.lambda_hvr = kwargs.get('lambda_hvr', 0.5)
        self.use_zscore = kwargs.get('use_zscore', True)
        self.target_scale = kwargs.get('target_scale', 3.0)

        print(f"🎯 [HVR初始化] alpha={self.alpha}, beta={self.beta}, lambda_hvr={self.lambda_hvr}")
        print(f"🎯 [HVR初始化] use_zscore={self.use_zscore}, target_scale={self.target_scale}")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        主要接口：计算HVR奖励
        """
        # 检查是否有logits数据
        if "logits" not in data.batch.keys() or data.batch["logits"] is None:
            print("⚠️  [HVR警告] 未找到logits数据或logits为None，回退到原始LogicRL")
            return super().__call__(data, return_dict)

        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            else:
                return data.batch["rm_scores"]

        print("🎯 [HVR] 开始计算混合价值重塑奖励...")

        # 初始化
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        all_hvr_metrics = {}

        # 获取基础数据
        logits = data.batch["logits"]  # (batch_size, seq_len, vocab_size)
        responses = data.batch["responses"]  # (batch_size, seq_len)

        # 按UID分组处理
        uid_to_indices = {}
        for i, uid in enumerate(data.non_tensor_batch["uid"]):
            if uid not in uid_to_indices:
                uid_to_indices[uid] = []
            uid_to_indices[uid].append(i)

        for uid, indices in uid_to_indices.items():
            group_data = []

            for idx in indices:
                # 获取response文本
                response_ids = responses[idx]
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # 计算外部奖励 (复用LogicRL逻辑)
                data_source = data.non_tensor_batch.get(self.reward_fn_key, ["unknown"])[idx]
                ground_truth = data.non_tensor_batch.get("ground_truth", [""])[idx]

                compute_score_fn = _select_rm_score_fn(data_source)
                external_score = compute_score_fn(response_str, ground_truth)

                # 准备HVR数据
                response_logits = logits[idx]  # (seq_len, vocab_size)
                group_data.append({
                    'logits': response_logits,
                    'ids': response_ids,
                    'r_final': external_score,
                    'external_score': external_score,
                    'index': idx
                })

            # 应用HVR算法
            hvr_returns, hvr_metrics = calculate_hvr_rewards_for_group(
                group_data, self.alpha, self.beta, self.lambda_hvr,
                self.use_zscore, self.target_scale
            )

            # 将HVR奖励分配到token级别
            for i, (data_item, hvr_return) in enumerate(zip(group_data, hvr_returns)):
                idx = data_item['index']
                # 将序列级奖励复制到所有token (保持与原LogicRL一致)
                reward_tensor[idx, :] = hvr_return

            # 收集指标
            for key, value in hvr_metrics.items():
                if key not in all_hvr_metrics:
                    all_hvr_metrics[key] = []
                all_hvr_metrics[key].append(value)

        # 汇总所有指标
        final_metrics = {}
        for key, values in all_hvr_metrics.items():
            if values:
                final_metrics[f"rewards/{key}"] = np.mean(values)

        # 打印关键指标
        if final_metrics:
            print(f"🎯 [HVR指标] V_ervf_mean: {final_metrics.get('rewards/v_ervf_mean', 0):.4f}")
            print(f"🎯 [HVR指标] V_target_mean: {final_metrics.get('rewards/v_target_mean', 0):.4f}")
            print(f"🎯 [HVR指标] final_return_mean: {final_metrics.get('rewards/final_return_mean', 0):.4f}")
            print(f"🎯 [HVR指标] external_score_mean: {final_metrics.get('rewards/external_score_mean', 0):.4f}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": final_metrics
            }
        else:
            return reward_tensor
