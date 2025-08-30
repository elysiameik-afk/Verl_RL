# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_with_innovations, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        
        # Initialize all innovation configs
        self.use_ema_smoothing = self.config.get("use_ema_smoothing", False)
        self.ema_beta = self.config.get("ema_beta", 0.9)
        self.use_gradient_adaptive_weighting = self.config.get("use_gradient_adaptive_weighting", False)
        self.gradient_weighting_temperature = self.config.get("gradient_weighting_temperature", 1.0)
        self.use_amic = self.config.get("use_amic", False)
        self.use_ptrw = self.config.get("use_ptrw", False)
        self.ptrw_sigma = self.config.get("ptrw_sigma", 0.2)
        self.use_temporal_decay = self.config.get("use_temporal_decay", False)
        self.temporal_decay_gamma = self.config.get("temporal_decay_gamma", 0.95)
        self.temporal_decay_normalize = self.config.get("temporal_decay_normalize", True)
        self.temporal_decay_use_lspd = self.config.get("temporal_decay_use_lspd", True)
        self.temporal_decay_lspd_alpha = self.config.get("temporal_decay_lspd_alpha", 2.0)
        self.temporal_decay_lspd_tau = self.config.get("temporal_decay_lspd_tau", 10.0)
        self.use_sca = self.config.get("use_sca", False)
        self.sca_answer_credit_ratio = self.config.get("sca_answer_credit_ratio", 0.3)
        self.sca_structure_credit_ratio = self.config.get("sca_structure_credit_ratio", 0.2)
        self.sca_process_credit_ratio = self.config.get("sca_process_credit_ratio", 0.5)
        self.use_asymmetric_clipping = self.config.get("use_asymmetric_clipping", False)
        self.clip_ratio_pos = self.config.get("clip_ratio_pos", 0.3)
        self.clip_ratio_neg = self.config.get("clip_ratio_neg", 0.1)

        # Initialize confidence calculation parameters
        self.lgc_window_size = 256
        self.lgc_avg_pool = torch.nn.AvgPool1d(kernel_size=self.lgc_window_size, stride=1)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"🎯 [创新点配置] EMA={self.use_ema_smoothing}, 梯度自适应={self.use_gradient_adaptive_weighting}, AMIC={self.use_amic}")
            print(f"🎯 [创新点配置] PTRW={self.use_ptrw}, 时序衰减={self.use_temporal_decay}, 非对称裁剪={self.use_asymmetric_clipping}")
            print(f"🎯 [自信度配置] LGC窗口大小={self.lgc_window_size}")

    def _compute_token_confidence_from_logits(self, logits: torch.Tensor, sampled_tokens: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """
        计算token级别的置信度，使用DeepConf方法：排除实际采样的token，计算剩余top-k token的平均log概率

        Args:
            logits: 形状 (..., vocab_size)
            sampled_tokens: 实际采样的token，形状与logits前导维度相同 (...)
            top_k: 考虑的top token数量，默认20

        Returns:
            token置信度，形状与logits前导维度相同 (...)
        """
        with torch.no_grad():  # 确保不计算梯度，节省显存
            # 计算log概率（原地操作，节省显存）
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (..., vocab_size)

            # 获取top-k的值和索引
            top_k_values, top_k_indices = torch.topk(log_probs, k=top_k, dim=-1)  # (..., top_k)

            # 立即释放大的log_probs张量
            del log_probs

            # 找到实际采样token在top-k中的位置
            sampled_tokens_expanded = sampled_tokens.unsqueeze(-1)  # (..., 1)
            mask = (top_k_indices == sampled_tokens_expanded)  # (..., top_k)

            # 计算排除采样token后的平均log概率
            # 使用更高效的计算方式
            valid_values = top_k_values.masked_fill(mask, 0.0)  # 将采样token设为0
            valid_count = (~mask).sum(dim=-1, keepdim=True).float()  # 有效token数量

            # 计算平均值
            confidence = valid_values.sum(dim=-1) / valid_count.squeeze(-1)  # (...)

            # 取反得到置信度分数（越高越自信）
            return -confidence

    def _compute_lgc_from_token_confidence(self, token_confidence: torch.Tensor) -> torch.Tensor:
        """
        从token置信度计算序列级别的LGC分数

        Args:
            token_confidence: 形状 (batch_size, response_len)

        Returns:
            LGC分数，形状 (batch_size,)
        """
        batch_size, response_len = token_confidence.shape

        # 边缘情况：如果序列长度小于窗口大小，直接计算平均
        if response_len < self.lgc_window_size:
            return token_confidence.mean(dim=-1)  # (batch_size,)

        # 计算组置信度：使用滑动窗口
        # 增加通道维度用于AvgPool1d
        token_confidence_expanded = token_confidence.unsqueeze(1)  # (batch_size, 1, response_len)

        # 应用滑动平均池化
        group_confidence = self.lgc_avg_pool(token_confidence_expanded)  # (batch_size, 1, num_groups)
        group_confidence = group_confidence.squeeze(1)  # (batch_size, num_groups)

        # 计算LGC：使用截尾平均（去掉极值后取平均）
        sorted_groups = torch.sort(group_confidence, dim=-1)[0]  # (batch_size, num_groups)
        num_groups = sorted_groups.shape[-1]
        start_idx = int(num_groups * 0.15)  # 去掉最小15%
        end_idx = int(num_groups * 0.85)    # 去掉最大15%

        # 如果组数太少，直接取平均
        if end_idx <= start_idx:
            lgc_scores = group_confidence.mean(dim=-1)  # (batch_size,)
        else:
            lgc_scores = sorted_groups[:, start_idx:end_idx].mean(dim=-1)  # (batch_size,)

        # 线性映射到[1.0, 2.0]范围
        lgc_scores = self._linear_scale_confidence(lgc_scores)

        # 调试：检查重复值
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            unique_scores = torch.unique(lgc_scores)
            if len(unique_scores) < len(lgc_scores) * 0.8:  # 如果80%以上都是重复值
                print(f"🔍 [自信度调试] 发现大量重复LGC分数!")
                print(f"🔍 [自信度调试] 唯一值数量: {len(unique_scores)}, 总样本数: {len(lgc_scores)}")
                print(f"🔍 [自信度调试] 唯一值: {unique_scores[:10]}")  # 只打印前10个
                print(f"🔍 [自信度调试] response_len={response_len}, lgc_window_size={self.lgc_window_size}")

        return lgc_scores

    def _linear_scale_confidence(self, confidences: torch.Tensor, target_range=(1.0, 2.0), min_spread=0.1) -> torch.Tensor:
        """
        将置信度线性缩放到指定范围

        Args:
            confidences: 原始置信度分数，形状 (batch_size,)
            target_range: 目标缩放范围，默认(1.0, 2.0)
            min_spread: 最小分布范围，避免除零

        Returns:
            缩放后的置信度，形状 (batch_size,)
        """
        min_conf = confidences.min()
        max_conf = confidences.max()

        # 如果置信度分布太窄，使用中间值
        if max_conf - min_conf < min_spread:
            target_mean = (target_range[0] + target_range[1]) / 2  # 1.5
            return torch.ones_like(confidences) * target_mean

        # 线性缩放到目标范围
        target_min, target_max = target_range
        scaled = target_min + (confidences - min_conf) / (max_conf - min_conf) * (target_max - target_min)

        return scaled

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, calculate_confidence=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len) or None
            log_probs: # (bs, response_len)
            confidence: # (bs,) or None - sequence-level confidence scores
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # rmpad情况下不支持自信度计算
                confidence = None

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    confidence = None  # fused_kernels不支持自信度计算

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    # 先计算entropy，立即释放中间张量
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        torch.cuda.empty_cache()  # 立即释放entropy计算的中间张量

                    # 再计算自信度，避免与entropy计算的显存叠加
                    confidence = None
                    if calculate_confidence:
                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                            print(f"🔍 [调试] 开始计算自信度...")

                        # 计算token级别的置信度
                        responses = micro_batch["responses"]  # (bsz, response_length)
                        token_confidence = self._compute_token_confidence_from_logits(logits, responses)  # (bsz, response_length)

                        # 计算序列级别的LGC分数
                        confidence = self._compute_lgc_from_token_confidence(token_confidence)  # (bsz,)

                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                            print(f"🔍 [调试] 自信度计算完成: shape={confidence.shape if confidence is not None else None}")

                        # 清理中间变量，释放显存
                        del token_confidence
                        torch.cuda.empty_cache()

            return entropy, log_probs, confidence

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"WARN: rank {rank} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> DataProto:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        confidence_lst = []

        # 在compute_log_prob中启用自信度计算（如果配置允许）
        # 优先从actor自己的配置中读取，如果没有则从meta_info中读取（兼容性）
        calculate_confidence = (
            self.config.get("algorithm", {}).get("use_confidence_scaling", False) or
            data.meta_info.get("use_confidence_scaling", False)
        )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"🔍 [调试] compute_log_prob: calculate_confidence={calculate_confidence}")
            print(f"🔍 [调试] actor config有algorithm: {hasattr(self.config, 'algorithm') or 'algorithm' in self.config}")
            if hasattr(self.config, 'algorithm') or 'algorithm' in self.config:
                print(f"🔍 [调试] algorithm.use_confidence_scaling: {self.config.get('algorithm', {}).get('use_confidence_scaling', 'NOT_FOUND')}")

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, confidence = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    calculate_confidence=calculate_confidence
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if calculate_confidence and confidence is not None:
                confidence_lst.append(confidence)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        confidences = None

        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_confidence and confidence_lst:
            confidences = torch.concat(confidence_lst, dim=0)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"🔍 [调试] 成功计算自信度: shape={confidences.shape}, values={confidences[:3] if len(confidences) > 0 else 'empty'}")
        elif calculate_confidence:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"🔍 [调试] 自信度计算失败: confidence_lst长度={len(confidence_lst)}")

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            if confidences is not None:
                confidences = confidences[revert_indices]

        # 构造返回的DataProto对象
        tensors = {"old_log_probs": log_probs}
        if entropys is not None:
            tensors["entropys"] = entropys
        if confidences is not None:
            tensors["confidences"] = confidences

        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={"temperature": temperature}
        )

        return output

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs or "uid" in data.non_tensor_batch:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = []
            if has_multi_modal_inputs:
                non_tensor_select_keys.append("multi_modal_inputs")
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs or ("uid" in data.non_tensor_batch if hasattr(data, 'non_tensor_batch') else False):
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    if hasattr(mini_batch, 'split'):
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    else:
                        # mini_batch is DataProto, use chunk instead
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = mini_batch.chunk(num_micro_batches)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    # 在update_policy中不计算自信度
                    entropy, log_prob, _ = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        calculate_confidence=False
                    )

                    # Use comprehensive innovation-enabled policy loss computation
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, innovation_metrics = compute_policy_loss_with_innovations(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        token_ids=responses,  # 传递token_ids用于SCA
                        # 创新点配置
                        use_ema_smoothing=self.use_ema_smoothing,
                        ema_beta=self.ema_beta,
                        use_gradient_adaptive_weighting=self.use_gradient_adaptive_weighting,
                        gradient_weighting_temperature=self.gradient_weighting_temperature,
                        use_amic=self.use_amic,
                        use_ptrw=self.use_ptrw,
                        ptrw_sigma=self.ptrw_sigma,
                        use_temporal_decay=self.use_temporal_decay,
                        temporal_decay_gamma=self.temporal_decay_gamma,
                        temporal_decay_normalize=self.temporal_decay_normalize,
                        temporal_decay_use_lspd=self.temporal_decay_use_lspd,
                        temporal_decay_lspd_alpha=self.temporal_decay_lspd_alpha,
                        temporal_decay_lspd_tau=self.temporal_decay_lspd_tau,
                        use_sca=self.use_sca,
                        sca_answer_credit_ratio=self.sca_answer_credit_ratio,
                        sca_structure_credit_ratio=self.sca_structure_credit_ratio,
                        sca_process_credit_ratio=self.sca_process_credit_ratio,
                        use_asymmetric_clipping=self.use_asymmetric_clipping,
                        clip_ratio_pos=self.clip_ratio_pos,
                        clip_ratio_neg=self.clip_ratio_neg,
                    )

                    # Add innovation metrics to the metrics dictionary
                    append_to_dict(metrics, innovation_metrics)

                    # 打印关键指标
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        active_innovations = []
                        if self.use_ema_smoothing:
                            active_innovations.append(f"EMA(β={self.ema_beta})")
                        if self.use_gradient_adaptive_weighting:
                            active_innovations.append(f"梯度自适应(T={self.gradient_weighting_temperature})")
                        if self.use_amic:
                            active_innovations.append("AMIC")
                        if self.use_ptrw:
                            active_innovations.append(f"PTRW(σ={self.ptrw_sigma})")
                        if self.use_temporal_decay:
                            active_innovations.append(f"时序衰减(γ={self.temporal_decay_gamma})")
                        if self.use_asymmetric_clipping:
                            active_innovations.append(f"非对称裁剪({self.clip_ratio_pos}/{self.clip_ratio_neg})")

                        if active_innovations:
                            print(f"🎯 [创新点激活] {', '.join(active_innovations)}")

                            # 打印关键指标
                            key_metrics = []
                            if 'ema/variance_reduction_ratio' in innovation_metrics:
                                key_metrics.append(f"方差降低={innovation_metrics['ema/variance_reduction_ratio']:.4f}")
                            if 'gradient_adaptive/weight_variance' in innovation_metrics:
                                key_metrics.append(f"梯度权重方差={innovation_metrics['gradient_adaptive/weight_variance']:.4f}")
                            if 'amic/sequence_weights_variance' in innovation_metrics:
                                key_metrics.append(f"AMIC方差={innovation_metrics['amic/sequence_weights_variance']:.4f}")
                            if 'ptrw/trust_weights_mean' in innovation_metrics:
                                key_metrics.append(f"PTRW信任度={innovation_metrics['ptrw/trust_weights_mean']:.4f}")
                            if 'temporal_decay/weight_sum' in innovation_metrics:
                                key_metrics.append(f"时序权重和={innovation_metrics['temporal_decay/weight_sum']:.4f}")
                            if 'asymmetric/pos_advantage_ratio' in innovation_metrics:
                                key_metrics.append(f"正优势比例={innovation_metrics['asymmetric/pos_advantage_ratio']:.4f}")

                            if key_metrics:
                                print(f"🎯 [关键指标] {', '.join(key_metrics)}")

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
