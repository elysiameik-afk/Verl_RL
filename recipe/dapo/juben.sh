#!/usr/bin/env bash
set -xeuo pipefail

# ==============================================================================
# 变量配置 (针对 32GB 显存优化)
# ==============================================================================
project_name='DAPO'
exp_name='DAPO-Qwen3-0.6b-juben'

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# 内存相关参数，这是一个针对 32GB 显存的均衡设置
# 如果依然 OOM，可以首先考虑降低 max_response_length
max_prompt_length=$((1024 * 4))    # 4096
max_response_length=$((1024 * 4))  # 4096, 总序列长度 8k
n_resp_per_prompt=4                # 从 8 降到 4，显著降低显存占用

train_prompt_bsz=2
gen_prompt_bsz=$((train_prompt_bsz * 2))
train_prompt_mini_bsz=2

enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
enable_filter_groups=False
max_num_gen_batches=10

# 路径配置 (请确保这些路径在你的机器上是正确的)
WORKING_DIR=${PWD}
NNODES=1
HOME_DIR=${HOME}
RAY_DATA_HOME="${HOME_DIR}/autodl-tmp/myverl"
MODEL_PATH="${RAY_DATA_HOME}/mymodels/qwen3-0.6b"
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"
TRAIN_FILE="${RAY_DATA_HOME}/data/juben/juben.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/juben/juben.parquet"
mkdir -p "${CKPTS_DIR}"

# 算法配置
temperature=1.0
top_p=1.0
top_k=-1
offload=True
use_dynamic_bsz=True

# ==============================================================================
# 直接运行 Python 脚本
# ==============================================================================

# 手动设置环境变量
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export HF_HUB_OFFLINE=1
export WANDB_API_KEY="9426da0e81fb52759c616fe8cbe799318108e980"
# 新增：为 LLMQualityRewardManager 添加 Gemini API 密钥
# 请将 "YOUR_GEMINI_API_KEY" 替换为您自己的真实密钥
export GEMINI_API_KEY= "AIzaSyCRTfMX1QSXItSiP6VjwRXyp6wWWnZEUYI"

echo "Starting DAPO training directly with Python..."
echo "Model Path: ${MODEL_PATH}"
echo "Train File: ${TRAIN_FILE}"
echo "Checkpoints will be saved to: ${CKPTS_DIR}"

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    critic.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=llm_quality_reward \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=12 \
    trainer.save_freq=12 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    \
    # ========================================================================== #
    #                  ↓↓↓ 核心修正和补充 ↓↓↓
    # ========================================================================== #
    \
    # 1. 明确为 Actor 和 Critic 设置 per-GPU 微批次大小，以解决核心报错
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    \
    # 2. 为 rollout 和 ref 也设置相应的 micro batch size，保持配置一致性
    #    它们的值通常可以设为 null，让框架自动推断，但明确指定更安全
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1