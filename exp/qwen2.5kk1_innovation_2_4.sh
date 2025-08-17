#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "🎯 开始创新点2.4测试: 概率性信任区域加权 (PTRW)..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/train.parquet \
    data.val_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/root/autodl-tmp/myverl/mymodels/qwen3-0.6b \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=Qwen2.5-0.5-Innovation2.4 \
    trainer.experiment_name=GRPO_PTRW_Test \
    trainer.n_gpus_per_node=1 \
    trainer.default_local_dir=/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/GRPO_PTRW_Test \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=1 \
    trainer.total_epochs=8 \
    reward_model.reward_manager=logic_rl \
    actor_rollout_ref.actor.use_ema_smoothing=False \
    actor_rollout_ref.actor.use_gradient_adaptive_weighting=False \
    actor_rollout_ref.actor.use_amic=False \
    actor_rollout_ref.actor.use_ptrw=True \
    actor_rollout_ref.actor.ptrw_sigma=0.2 \
    actor_rollout_ref.actor.use_temporal_decay=False \
    actor_rollout_ref.actor.use_asymmetric_clipping=False

echo "🎉 创新点2.4测试完成！"
