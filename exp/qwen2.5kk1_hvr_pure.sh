#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "🎯 开始HVR纯净测试: 内生奖励机制 (Hindsight Value Reshaping)..."
echo "🎯 [HVR特性] ERVF熵正则化价值函数 + 后见之明价值重塑"
echo "🎯 [HVR优势] 无需critic网络，基于模型自身logits的内生价值估计"

python3 -m verl.trainer.main_hvr \
    data.train_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/train.parquet \
    data.val_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/root/autodl-tmp/myverl/mymodels/qwen3-0.6b \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    trainer.project_name=Qwen2.5-0.5-HVR-Pure \
    trainer.experiment_name=HVR_ERVF_Pure_Test \
    trainer.n_gpus_per_node=1 \
    trainer.default_local_dir=/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/HVR_Pure \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=1 \
    trainer.total_epochs=8 \
    reward_model.reward_manager=logic_rl \
    actor_rollout_ref.actor.hvr_alpha=1.0 \
    actor_rollout_ref.actor.hvr_beta=0.1 \
    actor_rollout_ref.actor.hvr_lambda=0.5 \
    actor_rollout_ref.actor.hvr_cliprange=0.2

echo "🎉 HVR纯净测试完成！"
echo "📊 [HVR结果] 请查看WandB中的HVR专用指标："
echo "  - hvr/ervf_value_mean: ERVF价值函数均值"
echo "  - hvr/entropy_mean: 策略熵均值"
echo "  - hvr/hvr_reward_mean: HVR奖励均值"
echo "  - hvr/r_final_distribution: 稀疏奖励分布"
echo "  - hvr/success_rate: HVR处理成功率"
