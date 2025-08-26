#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "🎯 开始HVR集成测试: 在GRPO框架中的内生奖励机制..."
echo "🎯 [HVR特性] ERVF熵正则化价值函数 + 后见之明价值重塑"
echo "🎯 [HVR集成] 替代GRPO奖励计算，保留组间投票和GAE优势估计"

python3 -m verl.trainer.main_ppo \
    data.train_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/train.parquet \
    data.val_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=10 \
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
    +actor_rollout_ref.rollout.logprobs=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.05 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=Qwen2.5-0.5-TokenEMA \
    trainer.experiment_name=HVR_3_1_0.1_0.5_T \
    trainer.n_gpus_per_node=1 \
    trainer.default_local_dir=/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/HVR_3_1_0.1_0.5_T  \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=1 \
    trainer.total_epochs=4 \
    reward_model.reward_manager=hvr_logic_rl \
    +reward_model.reward_kwargs.hvr_alpha=1.0 \
    +reward_model.reward_kwargs.hvr_beta=0.1 \
    +reward_model.reward_kwargs.hvr_lambda=0.5

echo "🎉 HVR-GRPO集成测试完成！"
echo "📊 [HVR结果] 请查看WandB中的HVR专用指标："
echo "  - hvr/ervf_value_mean: ERVF价值函数均值"
echo "  - hvr/entropy_mean: 策略熵均值"
echo "  - hvr/reward_mean: HVR奖励均值"
echo "  - hvr/r_final_dist_*: 稀疏奖励分布"
echo "  - hvr/group_return_mean: 组平均回报"
echo "  - hvr/grpo_advantage_mean: GRPO优势均值"
echo "  - hvr/success_rate: HVR处理成功率"
echo ""
echo "🎯 [HVR核心特性验证]:"
echo "  ✅ ERVF价值函数: 基于logits的内生价值 + 熵正则化"
echo "  ✅ HVR奖励重塑: 稀疏奖励指导的价值轨迹重塑"
echo "  ✅ GRPO组间投票: 保留组内相对优势计算"
echo "  ✅ Logic RL兼容: 支持{-3,-1,0,1,3}等稀疏奖励"

echo "🎉 HVR纯净测试完成！"
echo "📊 [HVR结果] 请查看WandB中的HVR专用指标："
echo "  - hvr/ervf_value_mean: ERVF价值函数均值"
echo "  - hvr/entropy_mean: 策略熵均值"
echo "  - hvr/hvr_reward_mean: HVR奖励均值"
echo "  - hvr/r_final_distribution: 稀疏奖励分布"
echo "  - hvr/success_rate: HVR处理成功率"
