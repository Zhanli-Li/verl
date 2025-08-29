#!/bin/bash
# Multi-turn Dialogue Training Script with Custom Reward Function
# This script demonstrates how to set up training for multi-turn dialogue with custom scoring

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Training parameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}
NUM_EPOCHS=${NUM_EPOCHS:-15}

# Model paths - update these to your actual model paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
SCORING_MODEL_PATH=${SCORING_MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}  # Path to your scoring model

# Data paths - update these to your dialogue dataset
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/multiturn_dialogue/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/multiturn_dialogue/test.parquet"}

# Custom reward function path
REWARD_FUNCTION_PATH="$PROJECT_DIR/verl/utils/reward_score/multiturn_dialogue.py"

echo "Starting multi-turn dialogue training with custom reward function..."
echo "Model: $MODEL_PATH"
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Reward function: $REWARD_FUNCTION_PATH"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_dialogue_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/multiturn_dialogue_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='multiturn_dialogue_rl' \
    trainer.experiment_name='multiturn_dialogue_custom_reward' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$NUM_EPOCHS \
    $@

echo "Training completed!"