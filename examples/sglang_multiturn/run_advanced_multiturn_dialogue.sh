#!/bin/bash
# Advanced Multi-turn Dialogue Training with Separate Scoring Model
# This script demonstrates using a dedicated model for scoring dialogue quality

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Training parameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
OFFLOAD=${OFFLOAD:-True}
NUM_EPOCHS=${NUM_EPOCHS:-20}

# Model paths
DIALOGUE_MODEL_PATH=${DIALOGUE_MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}  # Main dialogue model
SCORING_MODEL_URL=${SCORING_MODEL_URL:-"http://localhost:8001/v1/chat/completions"}  # Scoring model API
SCORING_MODEL_NAME=${SCORING_MODEL_NAME:-"dialogue-scorer"}

# Data paths
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/dialogue_dataset/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/dialogue_dataset/test.parquet"}

# Reward function for model-based scoring
REWARD_FUNCTION_PATH="$PROJECT_DIR/verl/utils/reward_score/multiturn_dialogue.py"

echo "=== Advanced Multi-turn Dialogue Training Setup ==="
echo "Dialogue Model: $DIALOGUE_MODEL_PATH"
echo "Scoring Model URL: $SCORING_MODEL_URL"
echo "Scoring Model Name: $SCORING_MODEL_NAME"
echo "Training Data: $TRAIN_DATA"
echo "Validation Data: $VAL_DATA"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "=================================================="

# Check if scoring model is accessible
echo "Testing scoring model connectivity..."
if curl -s --max-time 5 "$SCORING_MODEL_URL" > /dev/null; then
    echo "✓ Scoring model accessible at $SCORING_MODEL_URL"
    USE_MODEL_SCORING=true
else
    echo "⚠ Scoring model not accessible, falling back to rule-based scoring"
    USE_MODEL_SCORING=false
fi

# Prepare reward function parameters
REWARD_KWARGS=""
if [ "$USE_MODEL_SCORING" = true ]; then
    REWARD_KWARGS="custom_reward_function.reward_kwargs.scoring_model_url=$SCORING_MODEL_URL"
    REWARD_KWARGS="$REWARD_KWARGS custom_reward_function.reward_kwargs.scoring_model_name=$SCORING_MODEL_NAME"
    REWARD_FUNCTION_NAME="compute_model_based_dialogue_score"
    echo "Using model-based scoring with external API"
else
    REWARD_FUNCTION_NAME="compute_score"
    echo "Using rule-based scoring"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_dialogue_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_FUNCTION_NAME" \
    $REWARD_KWARGS \
    actor_rollout_ref.model.path="$DIALOGUE_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/multiturn_dialogue_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    trainer.critic_warmup=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='advanced_multiturn_dialogue' \
    trainer.experiment_name="dialogue_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.log_val_generations=5 \
    trainer.val_before_train=True \
    trainer.total_epochs=$NUM_EPOCHS \
    $@

echo "=== Training Completed Successfully ==="