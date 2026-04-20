#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT_PATH:?Set CKPT_PATH to the best stage-2 checkpoint.}"
: "${DATA_PATHS:?Set DATA_PATHS to ':'-separated GRPO-format training JSON files.}"
: "${TEST_DATA_PATHS:?Set TEST_DATA_PATHS to ':'-separated GRPO-format evaluation JSON files.}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SAVE_STEPS="${SAVE_STEPS:-600}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EXP_NAME="${EXP_NAME:-REFORM_policy_refinement_g${NUM_GENERATIONS}}"
SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/outputs/stage3_${EXP_NAME}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/scripts/grpo_json_REFORM_ROM_BERTreward.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed_zero_tech_off.json}"

export ZYC_DEBUG_MODE="${ZYC_DEBUG_MODE:-false}"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-REFORM-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOG_PATH="${SAVE_PATH}/debug_log.txt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mkdir -p "${SAVE_PATH}"
PARAM_LOG="${SAVE_PATH}/training_hyperparameters.log"
{
    echo "===== Training Hyperparameters ====="
    echo "DATA_PATHS=${DATA_PATHS}"
    echo "TEST_DATA_PATHS=${TEST_DATA_PATHS}"
    echo "CKPT_PATH=${CKPT_PATH}"
    echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
    echo "EXP_NAME=${EXP_NAME}"
    echo "SAVE_PATH=${SAVE_PATH}"
    echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
    echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "EVAL_STEPS=${EVAL_STEPS}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
    echo "-----------------------------------"
} > "${PARAM_LOG}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT:-12346}" \
    "${TRAIN_SCRIPT}" \
    --output_dir "${SAVE_PATH}" \
    --model_name_or_path "${CKPT_PATH}" \
    --data_file_paths "${DATA_PATHS}" \
    --test_data_file_paths "${TEST_DATA_PATHS}" \
    --image_folders None \
    --dataset_name None \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --logging_steps 5 \
    --bf16 \
    --gradient_checkpointing False \
    --attn_implementation flash_attention_2 \
    --max_pixels 590000 \
    --num_train_epochs "${NUM_EPOCHS}" \
    --run_name "${EXP_NAME}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --save_only_model true \
    --num_generations "${NUM_GENERATIONS}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_strategy steps \
    --load_best_model_at_end False \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --save_on_each_node False \
    --report_to wandb

