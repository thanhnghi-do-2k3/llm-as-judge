#!/usr/bin/bash
# Submit all 68 GRPO training configurations as individual sbatch jobs.
# 2 models × 17 datasets × 2 reward types = 68
# Usage: bash grpo_submit.sh [--dry-run] [--batch_size=N] [--grad_acc=N] [--num_generations=N] [--max_steps=N] [--save_steps=N]
#                            [--length_penalty_alpha=N] [--length_penalty_max_ratio=N] [--kl_beta=N] [--wmt_split]

DRY_RUN=false
BATCH_SIZE=""
GRAD_ACC=""
NUM_GEN=""
MAX_STEPS="2000" #"2000"
SAVE_STEPS="100"
LENGTH_PENALTY_ALPHA="0.5" #"0.5"
LENGTH_PENALTY_MAX_RATIO="2.0" #"2.0"
KL_BETA="0.05" #"0.05"
WMT_PARALLEL_EXAMPLES="0" #""
WMT_SPLIT=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=true ;;
        --batch_size=*)  BATCH_SIZE="${arg#*=}" ;;
        --grad_acc=*)    GRAD_ACC="${arg#*=}" ;;
        --num_generations=*) NUM_GEN="${arg#*=}" ;;
        --max_steps=*)   MAX_STEPS="${arg#*=}" ;;
        --save_steps=*)  SAVE_STEPS="${arg#*=}" ;;
        --length_penalty_alpha=*)     LENGTH_PENALTY_ALPHA="${arg#*=}" ;;
        --length_penalty_max_ratio=*) LENGTH_PENALTY_MAX_RATIO="${arg#*=}" ;;
        --kl_beta=*)     KL_BETA="${arg#*=}" ;;
        --wmt_parallel_examples=*) WMT_PARALLEL_EXAMPLES="${arg#*=}" ;;
        --wmt_split)     WMT_SPLIT=true ;;
    esac
done

# --- Per model+dataset batch size overrides ---
# Key: "model_idx,ds_short_name"  Value: batch args
# Only override where the default (32x2) is known to OOM.
declare -A BATCH_OVERRIDE
BATCH_OVERRIDE["0,rose"]="--batch_size=4 --grad_acc=16"
# BATCH_OVERRIDE["0,wmt_en-cs"]="--batch_size=16 --grad_acc=4"
# BATCH_OVERRIDE["0,wmt_cs-uk"]="--batch_size=16 --grad_acc=4"
# BATCH_OVERRIDE["0"]="--batch_size=16 --grad_acc=4"

# Helper: build batch args for a given (model_idx, ds_name) pair
# Lookup priority: BATCH_OVERRIDE["m,ds"] > BATCH_OVERRIDE["m"] > CLI flags > nothing
build_batch_args() {
    local m_idx="$1" ds_name="$2"
    local args="${BATCH_OVERRIDE[$m_idx,$ds_name]:-${BATCH_OVERRIDE[$m_idx]:-}}"
    # CLI flags override per-job defaults
    [[ -n "$BATCH_SIZE" ]] && args="--batch_size=$BATCH_SIZE"
    [[ -n "$GRAD_ACC" ]]   && args+=" --grad_acc=$GRAD_ACC"
    [[ -n "$NUM_GEN" ]]    && args+=" --num_generations=$NUM_GEN"
    [[ -n "$MAX_STEPS" ]]  && args+=" --max_steps=$MAX_STEPS"
    [[ -n "$SAVE_STEPS" ]] && args+=" --save_steps=$SAVE_STEPS"
    [[ -n "$LENGTH_PENALTY_ALPHA" ]]     && args+=" --length_penalty_alpha=$LENGTH_PENALTY_ALPHA"
    [[ -n "$LENGTH_PENALTY_MAX_RATIO" ]] && args+=" --length_penalty_max_ratio=$LENGTH_PENALTY_MAX_RATIO"
    [[ -n "$KL_BETA" ]]                  && args+=" --kl_beta=$KL_BETA"
    [[ -n "$WMT_PARALLEL_EXAMPLES" ]]   && args+=" --wmt_parallel_examples=$WMT_PARALLEL_EXAMPLES"
    $WMT_SPLIT                          && args+=" --wmt_split"
    echo "$args"
}

LOG_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/finetune
WORKER=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/grpo_worker.sh
mkdir -p "$LOG_DIR"

# --- Models ---
MODELS=(
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/",
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/"
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/"
    "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/"
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-Instruct/snapshots/a75c9dc945567a9b6f568b8503a0307731607bee"
)
MODEL_NAMES=(
    # "gemma3-4b" 
    # "qwen35-9b"
    "gemma3-1b"
    # "qwen25_7b"
    )
# MODEL_NAMES=("llama1b" "gemma12b")

# --- Dataset configurations ---
# Each entry: "short_name|dataset_flag [extra_args...]"
DATASETS=(
    # "mocha|--dataset=mocha"
    # "qa_cz-orig|--dataset=cus_qa --cus_qa_lang=cz_orig"
    # "qa_cz-en|--dataset=cus_qa --cus_qa_lang=cz_en"
    # "qa_sk-orig|--dataset=cus_qa --cus_qa_lang=sk_orig"
    # "qa_sk-en|--dataset=cus_qa --cus_qa_lang=sk_en"
    # "qa_ua-orig|--dataset=cus_qa --cus_qa_lang=ua_orig"
    # "qa_ua-en|--dataset=cus_qa --cus_qa_lang=ua_en"
    # "qanb_cz-orig|--dataset=cus_qa_nonbin --cus_qa_lang=cz_orig"
    # "qanb_cz-en|--dataset=cus_qa_nonbin --cus_qa_lang=cz_en"
    # "qanb_sk-orig|--dataset=cus_qa_nonbin --cus_qa_lang=sk_orig"
    # "qanb_sk-en|--dataset=cus_qa_nonbin --cus_qa_lang=sk_en"
    # "qanb_ua-orig|--dataset=cus_qa_nonbin --cus_qa_lang=ua_orig"
    # "qanb_ua-en|--dataset=cus_qa_nonbin --cus_qa_lang=ua_en"
    # "rose|--dataset=rose"
    "wmt_en-is|--dataset=wmt --wmt_pair=en-is"
    "wmt_cs-uk|--dataset=wmt --wmt_pair=cs-uk"
    "wmt_en-cs|--dataset=wmt --wmt_pair=en-cs"
)

# --- Reward types ---
# Each entry: "short_name|flags"
# For cus_qa, --reward_mode=reg automatically uses per-dataset regression
REWARD_TYPES=(
    "sum|"
    "reg|--reward_mode=reg"
    "pca|--reward_mode=pca"
)

COUNT=0

for m_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$m_idx]}"
    MODEL_NAME="${MODEL_NAMES[$m_idx]}"

    for ds_entry in "${DATASETS[@]}"; do
        DS_NAME="${ds_entry%%|*}"
        DS_ARGS="${ds_entry#*|}"

        for rw_entry in "${REWARD_TYPES[@]}"; do
            RW_NAME="${rw_entry%%|*}"
            RW_ARGS="${rw_entry#*|}"

            JOB_NAME="grpo_${MODEL_NAME}_${DS_NAME}_${RW_NAME}"
            JOB_BATCH_ARGS="$(build_batch_args "$m_idx" "$DS_NAME")"

            if $DRY_RUN; then
                echo "[dry-run] $JOB_NAME ${JOB_BATCH_ARGS:+(batch:$JOB_BATCH_ARGS)}"
            else
                sbatch \
                    --job-name="$JOB_NAME" \
                    --output="${LOG_DIR}/%j_${JOB_NAME}.out" \
                    --error="${LOG_DIR}/%j_${JOB_NAME}.err" \
                    --export=ALL,GRPO_MODEL="$MODEL",GRPO_DATASET_ARGS="$DS_ARGS",GRPO_REWARD_ARGS="$RW_ARGS",GRPO_BATCH_ARGS="$JOB_BATCH_ARGS" \
                    "$WORKER"
            fi

            COUNT=$((COUNT + 1))
        done
    done
done

echo "Submitted $COUNT jobs."
