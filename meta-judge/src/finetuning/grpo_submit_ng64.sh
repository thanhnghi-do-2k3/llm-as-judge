#!/usr/bin/bash
# Submit GRPO training for qa_sk_en [nonbin] with num_generations=64
# to test if higher num_generations improves training stability.
# Usage: bash grpo_submit_ng64.sh [--dry-run]

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

MODEL="/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/"
DS_ARGS="--dataset=cus_qa_nonbin --cus_qa_lang=sk_en"
BATCH_ARGS="--batch_size=64 --grad_acc=1 --num_generations=64 --run_name=ng64"
LOG_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/finetune
WORKER=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/grpo_worker.sh
mkdir -p "$LOG_DIR"

REWARD_TYPES=(
    "sum|"
    "reg|--reward_mode=reg"
    "pca|--reward_mode=pca"
)

COUNT=0
for rw_entry in "${REWARD_TYPES[@]}"; do
    RW_NAME="${rw_entry%%|*}"
    RW_ARGS="${rw_entry#*|}"
    JOB_NAME="grpo_gemma3-1b_qanb_sk-en_${RW_NAME}_ng64"

    if $DRY_RUN; then
        echo "[dry-run] $JOB_NAME"
    else
        sbatch \
            --job-name="$JOB_NAME" \
            --output="${LOG_DIR}/%j_${JOB_NAME}.out" \
            --error="${LOG_DIR}/%j_${JOB_NAME}.err" \
            --export=ALL,GRPO_MODEL="$MODEL",GRPO_DATASET_ARGS="$DS_ARGS",GRPO_REWARD_ARGS="$RW_ARGS",GRPO_BATCH_ARGS="$BATCH_ARGS" \
            "$WORKER"
    fi
    COUNT=$((COUNT + 1))
done

echo "Submitted $COUNT jobs."
