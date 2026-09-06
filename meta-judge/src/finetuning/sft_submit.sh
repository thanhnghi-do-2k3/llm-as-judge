#!/usr/bin/bash
# Submit SFT distillation training jobs as individual sbatch jobs.
# Combinations: models × datasets × teacher_models × prompt_types
# Usage: bash sft_submit.sh [--dry-run] [--batch_size=N] [--grad_accum=N] [--epochs=N] [--prompt_type=few_shot|zero_shot]

DRY_RUN=false
BATCH_SIZE="16"
GRAD_ACCUM="4"
EPOCHS=""
PROMPT_TYPE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --batch_size=*)   BATCH_SIZE="${arg#*=}" ;;
        --grad_accum=*)   GRAD_ACCUM="${arg#*=}" ;;
        --epochs=*)       EPOCHS="${arg#*=}" ;;
        --prompt_type=*)  PROMPT_TYPE="${arg#*=}" ;;
    esac
done

LOG_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/sft
WORKER=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/sft_worker.sh
mkdir -p "$LOG_DIR"

# --- Student models ---
MODELS=(
    # "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c"
    # "unsloth/Llama-3.2-3B-Instruct"
    "/mnt/proj1/fta-25-74/.cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/"
)
MODEL_NAMES=(
    # "llama1b"
    # "llama3b"
    "gemma3-1b"
)

# --- Dataset configurations ---
# Each entry: "short_name|dataset_flag|split_flag|exclude_splits_flag"
#
# Non-eval datasets with generated reasoning answers:
#   ROSE:   rose_cnndm_test, rose_cnndm_protocol, rose_cnndm_protocol_gpt3, rose_xsum
#           (exclude rose_cnndm_validation = eval split)
#   MOCHA:  mocha_train, mocha_test
#           (exclude mocha_validation = eval split)
#   CUS-QA: all 6 language splits (leave-one-out eval handled via exclude)
#   WMT:    wmt_24_* and wmt_21_* are ALL eval sets — no non-eval reasoning data
DATASETS=(
    # --- ROSE: all non-eval splits combined ---
    "rose|--dataset=rose||--exclude_splits=rose_cnndm_validation"
    # --- MOCHA: train+test combined (exclude validation = eval) ---
    "mocha|--dataset=mocha||--exclude_splits=mocha_validation"
    # --- CUS-QA: per-language jobs (mirror GRPO leave-one-out: exclude the eval language) ---
    "qa_cz-en|--dataset=cus_qa||--exclude_splits=qa_cz_en"
    "qa_cz-orig|--dataset=cus_qa||--exclude_splits=qa_cz_orig"
    "qa_sk-en|--dataset=cus_qa||--exclude_splits=qa_sk_en"
    "qa_sk-orig|--dataset=cus_qa||--exclude_splits=qa_sk_orig"
    "qa_uk-en|--dataset=cus_qa||--exclude_splits=qa_uk_en"
    "qa_uk-orig|--dataset=cus_qa||--exclude_splits=qa_uk_orig"
)

# --- Teacher models ---
TEACHER_MODELS=(
    "llama70b"
    "llama4scout"
    "qwen3_30b"
)

# --- Prompt types ---
PROMPT_TYPES=(
    "few_shot"
    "zero_shot"
)
# CLI --prompt_type overrides the list above
if [[ -n "$PROMPT_TYPE" ]]; then
    PROMPT_TYPES=("$PROMPT_TYPE")
fi

COUNT=0

for m_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$m_idx]}"
    MODEL_NAME="${MODEL_NAMES[$m_idx]}"

    for ds_entry in "${DATASETS[@]}"; do
        IFS='|' read -r DS_NAME DS_ARGS SPLIT_ARGS EXCLUDE_ARGS <<< "$ds_entry"

        for TEACHER in "${TEACHER_MODELS[@]}"; do
            for PT in "${PROMPT_TYPES[@]}"; do

                JOB_NAME="sft_${MODEL_NAME}_${DS_NAME}_${TEACHER}_${PT}"

                # Build extra args
                EXTRA_ARGS="$DS_ARGS --teacher_model=$TEACHER --prompt_type=$PT --run_name=$DS_NAME"
                [[ -n "$SPLIT_ARGS" ]]   && EXTRA_ARGS+=" $SPLIT_ARGS"
                [[ -n "$EXCLUDE_ARGS" ]] && EXTRA_ARGS+=" $EXCLUDE_ARGS"
                [[ -n "$BATCH_SIZE" ]]   && EXTRA_ARGS+=" --batch_size=$BATCH_SIZE"
                [[ -n "$GRAD_ACCUM" ]]   && EXTRA_ARGS+=" --grad_accum=$GRAD_ACCUM"
                [[ -n "$EPOCHS" ]]       && EXTRA_ARGS+=" --epochs=$EPOCHS"

                if $DRY_RUN; then
                    echo "[dry-run] $JOB_NAME"
                    echo "          model:   $MODEL"
                    echo "          teacher: $TEACHER"
                    echo "          prompt:  $PT"
                    echo "          args:    $EXTRA_ARGS"
                else
                    sbatch \
                        --job-name="$JOB_NAME" \
                        --output="${LOG_DIR}/%j_${JOB_NAME}.out" \
                        --error="${LOG_DIR}/%j_${JOB_NAME}.err" \
                        --export=ALL,SFT_MODEL="$MODEL",SFT_EXTRA_ARGS="$EXTRA_ARGS" \
                        "$WORKER"
                fi

                COUNT=$((COUNT + 1))
            done
        done
    done
done

echo "Submitted $COUNT jobs."
