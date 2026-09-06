#!/usr/bin/bash
# Submit generation jobs for every saved checkpoint of a single trained model.
# Output filenames embed the checkpoint name so they never overwrite each other.
#
# Usage:
#   bash generate_checkpoints_submit.sh \
#       --model_dir=<path>          # dir containing checkpoint-* subdirs  (required)
#       --dataset_config=<name>     # passed as --dataset_config to generate.py  (required)
#       --output_subdir=<name>      # subdir under OUTPUT_BASE, e.g. cus_qa  (required)
#       --model_label=<name>        # short label for filenames/job names  (required)
#       [--prompt_type=zero_shot|few_shot]
#       [--include_final]           # also run the *_final_model if present
#       [--only_final]              # skip checkpoints, run only final model
#       [--dry-run]

DRY_RUN=false
PROMPT_TYPE=""
INCLUDE_FINAL=false
ONLY_FINAL=false
MODEL_DIR=""
DS_CONFIG=""
DS_SUBDIR=""
MODEL_LABEL=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)             DRY_RUN=true ;;
        --include_final)       INCLUDE_FINAL=true ;;
        --only_final)          ONLY_FINAL=true ;;
        --prompt_type=*)       PROMPT_TYPE="${arg#*=}" ;;
        --model_dir=*)         MODEL_DIR="${arg#*=}" ;;
        --dataset_config=*)    DS_CONFIG="${arg#*=}" ;;
        --output_subdir=*)     DS_SUBDIR="${arg#*=}" ;;
        --model_label=*)       MODEL_LABEL="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1 ;;
    esac
done

# --- Validate required args ---
missing=()
[[ -z "$MODEL_DIR" ]]   && missing+=(--model_dir)
[[ -z "$DS_CONFIG" ]]   && missing+=(--dataset_config)
[[ -z "$DS_SUBDIR" ]]   && missing+=(--output_subdir)
[[ -z "$MODEL_LABEL" ]] && missing+=(--model_label)
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing required arguments: ${missing[*]}" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model_dir not found: $MODEL_DIR" >&2
    exit 1
fi

LOG_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/eval_checkpoints
OUTPUT_BASE=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/checkpoints
WORKER=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/generate_worker.sh
mkdir -p "$LOG_DIR" "$OUTPUT_BASE/$DS_SUBDIR"

# --- Prompt types ---
PROMPT_TYPES=("zero_shot" "few_shot")
if [[ -n "$PROMPT_TYPE" ]]; then
    PROMPT_TYPES=("$PROMPT_TYPE")
fi

# --- Collect checkpoint paths ---
declare -a CKPT_PATHS=()
declare -a CKPT_LABELS=()

if ! $ONLY_FINAL; then
    while IFS= read -r -d '' ckpt_dir; do
        CKPT_PATHS+=("$ckpt_dir")
        CKPT_LABELS+=("$(basename "$ckpt_dir")")
    done < <(find "$MODEL_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print0 \
             | sort -z -t- -k2 -n)
fi

if $INCLUDE_FINAL || $ONLY_FINAL; then
    for candidate in "$MODEL_DIR"/*_final_model; do
        if [[ -d "$candidate" ]]; then
            CKPT_PATHS+=("$candidate")
            CKPT_LABELS+=("final_model")
            break
        fi
    done
fi

if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
    echo "No checkpoints found in: $MODEL_DIR" >&2
    exit 1
fi

echo "Found ${#CKPT_PATHS[@]} checkpoint(s) in $MODEL_DIR"

# --- Submit ---
COUNT=0

for i in "${!CKPT_PATHS[@]}"; do
    CKPT_PATH="${CKPT_PATHS[$i]}"
    CKPT_LABEL="${CKPT_LABELS[$i]}"

    for PT in "${PROMPT_TYPES[@]}"; do
        JOB_NAME="gen_chk_${MODEL_LABEL}_${CKPT_LABEL}_${PT}"
        OUTPUT_FILE="$OUTPUT_BASE/$DS_SUBDIR/${MODEL_LABEL}_${DS_CONFIG}_${CKPT_LABEL}_${PT}.jsonl"
        EXTRA_ARGS="--prompt_type=$PT"

        if $DRY_RUN; then
            echo "[dry-run] $JOB_NAME"
            echo "          model:   $CKPT_PATH"
            echo "          dataset: $DS_CONFIG"
            echo "          output:  $OUTPUT_FILE"
        else
            sbatch \
                --job-name="$JOB_NAME" \
                --output="${LOG_DIR}/%j_${JOB_NAME}.out" \
                --error="${LOG_DIR}/%j_${JOB_NAME}.err" \
                --export=ALL,GEN_MODEL="$CKPT_PATH",GEN_DATASET_CONFIG="$DS_CONFIG",GEN_OUTPUT_FILE="$OUTPUT_FILE",GEN_EXTRA_ARGS="$EXTRA_ARGS" \
                "$WORKER"
        fi

        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT jobs."
