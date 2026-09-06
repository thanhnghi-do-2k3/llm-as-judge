#!/usr/bin/bash
#SBATCH --job-name test
#SBATCH --account open-36-38
#SBATCH --nodes 1
#SBATCH --partition qgpu
#SBATCH --gpus-per-node=2
#SBATCH --time 12:00:00
#SBATCH --array=1-5
#SBATCH --output=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/test/slurm-%A-%a.out
#SBATCH --error=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/test/slurm-%A-%a.err

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================

# --- A. Choose your Model (Uncomment one) ---
# MODEL="llama70b"
MODEL="qwen3_30b"
# MODEL="llama4scout"
# MODEL="qwen3_30b_thinking"

# --- B. General Settings ---
BASE_DIR="/mnt/proj1/fta-25-74/dev/master_thesis"
VENV_PATH=".venv/bin/activate"
DEFAULT_BATCH=128
DATA_COUNT=-1 # Use -1 for all, or 50 for testing

# --- C. Modes (Set true/false) ---
REASONING=false
SAMPLE=false
TESTING=true
TESTING_SET=$SLURM_ARRAY_TASK_ID

# ==========================================
# 2. SETUP (LOGIC HANDLER)
# ==========================================
cd $BASE_DIR || exit
source $VENV_PATH

# Base path for datasets
ROOT_OUT="${BASE_DIR}/src/karolina_testing"
EXTRA_FLAGS=""

# Logic to determine Folder Name and Flags
if [ "$TESTING" = true ]; then
    FOLDER_SUFFIX="damaged_datasets_testing_set${TESTING_SET}"
    EXTRA_FLAGS="--test_set ${TESTING_SET}"

elif [ "$REASONING" = true ] && [ "$SAMPLE" = true ]; then
    # CASE 1: BOTH
    FOLDER_SUFFIX="damaged_datasets_reasoning_sample"
    EXTRA_FLAGS="--reasoning --sample"
elif [ "$REASONING" = true ]; then
    # CASE 2: REASONING ONLY
    FOLDER_SUFFIX="damaged_datasets_reasoning"
    EXTRA_FLAGS="--reasoning"
elif [ "$SAMPLE" = true ]; then
    # CASE 3: SAMPLE ONLY
    FOLDER_SUFFIX="damaged_datasets_sample"
    EXTRA_FLAGS="--sample"
else
    # CASE 4: NEITHER (Standard)
    FOLDER_SUFFIX="damaged_datasets"
    EXTRA_FLAGS=""
fi

OUT_BASE="${ROOT_OUT}/${FOLDER_SUFFIX}"

echo "========================================"
echo "Model:          $MODEL"
echo "Mode:           Reasoning=$REASONING | Sample=$SAMPLE"
echo "Output Folder:  $OUT_BASE"
echo "Extra Flags:    $EXTRA_FLAGS"
echo "========================================"

# ==========================================
# 3. HELPER FUNCTION
# ==========================================
# Usage: run_task [dataset_config] [folder_name] [batch_size (optional)]
run_task() {
    local dataset=$1
    local folder=$2
    local batch=${3:-$DEFAULT_BATCH}

    # Construct the full output path
    # NOTE: We use the dynamic OUT_BASE determined above
    local output_file="${OUT_BASE}/${folder}/dataset.json"

    echo "Processing: $dataset -> $output_file"
    
    # We use $EXTRA_FLAGS unquoted so it expands into multiple arguments if needed
    python src/karolina_testing/generate.py \
        --model_config="$MODEL" \
        --dataset_config="$dataset" \
        --output_file="$output_file" \
        --data_count="$DATA_COUNT" \
        --batch_size="$batch" \
        --save_as_dataset \
        $EXTRA_FLAGS 
        # --prompt_type="zero_shot" \


    echo "Done."
}

# ==========================================
# 4. TASK EXECUTION
# ==========================================

# --- GROUP: MOCHA ---
# run_task "mocha_train"      "mocha" 128
# run_task "mocha_test"       "mocha" 128
# run_task "mocha_validation" "mocha" 128

# --- GROUP: ROSE (CNNDM/XSUM) ---
# run_task "rose_cnndm_validation" "rose" 128
# run_task "rose_cnndm_test"       "rose" 128
# run_task "rose_cnndm_protocol"   "rose" 128
# run_task "rose_xsum"             "rose" 128
# run_task "rose_cnndm_protocol_gpt3" "rose" 128

# --- GROUP: CUSTOM QA (CZ/SK/UK) ---
# run_task "qa_cz_en"    "cus_qa" 64
run_task "qa_cz_orig"  "cus_qa" 256
# run_task "qa_sk_en"    "cus_qa" 256
# run_task "qa_sk_orig"  "cus_qa" 256
# run_task "qa_uk_en"    "cus_qa" 256
# run_task "qa_uk_orig"  "cus_qa" 256

# --- GROUP: WMT ---
# run_task "wmt_24_cs_uk" "wmt" 256
# run_task "wmt_24_en_is" "wmt" 256
# run_task "wmt_24_en_cs" "wmt" 256
# run_task "wmt_21_xh_zu" "wmt" 256
# run_task "wmt_21_en_ha" "wmt" 256