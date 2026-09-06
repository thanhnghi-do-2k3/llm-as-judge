#!/usr/bin/bash
# Submit generation jobs for specified model+dataset pairs as individual sbatch jobs.
# Usage: bash generate_submit.sh [--dry-run] [--prompt_type=zero_shot|few_shot]

DRY_RUN=false
PROMPT_TYPE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --prompt_type=*)  PROMPT_TYPE="${arg#*=}" ;;
    esac
done

LOG_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/eval
OUTPUT_BASE=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs
MODELS_DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained
HF_CACHE=/mnt/proj1/fta-25-74/.cache/huggingface/hub
WORKER=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/generate_worker.sh
mkdir -p "$LOG_DIR"

# --- Jobs ---
# Each entry: "short_name|model_path|dataset_config|output_subdir"
JOBS=(
    # Fine-tuned models
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_cz_en/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_cz_en|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_cz_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_cz_en|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_cz_orig/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_cz_orig|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_cz_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_cz_orig|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_nonbin_cz_en/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_cz_en|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_cz_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_cz_en|cus_qa_nonbin"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_nonbin_cz_orig/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_cz_orig|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_cz_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_cz_orig|cus_qa_nonbin"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_nonbin_sk_en/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_sk_en|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_sk_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_sk_en|cus_qa_nonbin"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_nonbin_sk_orig/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_sk_orig|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_sk_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_sk_orig|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_ua_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_uk_en|cus_qa_nonbin"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_nonbin_ua_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_uk_orig|cus_qa_nonbin"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_sk_en/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_sk_en|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_sk_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_sk_en|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_sk_orig/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_sk_orig|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_sk_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_sk_orig|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_ua_en/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_uk_en|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_ua_en/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_uk_en|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/cus_qa_ua_orig/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|qa_uk_orig|cus_qa"
    # "gemma-3-1b-it_sum|$MODELS_DIR/cus_qa_ua_orig/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|qa_uk_orig|cus_qa"
    # "gemma-3-1b-it_reg|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_sum|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_sum_e3_lp0.5_kl0.05/gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_reg_e3_lp0.5_kl0.05/gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_pca_e3_lp0.5_kl0.05/gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # s2000 (no parall) — wmt_cs-uk
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|wmt_24_cs_uk|wmt"
    # s2000 split — wmt_cs-uk
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_cs-uk/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|wmt_24_cs_uk|wmt"
    # "gemma-3-1b-it_reg|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_sum|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_sum_e3_lp0.5_kl0.05/gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_reg_e3_lp0.5_kl0.05/gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_pca_e3_lp0.5_kl0.05/gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # s2000 (no parall) — wmt_en-cs
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|wmt_24_en_cs|wmt"
    # s2000 split — wmt_en-cs
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_cs|wmt"
    "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-cs/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_cs|wmt"
    # "gemma-3-1b-it_reg|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_reg/gemma-3-1b-it_reg_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_sum|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_sum/gemma-3-1b-it_sum_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_sum_e3_lp0.5_kl0.05/gemma-3-1b-it_sum_e3_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_reg_e3_lp0.5_kl0.05/gemma-3-1b-it_reg_e3_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_pca_e3_lp0.5_kl0.05/gemma-3-1b-it_pca_e3_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # s2000 (no parall) — wmt_en-is
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_final_model|wmt_24_en_is|wmt"
    # s2000 split — wmt_en-is
    # "gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_pca_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_is|wmt"
    "gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_reg_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|$MODELS_DIR/wmt_en-is/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split/gemma-3-1b-it_sum_s2000_lp0.5_kl0.05_split_final_model|wmt_24_en_is|wmt"
    # "gemma-3-1b-it_sum|$MODELS_DIR/mocha/gemma-3-1b-it_sum/checkpoint-5750|mocha_validation|mocha"
    # "gemma-3-1b-it_reg|$MODELS_DIR/mocha/gemma-3-1b-it_reg/checkpoint-5500|mocha_validation|mocha"
    # Base model
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_cz_en|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_cz_orig|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_cz_en|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_cz_orig|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_sk_en|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_sk_orig|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_uk_en|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_uk_orig|cus_qa_nonbin"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_sk_en|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_sk_orig|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_uk_en|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|qa_uk_orig|cus_qa"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|wmt_24_en_is|wmt"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|wmt_24_cs_uk|wmt"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|wmt_24_en_cs|wmt"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|wmt_21_en_ha|wmt"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|wmt_21_xh_zu|wmt"
    # "gemma3-1b-base|$HF_CACHE/models--unsloth--gemma-3-1b-it/snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c/|mocha_validation|mocha"

    # SFT distill no_reasoning models — mocha
    # "llama4scout_few_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_llama4scout_few_shot_no_reasoning_mocha/_sft_distill_llama4scout_few_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"
    # "llama4scout_zero_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_llama4scout_zero_shot_no_reasoning_mocha/_sft_distill_llama4scout_zero_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"
    # "llama70b_few_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_llama70b_few_shot_no_reasoning_mocha/_sft_distill_llama70b_few_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"
    # "llama70b_zero_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_llama70b_zero_shot_no_reasoning_mocha/_sft_distill_llama70b_zero_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"
    # "qwen3_30b_few_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_qwen3_30b_few_shot_no_reasoning_mocha/_sft_distill_qwen3_30b_few_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"
    # "qwen3_30b_zero_shot_no_reasoning_mocha|$MODELS_DIR/mocha/_sft_distill_qwen3_30b_zero_shot_no_reasoning_mocha/_sft_distill_qwen3_30b_zero_shot_no_reasoning_mocha_final_model|mocha_validation|mocha"

    # SFT distill no_reasoning models — rose (eval split: rose_cnndm_validation)
    # "llama4scout_few_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_llama4scout_few_shot_no_reasoning_rose/_sft_distill_llama4scout_few_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"
    # "llama4scout_zero_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_llama4scout_zero_shot_no_reasoning_rose/_sft_distill_llama4scout_zero_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"
    # "llama70b_few_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_llama70b_few_shot_no_reasoning_rose/_sft_distill_llama70b_few_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"
    # "llama70b_zero_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_llama70b_zero_shot_no_reasoning_rose/_sft_distill_llama70b_zero_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"
    # "qwen3_30b_few_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_qwen3_30b_few_shot_no_reasoning_rose/_sft_distill_qwen3_30b_few_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"
    # "qwen3_30b_zero_shot_no_reasoning_rose|$MODELS_DIR/rose/_sft_distill_qwen3_30b_zero_shot_no_reasoning_rose/_sft_distill_qwen3_30b_zero_shot_no_reasoning_rose_final_model|rose_cnndm_validation|rose"

    # SFT distill no_reasoning models — cus_qa
    # "llama4scout_few_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_cz-en/_sft_distill_llama4scout_few_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_cz-en/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_cz-en/_sft_distill_llama70b_few_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_cz-en/_sft_distill_llama70b_zero_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_cz-en/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_cz-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_cz-en/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_cz-en_final_model|qa_cz_en|cus_qa"
    # "llama4scout_few_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_cz-orig/_sft_distill_llama4scout_few_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_cz-orig/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_cz-orig/_sft_distill_llama70b_few_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_cz-orig/_sft_distill_llama70b_zero_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_cz-orig/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_cz-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_cz-orig/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_cz-orig_final_model|qa_cz_orig|cus_qa"
    # "llama4scout_few_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_sk-en/_sft_distill_llama4scout_few_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_sk-en/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_sk-en/_sft_distill_llama70b_few_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_sk-en/_sft_distill_llama70b_zero_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_sk-en/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_sk-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_sk-en/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_sk-en_final_model|qa_sk_en|cus_qa"
    # "llama4scout_few_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_sk-orig/_sft_distill_llama4scout_few_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_sk-orig/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_sk-orig/_sft_distill_llama70b_few_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_sk-orig/_sft_distill_llama70b_zero_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_sk-orig/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_sk-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_sk-orig/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_sk-orig_final_model|qa_sk_orig|cus_qa"
    # "llama4scout_few_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_uk-en/_sft_distill_llama4scout_few_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_uk-en/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_uk-en/_sft_distill_llama70b_few_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_uk-en/_sft_distill_llama70b_zero_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_uk-en/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_uk-en|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_uk-en/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_uk-en_final_model|qa_uk_en|cus_qa"
    # "llama4scout_few_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_few_shot_no_reasoning_qa_uk-orig/_sft_distill_llama4scout_few_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
    # "llama4scout_zero_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_uk-orig/_sft_distill_llama4scout_zero_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
    # "llama70b_few_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_few_shot_no_reasoning_qa_uk-orig/_sft_distill_llama70b_few_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
    # "llama70b_zero_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_llama70b_zero_shot_no_reasoning_qa_uk-orig/_sft_distill_llama70b_zero_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
    # "qwen3_30b_few_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_uk-orig/_sft_distill_qwen3_30b_few_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
    # "qwen3_30b_zero_shot_no_reasoning_qa_uk-orig|$MODELS_DIR/cus_qa/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_uk-orig/_sft_distill_qwen3_30b_zero_shot_no_reasoning_qa_uk-orig_final_model|qa_uk_orig|cus_qa"
)

# --- Prompt types ---
PROMPT_TYPES=(
    "zero_shot"
    "few_shot"
)
# CLI --prompt_type overrides the list above
if [[ -n "$PROMPT_TYPE" ]]; then
    PROMPT_TYPES=("$PROMPT_TYPE")
fi

COUNT=0

for job_entry in "${JOBS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH DS_CONFIG DS_SUBDIR <<< "$job_entry"

    for PT in "${PROMPT_TYPES[@]}"; do
        JOB_NAME="gen_${MODEL_NAME}_${PT}"
        OUTPUT_FILE="$OUTPUT_BASE/$DS_SUBDIR/${MODEL_NAME}_${DS_CONFIG}_${PT}.jsonl"
        EXTRA_ARGS="--prompt_type=$PT"

        if $DRY_RUN; then
            echo "[dry-run] $JOB_NAME"
            echo "          model:   $MODEL_PATH"
            echo "          dataset: $DS_CONFIG"
            echo "          output:  $OUTPUT_FILE"
        else
            sbatch \
                --job-name="$JOB_NAME" \
                --output="${LOG_DIR}/%j_${JOB_NAME}.out" \
                --error="${LOG_DIR}/%j_${JOB_NAME}.err" \
                --export=ALL,GEN_MODEL="$MODEL_PATH",GEN_DATASET_CONFIG="$DS_CONFIG",GEN_OUTPUT_FILE="$OUTPUT_FILE",GEN_EXTRA_ARGS="$EXTRA_ARGS" \
                "$WORKER"
        fi

        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT jobs."
