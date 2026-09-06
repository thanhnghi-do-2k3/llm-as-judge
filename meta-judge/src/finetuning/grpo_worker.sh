#!/usr/bin/bash
#SBATCH --account open-36-38
#SBATCH --nodes 1
#SBATCH --gpus-per-node 2
#SBATCH --partition qgpu
#SBATCH --time 48:00:00

source ~/.bashrc

DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning
cd $DIR || exit
source ../../.venv_finetune/bin/activate

echo "===== GRPO Worker ====="
echo "Job Name: $SLURM_JOB_NAME"
echo "Model:    $GRPO_MODEL"
echo "Dataset:  $GRPO_DATASET_ARGS"
echo "Reward:   ${GRPO_REWARD_ARGS:-sum (no flags)}"
echo "Batch:    ${GRPO_BATCH_ARGS:-defaults}"
echo "========================"

python gemini_llama_grpo.py \
    --model="$GRPO_MODEL" \
    $GRPO_DATASET_ARGS \
    $GRPO_REWARD_ARGS \
    $GRPO_BATCH_ARGS
    
