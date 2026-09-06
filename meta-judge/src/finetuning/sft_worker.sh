#!/usr/bin/bash
#SBATCH --account open-36-38
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --partition qgpu
#SBATCH --time 48:00:00

source ~/.bashrc

DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning
cd $DIR || exit
source ../../.venv_finetune/bin/activate

echo "===== SFT Worker ====="
echo "Job Name: $SLURM_JOB_NAME"
echo "Model:    $SFT_MODEL"
echo "Args:     ${SFT_EXTRA_ARGS:-none}"
echo "======================"

python gemini_sft.py \
    --model="$SFT_MODEL" \
    $SFT_EXTRA_ARGS
