#!/usr/bin/bash
#SBATCH --account open-36-38
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --partition qgpu
#SBATCH --time 24:00:00

source ~/.bashrc

DIR=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning
cd $DIR || exit
source ../../.venv_finetune/bin/activate

echo "===== Generate Worker ====="
echo "Job Name:  $SLURM_JOB_NAME"
echo "Model:     $GEN_MODEL"
echo "Dataset:   $GEN_DATASET_CONFIG"
echo "Output:    $GEN_OUTPUT_FILE"
echo "Extra:     ${GEN_EXTRA_ARGS:-none}"
echo "==========================="

python ../karolina_testing/generate.py \
    --local_model_path="$GEN_MODEL" \
    --dataset_config="$GEN_DATASET_CONFIG" \
    --output_file="$GEN_OUTPUT_FILE" \
    --batch_size=128 \
    --data_count=-1 \
    --save_as_dataset \
    $GEN_EXTRA_ARGS
