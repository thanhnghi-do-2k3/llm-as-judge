#!/usr/bin/bash
#SBATCH --job-name bench_bs
#SBATCH --account fta-25-74
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --partition qgpu
#SBATCH --time 00:30:00
#SBATCH --output=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/slurm-%j.out
#SBATCH --error=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/slurm-%j.err


DIR=/mnt/proj1/fta-25-74/dev/master_thesis
cd $DIR || exit
source .venv/bin/activate

# Testing configurations
# We limit data_count to 40 to make it quick
DATASET="qa_cz_orig"
OUTPUT_DIR="/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa"

echo "------------------------------------------------"
echo "STARTING BATCH SIZE BENCHMARK"
echo "------------------------------------------------"

for BS in 4 8 16 32 64; do
    echo ">>> Testing Batch Size: $BS"
    
    start_time=$(date +%s)
    
    python src/karolina_testing/generate.py \
      --model_config=llama70b \
      --dataset_config=$DATASET \
      --output_file="${OUTPUT_DIR}/bench_${BS}.json" \
      --data_count=64 \
      --batch_size=$BS \
      --save_as_dataset
      
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    
    if [ $? -eq 0 ]; then
        echo ">>> Batch Size $BS: SUCCESS in $elapsed seconds"
    else
        echo ">>> Batch Size $BS: FAILED (Likely OOM)"
        break # Stop testing if we crash
    fi
    echo "------------------------------------------------"
done