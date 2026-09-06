#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account eu-25-10
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --partition qgpu_exp
#SBATCH --time 1:00:00
#SBATCH --output=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/eval/slurm-%j.out
#SBATCH --error=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/logs/eval/slurm-%j.err

DIR=/mnt/proj1/fta-25-74/dev/master_thesis

cd $DIR || exit

source .venv/bin/activate

# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/mocha/damaged_llama70b_mocha_mocha_v1.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/mocha/ --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa/damaged_cz_en_llama70b_qa_cz_en_cus_qa_gemini_1.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa/damaged_uk_en_llama70b_qa_uk_en_cus_qa_gemini_1.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa/damaged_sk_en_llama70b_qa_sk_en_cus_qa_gemini_1.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/rose/damaged_llama70b_rose_summarization_v2.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/rose/ --batch_size=128 --metrics=all


# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/mocha_train.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/mocha/ --batch_size=256 --metrics=all --limit=1500
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/qa_text-CZ_dev_en_binary.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=256 --metrics=all --limit=1500
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/qa_text-UA_dev_en_binary.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=256 --metrics=all --limit=1500
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/qa_text-SK_dev_en_binary.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=256 --metrics=all --limit=1500
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/rose_cnndm_validation.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/rose/ --batch_size=128 --metrics=all --limit=1500

# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/mocha/dataset_mocha_qwen3_30b_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/mocha/ --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/rose/dataset_rose_qwen3_30b_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/rose/ --batch_size=128 --metrics=all

# Existing datasets
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/wmt_2024 --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/metric_scores/wmt_2024 --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets/wmt_2021 --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/metric_scores/wmt_2021 --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/datasets --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/metric_scores --batch_size=256 --metrics=all

# Damaged datasets
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/cus_qa/ --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/rose/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/rose/ --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/mocha/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/mocha/ --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/wmt/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results/wmt/ --batch_size=256 --metrics=all

python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_fixed/wmt/dataset_wmt_21_en_ha_llama4scout_few_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results_fixed/wmt/ --batch_size=256 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets_fixed/cus_qa/dataset_qa_cz_en_llama4scout_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results_fixed/cus_qa/ --batch_size=256 --metrics=all



