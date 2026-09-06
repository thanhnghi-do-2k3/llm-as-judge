#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account open-36-38
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --output=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/eval/slurm-%j.out
#SBATCH --error=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/logs/eval/slurm-%j.err

DIR=/mnt/proj1/fta-25-74/dev/master_thesis

cd $DIR || exit

source .venv/bin/activate

export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/rose/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/rose --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_cus_qa_reg_nofmt_checkpoint-2286_few_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_cus_qa_reg_nofmt_checkpoint-2286_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_cus_qa_sum_nofmt_checkpoint-4572_few_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_cus_qa_sum_nofmt_checkpoint-4572_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_grpo_meta_judge_final_model_few_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig_grpo_meta_judge_final_model_zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
# python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/dataset_qa_cz_orig__zero_shot.jsonl --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all


python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/rose/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/rose --batch_size=128 --metrics=all
python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/mocha/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/mocha --batch_size=128 --metrics=all
python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa --batch_size=128 --metrics=all
python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/cus_qa_nonbin/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/cus_qa_nonbin --batch_size=128 --metrics=all
python src/metrics/eval_dataset.py --input_path=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/wmt/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/wmt --batch_size=128 --metrics=all
