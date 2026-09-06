============================================================
SUMMARY
============================================================
Dataset      | Model                                         | Max BS
-------------+-----------------------------------------------+-------
rose         | unsloth/Llama-3.2-1B-Instruct                 |     64
cus_qa       | unsloth/Llama-3.2-1B-Instruct                 |    128
mocha        | unsloth/Llama-3.2-1B-Instruct                 |    128
wmt_en-is    | unsloth/Llama-3.2-1B-Instruct                 |    128
wmt_cs-uk    | unsloth/Llama-3.2-1B-Instruct                 |    128
wmt_en-cs    | unsloth/Llama-3.2-1B-Instruct                 |    128
rose         | unsloth/Llama-3.2-3B-Instruct                 |     24
cus_qa       | unsloth/Llama-3.2-3B-Instruct                 |    128
mocha        | unsloth/Llama-3.2-3B-Instruct                 |     96
wmt_en-is    | unsloth/Llama-3.2-3B-Instruct                 |     64
wmt_cs-uk    | unsloth/Llama-3.2-3B-Instruct                 |     64
wmt_en-cs    | unsloth/Llama-3.2-3B-Instruct                 |     64



transformers==5.3
trl==0.29.1
unsloth==2026.3.6


# Corr:
```python
python ../metrics/correlation.py --datasets_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_outputs/ --scores_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_eval/ --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_corr/
```
# Meta-corr:
```python
python ../metrics/meta_correlation.py /mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/model_corr/ /mnt/proj1/fta-25-74/dev/master_thesis/correlations
```
# Plots
```python
python plot_training_curves.py --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots/kl_div --run s2000 --no-components --smooth=3  --metrics kl --exclude nonbin
python plot_training_curves.py --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots/reward_cv --run s2000 --no-components --smooth=3  --metrics reward_cv --exclude nonbin
python plot_training_curves.py --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots/frac_reward_zero_std --run s2000 --no-components --smooth=3  --metrics frac_reward_zero_std --exclude nonbin
python plot_training_curves.py --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots/completion_length --run s2000 --no-components --smooth=3  --metrics completion_length --exclude nonbin
python plot_training_curves.py --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots/clipped_ratio --run s2000 --no-components --smooth=3  --metrics clipped_ratio --exclude nonbin
```

```python
python plot_training_curves.py   --models_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_trained   --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/finetuning/models_training_plots   --exclude nonbin   --run s2000   --exclude-run parall split   --no-components   --smooth=3   --split-metrics   --metrics reward_cv kl frac_reward_zero_std completion_length clipped_ratio
```