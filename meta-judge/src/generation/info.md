Number of unique references:
qa_text-UA_dev_en_binary.jsonl: 385
qa_text-CZ_dev_en_binary.jsonl: 530
rose_xsum.jsonl: 500
qa_text-CZ_dev_orig_binary.jsonl: 530
qa_text-SK_dev_en_binary.jsonl: 493
mocha_validation.jsonl: 1110
rose_cnndm_protocol_gpt3.jsonl: 100
mocha_test.jsonl: 1736
qa_text-SK_dev_orig_binary.jsonl: 493
rose_cnndm_validation.jsonl: 998
rose_cnndm_protocol.jsonl: 100
qa_text-SK_dev_orig.jsonl: 493
wmt_older.jsonl: 3417
qa_text-UA_dev_en.jsonl: 385
rose_cnndm_test.jsonl: 500
qa_text-CZ_dev_en.jsonl: 530
qa_text-CZ_dev_orig.jsonl: 530
qa_text-UA_dev_orig_binary.jsonl: 385
mocha_train.jsonl: 7718
qa_text-SK_dev_en.jsonl: 493
qa_text-UA_dev_orig.jsonl: 385
mocha_minimal_pairs.jsonl: 183


# Benchmark llama70b
--- Benchmarking Dataset: ROSE ---
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
  > Processed 64/64 samples...
  > Speed: 15.851s/sample
  > Est. Time: 2 days, 4:43:45


# Meta-correlation

```python
# from src/karolina_testing
python ../metrics/meta_correlation.py /mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/corr_results /mnt/proj1/fta-25-74/dev/master_thesis/correlations
# python ../metrics/meta_correlation.py /mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/corr_results_backup /mnt/proj1/fta-25-74/dev/master_thesis/correlations_backup
```

# Correlation
```python
# from src/karolina_testing

# metric vs human feedback
python ../metrics/correlation.py --datasets_dir=/mnt/proj1/fta-25-74/dev/master_thesis/datasets --scores_dir=/mnt/proj1/fta-25-74/dev/master_thesis/metric_scores --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/correlations
# python ../metrics/correlation.py --datasets_dir=/mnt/proj1/fta-25-74/dev/master_thesis/datasets --scores_dir=/mnt/proj1/fta-25-74/dev/master_thesis/metric_scores_backup --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/correlations_backup

# metric vs pseudo-label
python ../metrics/correlation.py --datasets_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets --scores_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/eval_results --output_dir=/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/corr_results
```
