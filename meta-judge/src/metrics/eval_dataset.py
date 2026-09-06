import json
import argparse
import time
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Assuming these imports exist in your environment
from human_feedback_datasets.processed_dataset import ProcessedDataset
from metrics.hf_metrics import METRIC_FNS, all_kwargs, kwargs_str

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

def eval_dataset(
    dataset: ProcessedDataset, 
    metrics: List[Tuple[str, Dict[str, Any]]], 
    batch_size: Optional[int] = 32, 
    metric_times_path: Optional[Path] = None, 
    metric_results_path: Optional[Path] = None
) -> Dict[str, Any]:
    
    # Validation
    for m, _ in metrics:
        assert m in METRIC_FNS, f"Metric {m} not supported. Supported metrics: {list(METRIC_FNS.keys())}"
    
    metric_times = {m: 0.0 for m, _ in metrics}

    # Helper to save results periodically
    def save_results() -> None:
        if metric_results_path is not None:
            with open(metric_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        
        if metric_times_path is not None:
            with open(metric_times_path, 'w', encoding='utf-8') as f:
                for m_key, t in metric_times.items():
                    if t > 0:
                        f.write(f"Metric {m_key} took {t:.2f} seconds\n")

    # Resume logic
    if metric_results_path is not None and metric_results_path.exists():
        try:
            with open(metric_results_path, 'r', encoding='utf-8') as f:
                results: Dict[str, List[Any]] = json.load(f)
            metrics_done: List[str] = list(results.keys())
            print(f"Resuming {metric_results_path.name} with {len(metrics_done)} metrics already completed.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {metric_results_path}. Starting from scratch.")
            results = {}
            metrics_done = []
    else:
        results = {}
        metrics_done = []

    # Main Metric Loop
    for i, (metric_name, metric_kwargs) in enumerate(metrics):
        metric_key = f"{metric_name}_{kwargs_str(metric_kwargs)}"
        
        if metric_key in metrics_done:
            print(f"Skipping {metric_name} (already done)")
            continue
            
        print(f"Evaluating metric {i+1}/{len(metrics)}: {metric_name} | Args: {metric_kwargs}")
        start = time.time()

        metric_fn = METRIC_FNS[metric_name]
        results[metric_key] = []
        
        # Batch Processing Setup
        total_len = len(dataset)
        eff_batch_size = batch_size or total_len
        num_batches = (total_len + eff_batch_size - 1) // eff_batch_size
        
        for b_idx in tqdm(range(0, total_len, eff_batch_size), desc=f"  Running {metric_name}", total=num_batches):
            batch = dataset[b_idx : b_idx + eff_batch_size]
            refs = [ex.reference for ex in batch]
            preds = [ex.prediction for ex in batch]
            srcs = [ex.input for ex in batch]

            if metric_name == "comet":
                scores = metric_fn(refs, preds, src=srcs, **metric_kwargs)
            else:
                scores = metric_fn(refs, preds, **metric_kwargs)

            results[metric_key].extend(scores)

        metric_times[metric_key] = time.time() - start
        save_results()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate datasets using HuggingFace metrics.")
    
    # Input/Output
    parser.add_argument("--input_path", type=Path, required=True, help="Path to a single .jsonl file or a folder containing .jsonl files.")
    parser.add_argument("--output_dir", type=Path, default=Path("metric_scores"), help="Directory to save the output .json files.")
    
    # Configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric evaluation.")
    parser.add_argument("--metrics", nargs="+", default=["all"], help="List of metrics to run (e.g. 'bleu' 'rouge'). Default: 'all'")
    parser.add_argument("--skip_files", nargs="+", default=[], help="List of filenames (without extension) to skip.")
    
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples from each dataset. Useful for testing.")
    parser.add_argument("--save_time", action="store_true", help="Save evaluation times for each metric.")

    args = parser.parse_args()

    # 1. Resolve Input Files
    files_to_process = []
    if args.input_path.is_file():
        files_to_process = [args.input_path]
    elif args.input_path.is_dir():
        files_to_process = list(args.input_path.glob("*.jsonl"))
    else:
        print(f"Error: Input path {args.input_path} does not exist.")
        sys.exit(1)

    if not files_to_process:
        print(f"No .jsonl files found in {args.input_path}")
        sys.exit(0)

    # 2. Prepare Output Directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolve Metrics using all_kwargs(only=[...])
    # If user passed "all", we pass empty list [] to all_kwargs to get everything.
    # Otherwise, we pass the list from args.metrics.
    filter_list = [] if "all" in args.metrics else args.metrics
    target_metrics = all_kwargs(only=filter_list)

    if not target_metrics:
        print(f"Warning: No metrics found. Check if {args.metrics} are valid metric names.")
        sys.exit(0)

    print(f"Found {len(files_to_process)} datasets to evaluate.")
    print(f"Selected metrics: {set(m[0] for m in target_metrics)}\n")

    # 4. Processing Loop
    for dataset_path in files_to_process:
        if dataset_path.stem in args.skip_files:
            print(f"Skipping {dataset_path.name} (requested by user).")
            continue

        print(f"--- Processing: {dataset_path.name} ---")
        
        # Define output paths
        res_path = args.output_dir / f"{dataset_path.stem}.json"
        time_path = args.output_dir / f"{dataset_path.stem}_times.txt" if args.save_time else None

        try:
            dataset = ProcessedDataset.load(dataset_path)

            if args.limit is not None:
                original_len = len(dataset)
                dataset = dataset[:args.limit]
                print(f"  Dataset truncated from {original_len} to {len(dataset)} samples.")

        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")
            continue

        res = eval_dataset(
            dataset, 
            metrics=target_metrics, 
            batch_size=args.batch_size,
            metric_times_path=time_path,
            metric_results_path=res_path
        )
        
        # Final explicit dump
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
            
    print("\nAll evaluations complete.")