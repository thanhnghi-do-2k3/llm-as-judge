import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ConstantInputWarning
from human_feedback_datasets.processed_dataset import ProcessedDataset
from numpy.typing import NDArray
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from pprint import pprint

_CORR = {
    "pearson": lambda x, y: pearsonr(x, y)[0],
    "spearman": lambda x, y: spearmanr(x, y)[0],
    "kendall": lambda x, y: kendalltau(x, y)[0],
}

def safe_corr(name: str, xs: NDArray[np.float32], ys: NDArray[np.float32]) -> float:
    try:
        return float(_CORR[name](xs, ys))
    except ConstantInputWarning:
        return float("nan")

def hf_metrics_corr(dataset: ProcessedDataset, metric_scores: Dict[str, List[float]], max_pred_words: int | None = None) -> Dict[str, Dict[str, float]]:
    dataset_hf = [ex.processed_human_score for ex in dataset]
    assert all(isinstance(s, (int, float)) for s in dataset_hf), f"All human scores must be int or float: {[s for s in dataset_hf if not isinstance(s, (int, float))][:5]}..."

    if max_pred_words is not None:
        mask = [len(ex.prediction.split()) <= max_pred_words for ex in dataset]
        kept = sum(mask)
        if kept < len(dataset):
            print(f"  [Filter] max_pred_words={max_pred_words}: keeping {kept}/{len(dataset)} examples")
        dataset_hf = [s for s, m in zip(dataset_hf, mask) if m]
        metric_scores = {k: [s for s, m in zip(v, mask) if m] for k, v in metric_scores.items()}

    corrs: Dict[str, Dict[str, float]] = {}
    for metric_name, scores in metric_scores.items():
        if len(dataset_hf) != len(scores):
            print(f"  [Warning] Skipping metric {metric_name}: Dataset size {len(dataset_hf)} != Scores size {len(scores)}")
            continue
        corrs[metric_name] = {k: safe_corr(k, np.array(dataset_hf), np.array(scores)) for k in _CORR.keys()}
    return corrs

def metric_metric_corr(metric_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    corrs: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    metric_types = set([name.split('_')[0] for name in metric_scores.keys()])
    for metric_type in metric_types:
        corrs[metric_type] = {}
        # Filter metrics belonging to this type
        type_metrics = {k: v for k, v in metric_scores.items() if k.startswith(metric_type + '_')}
        
        for metric_name1, scores1 in type_metrics.items():
            for metric_name2, scores2 in type_metrics.items():
                if metric_name1 == metric_name2: continue
                assert len(scores1) == len(scores2), f"Size mismatch between {metric_name1} and {metric_name2}"
                # print(f"{scores1[:5]} vs {scores2[:5]}")
                corrs[metric_type][f"{metric_name1}***{metric_name2}"] = {
                    k: safe_corr(k, np.array(scores1), np.array(scores2)) for k in _CORR.keys()
                }
        
        # Calculate mean correlation for this metric type if valid correlations exist
        if corrs[metric_type]:
            corrs[metric_type]['mean'] = {
                k: float(np.mean([v[k] for v in corrs[metric_type].values()])) 
                for k in _CORR.keys()
            }
    return corrs

def save_corrs(corrs: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corrs, f, indent=2)
    # print(f"Saved correlations to {path}") # Reduced verbosity

def is_old_wmt(path: Path) -> bool:
    old_years = ['2017', '2018', '2019', '2020', '2022']
    for year in old_years:
        if any(year in part for part in path.parts): return True
    return False

def compute_correlations(datasets_folder: Path, metric_scores_folder: Path, output_corrs_folder: Path, max_pred_words: int | None = None) -> None:
    """Iterates through datasets and computes correlations."""
    print(f"--- Starting Correlation Computation ---")
    print(f"Datasets: {datasets_folder}\nScores: {metric_scores_folder}\nOutput: {output_corrs_folder}")

    # Track failures: List of tuples (Dataset Name, Reason)
    skipped_datasets: List[Tuple[str, str]] = []

    for dataset_path in datasets_folder.rglob('*.jsonl'):
        if any('backup' in part for part in dataset_path.parts): continue
        if is_old_wmt(dataset_path): continue

        print(f"Processing dataset {dataset_path.stem}")
        
        # Preserve subfolder structure
        rel_path = dataset_path.relative_to(datasets_folder)
        metric_scores_path = metric_scores_folder / rel_path.with_suffix('.json')
        output_corrs_path = output_corrs_folder / rel_path.parent
        output_corrs_path.mkdir(parents=True, exist_ok=True)

        # Check 1: Metric Scores File Missing
        if not metric_scores_path.exists():
            print(f"  [Error] Metric scores not found: {metric_scores_path}")
            skipped_datasets.append((dataset_path.stem, "Metric scores file not found"))
            continue

        # Read once to avoid race conditions with concurrent eval jobs writing the file
        try:
            metric_scores = json.loads(metric_scores_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f"  [Error] Failed to parse metric scores (file may still be written): {e}")
            skipped_datasets.append((dataset_path.stem, f"Invalid JSON in metric scores: {e}"))
            continue

        dataset = ProcessedDataset.load(dataset_path)

        print(f'  Computing human feedback correlations - {dataset_path.stem} with {metric_scores_path.stem}')
        hf_corrs = hf_metrics_corr(dataset, metric_scores, max_pred_words=max_pred_words)

        # Check 2: Results Empty (Implies Size Mismatch or Data Issues)
        if not hf_corrs:
            print(f"  [Error] No correlations computed (likely size mismatch).")
            skipped_datasets.append((dataset_path.stem, "Size mismatch or invalid data"))
            continue

        save_corrs(hf_corrs, output_corrs_path / f'{dataset_path.stem}_hf_corrs.json')

        print(f'  Computing metric-metric correlations...')
        metric_corrs = metric_metric_corr(metric_scores)
        save_corrs(metric_corrs, output_corrs_path / f'{dataset_path.stem}_metric_corrs.json')

    # --- PRINT SUMMARY AT THE END ---
    print("\n" + "="*80)
    print(f"COMPUTATION SUMMARY")
    print("="*80)
    
    if not skipped_datasets:
        print("All eligible datasets processed successfully.")
    else:
        print(f"Total Skipped/Failed: {len(skipped_datasets)}")
        print("-" * 80)
        # Format table with aligned columns
        print(f"{'DATASET NAME':<50} | {'REASON'}")
        print("-" * 80)
        for name, reason in sorted(skipped_datasets, key=lambda x: x[0]):
            print(f"{name:<50} | {reason}")
    print("="*80 + "\n")

def mean_corrs(datasets_folder: Path, corrs_folder: Path, save_path: Path, weight_by_len: bool = False) -> None:
    """Aggregates correlation files into a mean score."""
    print(f"--- Starting Mean Aggregation ---")
    
    lens = {}
    if weight_by_len:
        print("Loading dataset lengths for weighting...")
        for file in datasets_folder.rglob('*.jsonl'):
            if any('backup' in part for part in file.parts): continue
            dataset = ProcessedDataset.load(file)
            lens[file.stem] = len(dataset)

    # Use rglob to find files in subdirectories as well
    corrs_files = list(corrs_folder.rglob('*_hf_corrs.json'))
    if not corrs_files:
        print(f"No correlation files found in {corrs_folder}")
        return

    all_corrs: Dict[str, Dict[str, List[float]]] = {}
    
    for file in corrs_files:
        corrs = json.loads(file.read_text(encoding='utf-8'))
        dataset_name = file.stem.split('_hf_corrs')[0]
        
        for metric_name, metric_corrs in corrs.items():
            if metric_name not in all_corrs:
                all_corrs[metric_name] = {k: [] for k in _CORR.keys()}
            
            for k, v in metric_corrs.items():
                if k in _CORR:
                    if np.isnan(v): continue
                    
                    if weight_by_len:
                        if dataset_name in lens:
                            all_corrs[metric_name][k].append(v * lens[dataset_name])
                        else:
                            print(f"Warning: Length for {dataset_name} not found, skipping weight.")
                    else:
                        all_corrs[metric_name][k].append(v)

    mean_corrs_res: Dict[str, Dict[str, float]] = {}
    total_len = sum(lens.values()) if weight_by_len else 0

    for metric_name, metric_corrs in all_corrs.items():
        # Calculate initial mean (or sum if weighted)
        mean_corrs_res[metric_name] = {
            k: float(np.mean(v) if not weight_by_len else np.sum(v)) if v else float('nan') 
            for k, v in metric_corrs.items()
        }
        
        # Divide by total length if weighted
        if weight_by_len and total_len > 0:
            mean_corrs_res[metric_name] = {
                k: v / total_len for k, v in mean_corrs_res[metric_name].items()
            }
            
    print("\nCalculated Means:")
    pprint(mean_corrs_res)
    save_corrs(mean_corrs_res, save_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate correlations for Human Feedback datasets.")
    
    # Path Arguments
    parser.add_argument("--datasets_dir", type=Path, default="outputs", 
                        help="Folder containing .jsonl datasets (default: outputs)")
    parser.add_argument("--scores_dir", type=Path, default="outputs_metric_scores", 
                        help="Folder containing metric scores (default: outputs_metric_scores)")
    parser.add_argument("--output_dir", type=Path, default="outputs_correlations", 
                        help="Folder to save correlation results (default: outputs_correlations)")
    
    # Operation Arguments
    parser.add_argument("--mode", choices=['compute', 'mean', 'all'], default='all',
                        help="Action to perform: 'compute' (process files), 'mean' (aggregate), or 'all' (default: all)")
    
    # Aggregation Specific Arguments
    parser.add_argument("--mean_output_file", type=Path, default="mean_corrs.json",
                        help="Filename for the aggregated mean correlations (saved inside output_dir)")
    parser.add_argument("--weight_by_len", action="store_true",
                        help="If set, mean aggregation will be weighted by dataset length.")
    parser.add_argument("--max_pred_words", type=int, default=None,
                        help="If set, exclude predictions longer than this many words before computing correlations.")

    args = parser.parse_args()

    # Execution Logic
    if args.mode in ['compute', 'all']:
        compute_correlations(args.datasets_dir, args.scores_dir, args.output_dir, max_pred_words=args.max_pred_words)

    if args.mode in ['mean', 'all']:
        save_path = args.output_dir / args.mean_output_file
        mean_corrs(args.datasets_dir, args.output_dir, save_path, weight_by_len=args.weight_by_len)

if __name__ == "__main__":
    main()