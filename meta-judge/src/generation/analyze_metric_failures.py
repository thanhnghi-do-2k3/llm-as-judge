import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from human_feedback_datasets.processed_dataset import ProcessedDataset
from metrics.correlation import safe_corr, _CORR

def corr(x: List[float], y: List[float]) -> float:
    return safe_corr("spearman", x, y)

def parse_id(ex_id: str) -> Tuple[str, int]:
    """
    Parses '0_level0' -> ('0', 0)
    Parses '15_level3' -> ('15', 3)
    """
    try:
        # Split by the last underscore to separate group from level
        parts = ex_id.rsplit('_', 1)
        group_id = parts[0]
        level_str = parts[1] # 'level0'
        level_num = int(level_str.replace('level', ''))
        return group_id, level_num
    except (IndexError, ValueError):
        return "unknown", -1

def analyze_damage_groups(dataset_path: Path, metrics_path: Path):
    dataset = ProcessedDataset.load(dataset_path)
    metric_scores: Dict[str, List[float]] = json.loads(metrics_path.read_text(encoding='utf-8'))
    
    # 1. Group Data by Source ID
    # groups[group_id] = { level_num: (dataset_index, example_obj) }
    groups = defaultdict(dict)
    
    for idx, ex in enumerate(dataset):
        grp_id, lvl = parse_id(ex.id)
        if lvl != -1:
            groups[grp_id][lvl] = (idx, ex)

    print(f"Found {len(groups)} unique source groups.")
    
    # 2. Analyze Metrics
    for metric_name, scores in metric_scores.items():
        print(f"\n{'='*40}")
        print(f"METRIC: {metric_name}")
        print(f"{'='*40}")
        
        # Counters
        total_comparisons = 0
        failures = 0
        worst_failures = [] # Store (score_diff, group_id, lvl_clean, lvl_damaged)

        # Iterate over every group (e.g., Story #0, Story #1...)
        for grp_id, variants in groups.items():
            
            # We want to compare the "Cleanest" available version (usually level 0)
            # against all other "Damaged" versions (level > 0).
            if 0 not in variants: 
                continue # Skip if no clean baseline
            
            idx_clean, ex_clean = variants[0]
            score_clean = scores[idx_clean]
            
            # Compare Clean (0) vs Damaged (1, 2, 3, 4)
            for lvl in sorted(variants.keys()):
                if lvl == 0: continue # Don't compare with self
                
                idx_dmg, ex_dmg = variants[lvl]
                score_dmg = scores[idx_dmg]
                
                total_comparisons += 1
                
                # FAILURE CONDITION:
                # If the metric thinks Damaged (Level X) is BETTER than Clean (Level 0).
                # Assuming standard Quality Metric (Higher = Better).
                # Failure: Score(Damaged) > Score(Clean)
                if score_dmg > score_clean:
                    failures += 1
                    diff = score_dmg - score_clean
                    worst_failures.append({
                        'diff': diff,
                        'grp': grp_id,
                        'lvl_bad': lvl,
                        'score_clean': score_clean,
                        'score_bad': score_dmg,
                        'txt_clean': ex_clean.prediction,
                        'txt_bad': ex_dmg.prediction
                    })

        # 3. Report Stats
        if total_comparisons > 0:
            fail_rate = (failures / total_comparisons) * 100
            print(f"Failure Rate (Clean vs Damaged): {fail_rate:.2f}%")
            print(f"  (Metric preferred the damaged version in {failures}/{total_comparisons} pairs)")
        else:
            print("No valid comparisons found (check IDs).")

        # 4. Show Worst Examples
        if worst_failures:
            # Sort by biggest difference (Metric was MOST confident about the WRONG answer)
            worst_failures.sort(key=lambda x: x['diff'], reverse=True)
            
            print(f"\n  [Top 3 Worst Failures for {metric_name}]")
            for i, fail in enumerate(worst_failures[:3]):
                print(f"  {i+1}. ID: {fail['grp']} (Level 0 vs Level {fail['lvl_bad']})")
                print(f"     Metric Score: Clean={fail['score_clean']:.4f} vs Damaged={fail['score_bad']:.4f} (Diff: +{fail['diff']:.4f})")
                print(f"     Clean Output:   {fail['txt_clean'][:100]}...")
                print(f"     Damaged Output: {fail['txt_bad'][:100]}...")
                print("     --------------------------------------------------")

def worst_correlated_examples(metric_scores: List[float], dataset: ProcessedDataset, top_k: int = 5) -> List[Dict]:
    """
    Identifies groups where the metric score fails to drop as damage level increases.
    Returns the top_k groups with the highest (most positive/least negative) correlation.
    """
    
    # 1. Group by Source ID
    # groups[grp_id] = list of (level, score, prediction)
    groups = defaultdict(list)
    
    for idx, ex in enumerate(dataset):
        grp_id, _ = parse_id(ex.id) if isinstance(ex.id, str) else (ex.id, None)
        lvl = ex.processed_human_score
        # Store score, level, and text
        groups[grp_id].append({
            "level": lvl,
            "score": metric_scores[idx],
            "text": ex.prediction
        })

    results = []

    # 2. Calculate Correlation per Group
    for grp_id, items in groups.items():
        if len(items) < 2:
            continue
            
        # Sort items by level for consistent checking
        items.sort(key=lambda x: x["level"])
        
        scores = [x["score"] for x in items]
        levels = [x["level"] for x in items]
        
        # Calculate Spearman correlation
        c = corr(scores, levels)
        
        # Check if NaN (happens if all scores are identical)
        # If scores are identical, correlation is effectively 0 (no signal), which is "bad".
        if np.isnan(c):
            c = 1.0 # Treat constant scores as "Worst" failure (Model didn't damage or Metric didn't react)

        results.append({
            "group_id": grp_id,
            "correlation": c,
            "items": items
        })

    # 3. Sort by "Worst" Correlation
    # For Quality Metrics (BLEU/COMET):
    #   Good = -1.0 (Score drops as Level increases)
    #   Bad  = +1.0 (Score increases as Level increases) or 0.0 (Random)
    # We sort Descending (reverse=True) to put positive correlations first.
    results.sort(key=lambda x: x["correlation"], reverse=True)
    
    return results[:top_k]

def get_word_count(text: str) -> int:
    if not text: return 0
    return len(text.split())

def analyze_length_vs_reference(dataset_path: Path):
    print(f"Loading {dataset_path.name}...")
    dataset = ProcessedDataset.load(dataset_path)
    
    # Store ratios per level: stats[level] = [ratio1, ratio2, ...]
    stats = defaultdict(list)
    
    print(f"Processing {len(dataset)} examples...")

    for ex in dataset:
        id, lvl = parse_id(ex.id)
        if lvl == -1: continue

        pred_len = get_word_count(ex.prediction)
        ref_len = get_word_count(ex.reference)

        # Avoid division by zero
        if ref_len == 0:
            continue

        # Ratio: 1.0 means same length. 0.5 means half as long. 1.5 means 50% longer.
        ratio = pred_len / ref_len
        stats[lvl].append(ratio)

    # --- Print Report ---
    print("\n" + "="*70)
    print(f"{'Level':<10} | {'Avg Ratio (Pred/Ref)':<25} | {'Interpretation'}")
    print("="*70)
    
    for lvl in sorted(stats.keys()):
        ratios = stats[lvl]
        avg_ratio = np.mean(ratios)
        
        # Create a visual interpretation string
        pct = avg_ratio * 100
        
        if 95 <= pct <= 105:
            interp = "Same length (~100%)"
        elif pct < 95:
            interp = f"Shorter ({pct:.1f}%)"
        else:
            interp = f"Longer  ({pct:.1f}%)"
            
        print(f"Level {lvl:<4} | {avg_ratio:<25.4f} | {interp}")

    print("="*70)

def save_worst_examples_to_json(worst_cases: List[Dict], output_path: Path, metric_name: str):
    """
    Saves the analyzed failures to a JSON file for manual inspection.
    """
    output_data = {
        "metric": metric_name,
        "failures": worst_cases
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"   >>> Saved worst examples to: {output_path.name}")

if __name__ == "__main__":
    d_folder = Path("/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/cus_qa/")
    # d_folder = Path("/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/damaged_datasets/rose/")
    results_folder = Path("inspection_results") 
    results_folder.mkdir(exist_ok=True)

    for dataset_file in d_folder.glob("*.jsonl"):
        corresponding_metrics_file = dataset_file.parent.parent.parent / f"eval_results/cus_qa/{dataset_file.stem}.json"
        
        print(f"\nAnalyzing dataset: {dataset_file.name}")
        if not corresponding_metrics_file.exists():
            print(f"  [Skipping] No metrics file: {corresponding_metrics_file.name}")
            continue
            
        metric_scores_map = json.loads(corresponding_metrics_file.read_text(encoding='utf-8'))
        dataset = ProcessedDataset.load(dataset_file)
        
        # --- COLLECT DATA ---
        all_metrics_failures = [] # List to hold all metric results for this dataset

        for metric_name, scores in metric_scores_map.items():
            worst_cases = worst_correlated_examples(scores, dataset, top_k=5)
            
            if not worst_cases:
                continue

            # Create dictionary for this metric
            record = {
                "metric": metric_name,
                "dataset": dataset_file.name,
                "description": "Top groups where metric score correlates POSITIVELY with damage level (Failure cases).",
                "failures": worst_cases
            }
            all_metrics_failures.append(record)

        # --- SAVE AS READABLE JSON ---
        if all_metrics_failures:
            # Change extension to .json (standard format)
            output_path = results_folder / f"FAILURES_{dataset_file.stem}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                print(f"  Writing {len(all_metrics_failures)} metric reports to: {output_path.name}")
                # indent=4 makes it readable (pretty-printed)
                json.dump(all_metrics_failures, f_out, indent=4, ensure_ascii=False)