from pathlib import Path
import json
from typing import Dict, List, Any, Tuple
from pprint import pprint
import numpy as np

def add_metrics(metric_closeness: Dict[str, Dict[str, int]], first: str, second: str, corrs: List[str]) -> None:
    if first not in metric_closeness:
        metric_closeness[first] = {}
    if second not in metric_closeness:
        metric_closeness[second] = {}
    for c in corrs:
        if c not in metric_closeness[first]:
            metric_closeness[first][c] = {'close': 0, 'total': 0}
        if c not in metric_closeness[second]:
            metric_closeness[second][c] = {'close': 0, 'total': 0}

def print_closeness(metric_closeness: Dict[str, Dict[str, Dict[str, int]]], atleast: float = 0.01) -> None:
    for metric, corrs in metric_closeness.items():
        print(f"Metric: {metric}")
        for c, values in corrs.items():
            close = values['close']
            total = values['total']
            percentage = (close / total * 100) if total > 0 else 0
            if percentage >= atleast * 100:
                print(f"  Correlation: {c}, Close: {close}, Total: {total}, Percentage: {percentage:.2f}%, Close Metrics: ", end="")
                close_metrics = [str(m) + " (" + str(count) + ")" for m, count in values.items()]
                print(", ".join(close_metrics))
        print()

if __name__ == "__main__":
    threshold = 0.98
    metric_closeness: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metric_corr_mean: Dict[str, Dict[str, float]] = {}
    corrs_to_use: List[str] = ["pearson", "spearman", "kendall"]
    total = 0

    folder = Path("/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/corr_results")
    for file in folder.rglob("*metric_corrs.json"):
        metric_corrs = json.load(open(file, "r"))
        for base_metric, corrs in metric_corrs.items():
            for two_metrics, corr_values in corrs.items():
                if 'mean' in two_metrics:
                    continue
                first, second = two_metrics.split("***")
                add_metrics(metric_closeness, first, second, corrs_to_use)
                if first not in metric_corr_mean:
                    metric_corr_mean[first] = {c: 0.0 for c in corrs_to_use}
                if second not in metric_corr_mean:
                    metric_corr_mean[second] = {c: 0.0 for c in corrs_to_use}

                for c in corrs_to_use:
                    val = corr_values[c]
                    if np.isnan(val):
                        print(f"NaN value for {first} and {second} in correlation {c} in file {file}")
                        continue
                    if val >= threshold:
                        metric_closeness[first][c]['close'] += 1
                        if second not in metric_closeness[first][c]:
                            metric_closeness[first][c][second] = 0
                        metric_closeness[second][c]['close'] += 1
                        if first not in metric_closeness[second][c]:
                            metric_closeness[second][c][first] = 0
                        metric_closeness[first][c][second] += 1
                        metric_closeness[second][c][first] += 1
                    metric_closeness[first][c]['total'] += 1
                    metric_closeness[second][c]['total'] += 1

                    metric_corr_mean[first][c] += val
                    metric_corr_mean[second][c] += val
            total += 1
    # print("Metric closeness summary:")
    # pprint(metric_closeness)
    # print_closeness(metric_closeness)
    pprint({k: {c: v / total for c, v in vals.items()} for k, vals in metric_corr_mean.items()})

