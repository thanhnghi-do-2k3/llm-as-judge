from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from ..data.loaders import load_jsonl


def _corr(xs: List[float], ys: List[float]) -> Dict[str, float]:
    if len(xs) != len(ys) or len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {name: float("nan") for name in ("pearson", "spearman", "kendall")}
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    return {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic), "kendall": float(kendalltau(x, y).statistic)}


def _read_scores(path: Path) -> Dict[str, List[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("scores", payload)


def compute_human_correlations(human_jsonl: Path, scores_json: Path) -> Dict[str, Dict[str, float]]:
    rows = load_jsonl(human_jsonl)
    human = [row.get("human_score") for row in rows]
    if any(score is None for score in human):
        raise ValueError(f"Human score missing in {human_jsonl}; export clean/all explicitly first.")
    scores = _read_scores(scores_json)
    return {key: _corr([float(x) for x in human], [float(x) for x in values]) for key, values in scores.items() if len(values) == len(rows)}


def compute_synthetic_correlations(synthetic_jsonl: Path, scores_json: Path) -> Dict[str, Dict[str, float]]:
    rows = [row for row in load_jsonl(synthetic_jsonl) if row.get("status", "ok") == "ok" and row.get("prediction_vi", "")]
    scores = _read_scores(scores_json)
    # Equation (2): quality score is correlated with -damage_level.
    labels = [-float(row["damage_level"]) for row in rows]
    kept_indexes = [index for index, row in enumerate(load_jsonl(synthetic_jsonl)) if row.get("status", "ok") == "ok" and row.get("prediction_vi", "")]
    return {key: _corr(labels, [float(values[index]) for index in kept_indexes]) for key, values in scores.items() if len(values) == len(load_jsonl(synthetic_jsonl))}


def compute_meta_correlation(human: Dict[str, Dict[str, float]], synthetic: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    common = sorted(set(human).intersection(synthetic))
    return {
        "common_metrics": len(common),
        "metrics": common,
        "meta_pearson": _corr([human[key]["pearson"] for key in common], [synthetic[key]["pearson"] for key in common])["pearson"],
        "meta_spearman": _corr([human[key]["spearman"] for key in common], [synthetic[key]["spearman"] for key in common])["spearman"],
        "meta_kendall": _corr([human[key]["kendall"] for key in common], [synthetic[key]["kendall"] for key in common])["kendall"],
    }


def write_correlation_report(human_jsonl: Path, human_scores: Path, synthetic_jsonl: Path, synthetic_scores: Path, output: Path) -> Dict[str, Any]:
    human = compute_human_correlations(human_jsonl, human_scores)
    synthetic = compute_synthetic_correlations(synthetic_jsonl, synthetic_scores)
    report = {"human": human, "synthetic": synthetic, "meta": compute_meta_correlation(human, synthetic), "equations": {"human": "rho(metric(D), human_score)", "synthetic": "rho(metric(D_syn), -damage_level)", "meta": "rho(r_hum, r_syn)"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return report
