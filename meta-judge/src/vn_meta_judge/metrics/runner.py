from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import paper_metric_specs
from ..data.loaders import load_jsonl
from ..preprocessing.vietnamese import tokenize_text


def _load_author_metric_functions():
    """Lazy adapter to the downloaded author's implementation.

    Importing it only inside a metric run keeps data/prompt/rule-baseline
    commands usable on a CPU machine without the heavy metric stack.
    """

    try:
        from metrics.hf_metrics import METRIC_FNS
    except ImportError as exc:
        raise RuntimeError(
            "Author metric code could not be imported. Run from src/meta-judge "
            "with PYTHONPATH=src and install requirements-metrics.txt."
        ) from exc
    return METRIC_FNS


def _metric_inputs(rows: List[Dict[str, Any]], tokenization: str):
    refs = [tokenize_text(row["reference_vi"], tokenization) for row in rows]
    preds = [tokenize_text(row["prediction_vi"], tokenization) for row in rows]
    srcs = [row.get("source_zh", "") for row in rows]
    return refs, preds, srcs


def run_metric_scores(
    input_path: Path,
    output_path: Path,
    tokenization: str = "syllable",
    include_bleurt: bool = True,
    selected_families: Optional[Iterable[str]] = None,
    batch_size: int = 32,
    selected_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Score a synthetic or human JSONL file with a resumable 28-config run."""

    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows in {input_path}")
    metrics = _load_author_metric_functions()
    families = (
        {item.lower() for item in selected_families} if selected_families else None
    )
    specs = [
        spec
        for spec in paper_metric_specs(include_bleurt=include_bleurt)
        if families is None or spec.family.lower() in families
    ]
    if selected_keys is not None:
        keys = set(selected_keys)
        specs = [spec for spec in specs if spec.key in keys]
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    results: Dict[str, List[Any]] = {}
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    if output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        config = saved.get("config", {})
        if config.get("tokenization", tokenization) != tokenization:
            raise ValueError(
                "Metric checkpoint tokenization differs; use a new output path."
            )
        if config.get("input_sha256", fingerprint) != fingerprint:
            raise ValueError("Metric checkpoint input differs; use a new output path.")
        results = saved.get("scores", saved if isinstance(saved, dict) else {})
        errors = saved.get("errors", {}) if isinstance(saved, dict) else {}
        timings = saved.get("timings", {}) if isinstance(saved, dict) else {}
    refs, preds, srcs = _metric_inputs(rows, tokenization)
    for spec in specs:
        if spec.key in results and len(results[spec.key]) == len(rows):
            continue
        started = time.time()
        try:
            fn = metrics[spec.name]
            values: List[Any] = []
            for start in range(0, len(rows), batch_size):
                r, p, s = (
                    refs[start : start + batch_size],
                    preds[start : start + batch_size],
                    srcs[start : start + batch_size],
                )
                values.extend(
                    fn(r, p, src=s, **spec.kwargs)
                    if spec.name == "comet"
                    else fn(r, p, **spec.kwargs)
                )
            if len(values) != len(rows):
                raise ValueError(f"returned {len(values)} scores for {len(rows)} rows")
            results[spec.key] = values
            errors.pop(spec.key, None)
        except Exception as exc:
            errors[spec.key] = f"{type(exc).__name__}: {exc}"
        timings[spec.key] = round(time.time() - started, 4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "scores": results,
                    "errors": errors,
                    "timings": timings,
                    "config": {
                        "tokenization": tokenization,
                        "input_sha256": fingerprint,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "completed_metrics": len(results),
        "failed_metrics": len(errors),
        "tokenization": tokenization,
    }
