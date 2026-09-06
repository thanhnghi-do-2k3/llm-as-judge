from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..data.loaders import load_jsonl


META_PATTERNS = (
    re.compile(r"^\s*(?:level|damage_level)\s*[:=]", re.I),
    re.compile(r"^\s*(?:here is|bản viết lại|rewritten summary)\b", re.I),
    re.compile(r"^\s*đây là bản (?:viết lại|tóm tắt)\b", re.I),
)


def audit_damage_file(path: Path, expected_per_id: int = 6, manual_audit_path: Optional[Path] = None) -> Dict[str, Any]:
    rows = load_jsonl(path)
    by_id: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row.get("id", ""))].append(row)
    levels = Counter(row.get("damage_level") for row in rows)
    errors = [row for row in rows if row.get("status") == "error"]
    empty = [row for row in rows if row.get("status", "ok") == "ok" and not str(row.get("prediction_vi", "")).strip()]
    meta_text = [row for row in rows if any(pattern.search(str(row.get("prediction_vi", ""))) for pattern in META_PATTERNS)]
    duplicate_keys = [key for key, values in Counter(row.get("run_key") for row in rows if row.get("run_key")).items() if values > 1]
    backend_counts = Counter(str(row.get("backend", "unknown")) for row in rows)
    explicit_backends = {backend for backend in backend_counts if backend != "unknown"}
    mixed_backends = len(explicit_backends) > 1
    missing_levels = {
        item_id: sorted(set(range(expected_per_id)) - {row.get("damage_level") for row in values})
        for item_id, values in by_id.items()
        if len(values) != expected_per_id or {row.get("damage_level") for row in values} != set(range(expected_per_id))
    }
    result: Dict[str, Any] = {
        "input": str(path),
        "rows": len(rows),
        "unique_ids": len(by_id),
        "level_counts": {str(level): count for level, count in sorted(levels.items(), key=lambda item: str(item[0]))},
        "api_error_rows": len(errors),
        "empty_output_rows": len(empty),
        "meta_text_rows": len(meta_text),
        "duplicate_run_keys": len(duplicate_keys),
        "backend_counts": dict(sorted(backend_counts.items())),
        "mixed_backends": mixed_backends,
        "ids_with_missing_or_duplicate_levels": missing_levels,
        "pass_basic_schema": not errors and not empty and not meta_text and not duplicate_keys and not mixed_backends and not missing_levels,
        "manual_audit": None,
    }
    if manual_audit_path and manual_audit_path.exists():
        manual = load_jsonl(manual_audit_path)
        valid = [row for row in manual if row.get("requested_level") is not None and row.get("observed_level") is not None]
        if valid:
            absolute_errors = [abs(int(row["requested_level"]) - int(row["observed_level"])) for row in valid]
            confusion = {str(requested): {str(observed): 0 for observed in range(expected_per_id)} for requested in range(expected_per_id)}
            for row in valid:
                requested, observed = int(row["requested_level"]), int(row["observed_level"])
                if requested in range(expected_per_id) and observed in range(expected_per_id):
                    confusion[str(requested)][str(observed)] += 1
            result["manual_audit"] = {
                "rows": len(valid),
                "exact_accuracy": sum(error == 0 for error in absolute_errors) / len(valid),
                "mean_absolute_level_error": sum(absolute_errors) / len(valid),
                "confusion": confusion,
            }
    return result
