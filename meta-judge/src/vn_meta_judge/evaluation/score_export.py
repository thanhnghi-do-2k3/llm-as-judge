from __future__ import annotations

import json
import math
import csv
from pathlib import Path
from typing import Any, Dict

from ..data.loaders import export_benchmark_artifacts, load_benchmark_xlsx, load_jsonl


def _safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def export_machine_scores(xlsx_path: Path, output_dir: Path, expected_rows: int = 300) -> Dict[str, Any]:
    """Export the workbook's human scores as model-readable JSON and JSONL.

    The canonical benchmark artifacts are written beside the machine export so
    the score context remains auditable. Final scores are preserved exactly;
    primary/cross scores are exposed as separate fields and alignment flags are
    never silently deleted.
    """

    summary = export_benchmark_artifacts(xlsx_path, output_dir, expected_rows=expected_rows)
    rows = load_jsonl(output_dir / "human_benchmark_all.jsonl")
    machine_rows = []
    for row in rows:
        flags = list(row.get("quality_flags") or [])
        excluded = bool(
            "reference_alignment_issue" in flags
            or "missing_text" in flags
            or any(flag.startswith("invalid_score_") for flag in flags)
        )
        metadata = row.get("metadata") or {}
        machine_rows.append({
            "record_type": "human_translation_score",
            "schema_version": "1.0",
            "id": row.get("id"),
            "system_id": row.get("system_id"),
            "source_zh": row.get("source_zh"),
            "reference_vi": row.get("reference_vi"),
            "prediction_vi": row.get("prediction_vi"),
            "final_score": row.get("human_score"),
            "primary_score": metadata.get("primary_score"),
            "cross_score": metadata.get("cross_score"),
            "primary_rater": metadata.get("primary_rater"),
            "cross_rater": metadata.get("cross_rater"),
            "quality_flags": flags,
            "row_status": "excluded_from_clean_correlation" if excluded else "clean",
            "do_not_use_for_final_correlation": excluded,
            "score_source": "Google Sheet final score: Diem_A_final/Diem_B_final, falling back to Diem_*_chinh",
            "score_semantics": "holistic human translation score on a 0-100 scale",
            "notes": {
                "common_note": metadata.get("common_note"),
                "ai_note": metadata.get("ai_note"),
            },
        })

    jsonl_path = output_dir / "human_scoring_machine.jsonl"
    json_path = output_dir / "human_scoring_machine.json"
    jsonl_path.write_text(
        "\n".join(json.dumps(_safe(row), ensure_ascii=False) for row in machine_rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "1.0",
        "record_type": "human_translation_score_dataset",
        "purpose": "Machine-readable export of the latest human scoring workbook for analysis or model context.",
        "source_workbook": str(xlsx_path),
        "score_scale": {"min": 0, "max": 100, "unit": "points"},
        "score_policy": {
            "final_score": "Use final_score as the selected human score.",
            "primary_score": "First independent score when present.",
            "cross_score": "Cross-check score when present; it is not silently averaged here.",
            "quality_rule": "Rows marked do_not_use_for_final_correlation must be excluded from clean correlation.",
        },
        "reader_instructions": [
            "For a row-level score, read final_score, prediction_vi and reference_vi.",
            "Use row_status=clean for the clean benchmark branch.",
            "Keep excluded rows for audit; do not silently delete them.",
            "The score is a human judgment, not a BLEU/COMET/model score.",
        ],
        "summary": summary,
        "row_count": len(machine_rows),
        "clean_row_count": sum(row["row_status"] == "clean" for row in machine_rows),
        "excluded_row_count": sum(row["row_status"] != "clean" for row in machine_rows),
        "rows": machine_rows,
    }
    json_path.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "row_count": len(machine_rows),
        "clean_row_count": payload["clean_row_count"],
        "excluded_row_count": payload["excluded_row_count"],
        "summary": summary,
    }


def export_translations_csv(xlsx_path: Path, output_path: Path, expected_rows: int = 300) -> Dict[str, Any]:
    """Export one reusable row per source with both existing A/B translations."""

    rows = load_benchmark_xlsx(xlsx_path, expected_rows=expected_rows)
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = merged.setdefault(
            row.id,
            {
                "STT": row.metadata.get("stt"),
                "ID_cau_VLSP": row.id,
                "Nguon_ZH": row.source_zh,
                "Tham_chieu_VI": row.reference_vi,
                "Ban_dich_He_A": "",
                "Ban_dich_He_B": "",
            },
        )
        if item["Nguon_ZH"] != row.source_zh or item["Tham_chieu_VI"] != row.reference_vi:
            raise ValueError(f"Source/reference mismatch between systems for {row.id}.")
        if row.system_id not in {"A", "B"}:
            raise ValueError(f"Unexpected system_id={row.system_id!r} for {row.id}.")
        item[f"Ban_dich_He_{row.system_id}"] = row.prediction_vi or ""

    output_rows = sorted(merged.values(), key=lambda item: int(item["STT"]))
    if expected_rows and len(output_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} translation rows, found {len(output_rows)}.")
    missing = [
        item["STT"]
        for item in output_rows
        if not item["Nguon_ZH"]
        or not item["Tham_chieu_VI"]
        or not item["Ban_dich_He_A"]
        or not item["Ban_dich_He_B"]
    ]
    if missing:
        raise ValueError(f"Missing source/reference/A/B translation at STT: {missing[:20]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "STT",
        "ID_cau_VLSP",
        "Nguon_ZH",
        "Tham_chieu_VI",
        "Ban_dich_He_A",
        "Ban_dich_He_B",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "input": str(xlsx_path),
        "output": str(output_path),
        "row_count": len(output_rows),
        "system_a_rows": sum(bool(item["Ban_dich_He_A"]) for item in output_rows),
        "system_b_rows": sum(bool(item["Ban_dich_He_B"]) for item in output_rows),
    }
