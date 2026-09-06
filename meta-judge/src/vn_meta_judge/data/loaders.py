from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BenchmarkRow:
    """Canonical row shared by human and synthetic branches."""

    id: str
    source_zh: str
    reference_vi: str
    system_id: Optional[str] = None
    prediction_vi: Optional[str] = None
    human_score: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _number(value: Any) -> Optional[float]:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_value(row: Any, columns: Iterable[str]) -> Any:
    """Return the first non-empty value among workbook-version aliases."""

    for column in columns:
        value = row.get(column)
        if _clean(value):
            return value
    return None


def save_jsonl(rows: Iterable[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _read_xlsx(path: Path, sheet: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading the group's Google Sheet export requires pandas and openpyxl.") from exc
    return pd.read_excel(path, sheet_name=sheet)


def _is_data_row(value: Any) -> bool:
    try:
        int(float(value))
        return True
    except (TypeError, ValueError):
        return False


def load_benchmark_xlsx(path: Path, expected_rows: int = 300) -> List[BenchmarkRow]:
    """Load the current Google Sheet export without trusting its summary tab.

    ``Du_lieu_goc`` supplies source/reference alignment when the workbook has
    the multi-sheet template. Some Google Sheet exports contain only
    ``Cham_diem``; in that schema the source/reference columns in that sheet
    are the alignment source. The summary worksheet is treated as a
    human-readable note, never as the calculation source.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading the group's Google Sheet export requires pandas and openpyxl.") from exc

    with pd.ExcelFile(path) as workbook:
        sheets = workbook.sheet_names
    if "Cham_diem" not in sheets:
        raise ValueError(f"Workbook must contain a Cham_diem sheet; found {sheets}.")
    scored = _read_xlsx(path, "Cham_diem")
    scored = scored[scored["STT"].map(_is_data_row)].copy()
    scored["STT"] = scored["STT"].map(lambda value: int(float(value)))

    original_by_id = None
    if "Du_lieu_goc" in sheets:
        original = _read_xlsx(path, "Du_lieu_goc")
        original = original[original["STT"].map(_is_data_row)].copy()
        original["STT"] = original["STT"].map(lambda value: int(float(value)))
        if expected_rows and len(original) != expected_rows:
            raise ValueError(f"Expected {expected_rows} source rows, found {len(original)} in Du_lieu_goc.")
        if len(scored) != len(original):
            raise ValueError(f"Source/scored row count mismatch: {len(original)} vs {len(scored)}.")
        if original["STT"].duplicated().any():
            raise ValueError("STT must be unique in Du_lieu_goc.")
        original_by_id = original.set_index("STT").to_dict("index")
    elif expected_rows and len(scored) != expected_rows:
        raise ValueError(f"Expected {expected_rows} scored rows, found {len(scored)} in Cham_diem.")

    if scored["STT"].duplicated().any():
        raise ValueError("STT must be unique in Cham_diem.")

    rows: List[BenchmarkRow] = []
    for _, item in scored.sort_values("STT").iterrows():
        number = int(item["STT"])
        if original_by_id is not None:
            source = _clean(original_by_id[number].get("Cau_nguon_ZH"))
            reference = _clean(original_by_id[number].get("Cau_tham_chieu_VI"))
        else:
            source = _clean(_first_value(item, ("Nguon_ZH", "Cau_nguon_ZH")))
            reference = _clean(_first_value(item, ("Tham_chieu_VI", "Cau_tham_chieu_VI")))
        sheet_source = _clean(item.get("Nguon_ZH"))
        sheet_reference = _clean(item.get("Tham_chieu_VI"))
        common_flags: List[str] = []
        if original_by_id is not None and (source != sheet_source or reference != sheet_reference):
            common_flags.append("sheet_alignment_mismatch")
        common_note = _clean(_first_value(item, ("Ghi_chu_chung", "Ghi_chu")))
        ai_note = _clean(item.get("Ket_luan_AI"))
        if "LỖI DỮ LIỆU GỐC" in ai_note or "Lỗi dóng hàng" in common_note or "lỗi dữ liệu" in common_note.lower():
            common_flags.append("reference_alignment_issue")
        for system_id, prediction_col, score_col in (
            ("A", "Ban_dich_He_A", ("Diem_A_final", "Diem_A_chinh")),
            ("B", "Ban_dich_He_B", ("Diem_B_final", "Diem_B_chinh")),
        ):
            flags = list(common_flags)
            prediction = _clean(item.get(prediction_col))
            score = _number(_first_value(item, score_col))
            primary_score = _number(item.get("Diem_A_chinh" if system_id == "A" else "Diem_B_chinh"))
            if not source or not reference or not prediction:
                flags.append("missing_text")
            if score is None or not 0 <= score <= 100:
                flags.append(f"invalid_score_{system_id}")
            rows.append(BenchmarkRow(
                id=f"vi-zh-2022-test-{number:04d}",
                source_zh=source,
                reference_vi=reference,
                system_id=system_id,
                prediction_vi=prediction,
                human_score=score,
                quality_flags=sorted(set(flags)),
                metadata={
                    "stt": number,
                    "common_note": common_note,
                    "ai_note": ai_note,
                    "primary_rater": _clean(item.get("Nguoi_cham_chinh")),
                    "primary_score": primary_score,
                    "cross_rater": _clean(item.get("Nguoi_cham_cheo")),
                    "cross_score": _number(item.get("Diem_A_cheo" if system_id == "A" else "Diem_B_cheo")),
                },
            ))
    return rows


def export_benchmark_artifacts(path: Path, output_dir: Path, expected_rows: int = 300) -> Dict[str, Any]:
    rows = load_benchmark_xlsx(path, expected_rows=expected_rows)
    references: Dict[str, BenchmarkRow] = {}
    for row in rows:
        references[row.id] = BenchmarkRow(
            id=row.id,
            source_zh=row.source_zh,
            reference_vi=row.reference_vi,
            quality_flags=[flag for flag in row.quality_flags if not flag.startswith("invalid_score_")],
            metadata=row.metadata,
        )
    human_all = rows
    human_clean = [row for row in rows if "reference_alignment_issue" not in row.quality_flags]
    human_scored = [
        row for row in human_clean
        if row.human_score is not None and not any(flag.startswith("invalid_score_") for flag in row.quality_flags)
    ]
    human_cross = [
        {
            "id": row.id,
            "system_id": row.system_id,
            "source_zh": row.source_zh,
            "reference_vi": row.reference_vi,
            "prediction_vi": row.prediction_vi,
            "primary_score": row.human_score,
            "cross_score": row.metadata.get("cross_score"),
            "primary_rater": row.metadata.get("primary_rater"),
            "cross_rater": row.metadata.get("cross_rater"),
            "metadata": row.metadata,
        }
        for row in human_clean
        if row.metadata.get("cross_score") is not None
    ]
    save_jsonl(references.values(), output_dir / "references_300.jsonl")
    save_jsonl(human_all, output_dir / "human_benchmark_all.jsonl")
    save_jsonl(human_clean, output_dir / "human_benchmark_clean.jsonl")
    save_jsonl(human_scored, output_dir / "human_benchmark_scored.jsonl")
    save_jsonl(human_cross, output_dir / "human_cross_ratings.jsonl")
    flagged = sorted({row.id for row in rows if "reference_alignment_issue" in row.quality_flags})
    score_audit: Dict[str, Any] = {}
    for system in sorted({row.system_id for row in rows if row.system_id}):
        system_rows = [row for row in rows if row.system_id == system]
        arithmetic_mismatches = []
        for row in system_rows:
            primary = row.metadata.get("primary_score")
            final = row.human_score
            cross = row.metadata.get("cross_score")
            if primary is None or final is None:
                continue
            expected = primary if cross is None else (primary + cross) / 2
            # A whole-number final score can be a rounded mean; only larger
            # deviations are treated as a manual override requiring review.
            if abs(final - expected) > 0.5:
                arithmetic_mismatches.append(row)
        score_audit[system] = {
            "primary_score_rows": sum(row.metadata.get("primary_score") is not None for row in system_rows),
            "cross_score_rows": sum(row.metadata.get("cross_score") is not None for row in system_rows),
            "final_score_rows": sum(row.human_score is not None for row in system_rows),
            "arithmetic_mismatch_rows_gt_0_5": len(arithmetic_mismatches),
            "alignment_override_rows": sum("reference_alignment_issue" in row.quality_flags for row in arithmetic_mismatches),
            "unannotated_mismatch_rows": sum("reference_alignment_issue" not in row.quality_flags for row in arithmetic_mismatches),
        }
    summary = {
        "input": str(path),
        "reference_rows": len(references),
        "human_rows_all": len(human_all),
        "human_rows_clean": len(human_clean),
        "human_rows_scored": len(human_scored),
        "human_rows_missing_score": sum(row.human_score is None for row in human_clean),
        "human_cross_rows": len(human_cross),
        "primary_raters": sorted({row.metadata.get("primary_rater") for row in rows if row.metadata.get("primary_rater")}),
        "cross_raters": sorted({row.metadata.get("cross_rater") for row in rows if row.metadata.get("cross_rater")}),
        "primary_score_rows_by_system": {
            system: sum(row.system_id == system and row.metadata.get("primary_score") is not None for row in rows)
            for system in sorted({row.system_id for row in rows if row.system_id})
        },
        "cross_score_rows_by_system": {
            system: sum(row.system_id == system and row.metadata.get("cross_score") is not None for row in rows)
            for system in sorted({row.system_id for row in rows if row.system_id})
        },
        "final_score_rows_by_system": {
            system: sum(row.system_id == system and row.human_score is not None for row in rows)
            for system in sorted({row.system_id for row in rows if row.system_id})
        },
        "score_audit": score_audit,
        "systems": sorted({row.system_id for row in rows if row.system_id}),
        "flagged_reference_ids": flagged,
        "score_missing": sum(row.human_score is None for row in rows),
        "quality_flag_counts": {
            flag: sum(flag in row.quality_flags for row in rows)
            for flag in sorted({flag for row in rows for flag in row.quality_flags})
        },
        "ready_for_r_hum": not any(row.human_score is None for row in human_clean) and not flagged,
        "decision": "Use clean/all/scored explicitly; missing scores and flagged rows are never silently discarded.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
