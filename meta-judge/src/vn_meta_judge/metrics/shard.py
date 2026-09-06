"""Fast, resumable metric scoring for one synthetic branch.

The Colab shard notebook combines the human rows and one synthetic branch so
each model-based metric is loaded once. Scores are split back to their source
files without changing row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import paper_metric_specs
from ..data.loaders import load_jsonl, save_jsonl
from .runner import run_metric_scores


TEXT_FIELDS = ("id", "source_zh", "reference_vi", "prediction_vi")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pending.replace(path)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_common(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"File không có dòng dữ liệu: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dòng {index + 1} không phải JSON object: {path}")
        missing = [name for name in TEXT_FIELDS if not _nonempty_text(row.get(name))]
        if missing:
            raise ValueError(
                f"Dòng {index + 1} thiếu trường văn bản {missing}: {path}"
            )


def _clean_human_rows(
    rows: List[Dict[str, Any]], path: Path
) -> tuple[List[Dict[str, Any]], List[int]]:
    _validate_common(rows, path)
    selected: List[Dict[str, Any]] = []
    indexes: List[int] = []
    for index, row in enumerate(rows):
        score = row.get("human_score")
        flags = row.get("quality_flags") or []
        if flags or not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        selected.append(row)
        indexes.append(index)
    if not selected:
        raise ValueError(
            f"Không có dòng human sạch: cần human_score hữu hạn và quality_flags rỗng ({path})."
        )
    return selected, indexes


def _validate_synthetic_rows(
    rows: List[Dict[str, Any]],
    path: Path,
    branch_name: str,
    require_six_levels: bool,
) -> None:
    _validate_common(rows, path)
    failed = [
        index + 1
        for index, row in enumerate(rows)
        if row.get("status", "ok") != "ok"
    ]
    if failed:
        preview = ", ".join(map(str, failed[:10]))
        raise ValueError(
            f"Synthetic còn {len(failed)} dòng status != ok; ví dụ dòng {preview}."
        )

    prompt_types = {
        str(row.get("prompt_type"))
        for row in rows
        if row.get("prompt_type") is not None
    }
    if prompt_types and prompt_types != {branch_name}:
        raise ValueError(
            f"prompt_type không khớp BRANCH_NAME={branch_name}: {sorted(prompt_types)}"
        )

    levels_by_id: Dict[str, List[int]] = defaultdict(list)
    seen = set()
    for index, row in enumerate(rows):
        level = row.get("damage_level")
        if isinstance(level, bool) or not isinstance(level, (int, float)):
            raise ValueError(f"damage_level không hợp lệ tại dòng {index + 1}: {level!r}")
        level_int = int(level)
        if level_int != level or level_int not in range(6):
            raise ValueError(f"damage_level phải thuộc 0..5 tại dòng {index + 1}")
        identity = (str(row["id"]), level_int)
        if identity in seen:
            raise ValueError(f"Trùng id/damage_level tại dòng {index + 1}: {identity}")
        seen.add(identity)
        levels_by_id[str(row["id"])].append(level_int)

    if require_six_levels:
        incomplete = {
            row_id: sorted(levels)
            for row_id, levels in levels_by_id.items()
            if sorted(levels) != list(range(6))
        }
        if incomplete:
            preview = list(incomplete.items())[:5]
            raise ValueError(f"Mỗi id phải đủ damage_level 0..5; ví dụ: {preview}")


def _select_sentence_range(
    rows: List[Dict[str, Any]],
    path: Path,
    sentence_start: int,
    sentence_end: Optional[int],
) -> tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    """Select 1-based inclusive sentence positions in first-appearance order."""

    ordered_ids = []
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not _nonempty_text(row.get("id")):
            raise ValueError(f"Dòng {index + 1} thiếu id hợp lệ: {path}")
        row_id = str(row["id"])
        if row_id not in seen_ids:
            seen_ids.add(row_id)
            ordered_ids.append(row_id)

    total = len(ordered_ids)
    if isinstance(sentence_start, bool) or not isinstance(sentence_start, int):
        raise ValueError("sentence_start phải là số nguyên, đánh số từ 1.")
    if sentence_end is not None and (
        isinstance(sentence_end, bool) or not isinstance(sentence_end, int)
    ):
        raise ValueError("sentence_end phải là số nguyên hoặc None.")
    actual_end = total if sentence_end is None else sentence_end
    if sentence_start < 1 or actual_end < sentence_start or actual_end > total:
        raise ValueError(
            f"Khoảng câu không hợp lệ: {sentence_start}..{actual_end}; "
            f"file có {total} câu theo thứ tự xuất hiện."
        )

    selected_ids = set(ordered_ids[sentence_start - 1 : actual_end])
    indexes = [
        index for index, row in enumerate(rows) if str(row["id"]) in selected_ids
    ]
    selected = [rows[index] for index in indexes]
    return selected, indexes, {
        "sentence_start": sentence_start,
        "sentence_end": actual_end,
        "selected_sentence_count": actual_end - sentence_start + 1,
        "total_sentence_count": total,
        "is_full_input": sentence_start == 1 and actual_end == total,
        "selection_order": "first appearance of id in source JSONL",
    }


def prepare_shard_inputs(
    synthetic_path: Path,
    combined_path: Path,
    manifest_path: Path,
    *,
    branch_name: str,
    human_path: Optional[Path] = None,
    require_six_levels: bool = True,
    sentence_start: int = 1,
    sentence_end: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate inputs and concatenate them while retaining exact source offsets."""

    synthetic_path = Path(synthetic_path).resolve()
    human_path = Path(human_path).resolve() if human_path else None
    combined_path = Path(combined_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy synthetic JSONL: {synthetic_path}")
    if human_path and not human_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy human JSONL: {human_path}")

    datasets = []
    combined: List[Dict[str, Any]] = []

    def append_dataset(
        name: str,
        kind: str,
        source_path: Path,
        rows: List[Dict[str, Any]],
        indexes: List[int],
        input_rows: int,
    ) -> None:
        start = len(combined)
        combined.extend(rows)
        datasets.append(
            {
                "name": name,
                "kind": kind,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "input_rows": input_rows,
                "selected_rows": len(rows),
                "selected_input_indexes": indexes,
                "start": start,
                "end": len(combined),
            }
        )

    if human_path:
        human_all = load_jsonl(human_path)
        human_rows, human_indexes = _clean_human_rows(human_all, human_path)
        append_dataset(
            "human", "human", human_path, human_rows, human_indexes, len(human_all)
        )

    synthetic_all = load_jsonl(synthetic_path)
    synthetic_rows, synthetic_indexes, sentence_range = _select_sentence_range(
        synthetic_all,
        synthetic_path,
        sentence_start,
        sentence_end,
    )
    _validate_synthetic_rows(
        synthetic_rows,
        synthetic_path,
        branch_name,
        require_six_levels,
    )
    append_dataset(
        branch_name,
        "synthetic",
        synthetic_path,
        synthetic_rows,
        synthetic_indexes,
        len(synthetic_all),
    )
    datasets[-1]["sentence_range"] = sentence_range

    save_jsonl(combined, combined_path)
    manifest = {
        "version": 1,
        "branch_name": branch_name,
        "combined_path": str(combined_path),
        "combined_sha256": _sha256(combined_path),
        "combined_rows": len(combined),
        "sentence_range": sentence_range,
        "datasets": datasets,
        "order_contract": "Scores and scored JSONL rows preserve source-file order.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_multi_dataset_inputs(
    dataset_paths: Dict[str, Path],
    combined_path: Path,
    manifest_path: Path,
    *,
    require_six_levels: bool = True,
    synthetic_sentence_limit: Optional[int] = None,
    human_row_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and combine every dataset so each metric model loads once.

    The first-class shard CLI handles one human/synthetic pair. The main
    notebook has human, rule-based, zero-shot and few-shot branches, so this
    variant preserves an offset and source-index contract for all of them.
    """

    if not dataset_paths:
        raise ValueError("Không có dataset để chấm metric.")
    if synthetic_sentence_limit is not None and synthetic_sentence_limit < 1:
        raise ValueError("synthetic_sentence_limit phải dương hoặc None.")
    if human_row_limit is not None and human_row_limit < 1:
        raise ValueError("human_row_limit phải dương hoặc None.")

    combined_path = Path(combined_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    combined: List[Dict[str, Any]] = []
    datasets = []

    for name, raw_path in dataset_paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy dataset {name}: {path}")
        source_rows = load_jsonl(path)
        if name == "human":
            selected_rows, selected_indexes = _clean_human_rows(source_rows, path)
            if human_row_limit is not None:
                selected_rows = selected_rows[:human_row_limit]
                selected_indexes = selected_indexes[:human_row_limit]
            kind = "human"
        else:
            selected_rows, selected_indexes, _ = _select_sentence_range(
                source_rows,
                path,
                1,
                synthetic_sentence_limit,
            )
            _validate_synthetic_rows(
                selected_rows,
                path,
                name,
                require_six_levels,
            )
            kind = "synthetic"

        start = len(combined)
        combined.extend(selected_rows)
        datasets.append(
            {
                "name": name,
                "kind": kind,
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "input_rows": len(source_rows),
                "selected_rows": len(selected_rows),
                "selected_input_indexes": selected_indexes,
                "start": start,
                "end": len(combined),
            }
        )

    save_jsonl(combined, combined_path)
    manifest = {
        "version": 1,
        "mode": "multi_dataset",
        "combined_path": str(combined_path),
        "combined_sha256": _sha256(combined_path),
        "combined_rows": len(combined),
        "datasets": datasets,
        "order_contract": "Scores and scored JSONL rows preserve source-file order.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _merge_payloads(
    checkpoint_paths: Iterable[Path], combined_rows: int
) -> tuple[Dict[str, List[float]], Dict[str, str], Dict[str, float]]:
    scores: Dict[str, List[float]] = {}
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}
    for path in map(Path, checkpoint_paths):
        if not path.is_file():
            errors[f"missing::{path.name}"] = "checkpoint file does not exist"
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, values in payload.get("scores", {}).items():
            if len(values) != combined_rows:
                errors[key] = (
                    f"checkpoint returned {len(values)} scores; expected {combined_rows}"
                )
                continue
            if key in scores and scores[key] != values:
                raise ValueError(f"Hai checkpoint cho kết quả khác nhau: {key}")
            scores[key] = values
        errors.update(payload.get("errors", {}))
        timings.update(payload.get("timings", {}))
    for key in scores:
        errors.pop(key, None)
    return scores, errors, timings


def finalize_shard_scores(
    manifest_path: Path,
    checkpoint_paths: Iterable[Path],
    result_dir: Path,
    *,
    tokenization: str,
    expected_metric_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Merge task checkpoints and split score vectors back in input order."""

    manifest_path = Path(manifest_path).resolve()
    result_dir = Path(result_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined_rows = int(manifest["combined_rows"])
    scores, errors, timings = _merge_payloads(checkpoint_paths, combined_rows)
    outputs = {}

    for dataset in manifest["datasets"]:
        name = dataset["name"]
        start, end = int(dataset["start"]), int(dataset["end"])
        source_path = Path(dataset["source_path"])
        source_rows = load_jsonl(source_path)
        indexes = dataset["selected_input_indexes"]
        selected_rows = [source_rows[index] for index in indexes]
        split_scores = {key: values[start:end] for key, values in scores.items()}
        if any(len(values) != len(selected_rows) for values in split_scores.values()):
            raise ValueError(f"Không thể tách score đúng offset cho dataset {name}")

        score_path = result_dir / f"scores_{name}_{tokenization}.json"
        scored_path = result_dir / f"scored_{name}_{tokenization}.jsonl"
        _write_json(
            score_path,
            {
                "scores": split_scores,
                "errors": errors,
                "timings": timings,
                "config": {
                    "tokenization": tokenization,
                    "source_sha256": dataset["source_sha256"],
                    "rows": len(selected_rows),
                    "order": "source_file",
                },
            },
        )
        scored_rows = []
        for position, (input_index, row) in enumerate(zip(indexes, selected_rows)):
            item = dict(row)
            item["input_index"] = input_index
            item["metric_scores"] = {
                key: values[position] for key, values in split_scores.items()
            }
            scored_rows.append(item)
        save_jsonl(scored_rows, scored_path)
        outputs[name] = {
            "scores": str(score_path),
            "scored_rows": str(scored_path),
            "rows": len(selected_rows),
        }

    expected = list(expected_metric_keys) if expected_metric_keys is not None else [
        spec.key
        for spec in paper_metric_specs()
        if tokenization == "syllable" or spec.family in {"BLEU", "chrF"}
    ]
    summary = {
        "tokenization": tokenization,
        "combined_rows": combined_rows,
        "completed_metrics": len(scores),
        "expected_metrics": len(expected),
        "missing_metrics": sorted(set(expected) - set(scores)),
        "errors": errors,
        "outputs": outputs,
    }
    _write_json(result_dir / f"summary_{tokenization}.json", summary)
    return summary


def _prepare_heavy_asset(selected_key: Optional[str]) -> None:
    if not selected_key:
        return
    spec = next(spec for spec in paper_metric_specs() if spec.key == selected_key)
    if spec.family == "COMET":
        from vn_meta_judge.metrics.comet_cache import download_comet_checkpoint

        download_comet_checkpoint(spec.kwargs["model_name"])
    elif spec.family == "BLEURT":
        import evaluate
        from metrics.hf_metrics import _find_bleurt_local

        try:
            _find_bleurt_local(spec.kwargs["checkpoint"])
        except FileNotFoundError:
            delays = (15.0, 30.0, 60.0)
            for attempt in range(len(delays) + 1):
                try:
                    evaluate.load(
                        "bleurt",
                        config_name=spec.kwargs["checkpoint"],
                    )
                    break
                except Exception:
                    if attempt == len(delays):
                        raise
                    print(
                        f"BLEURT download tạm lỗi; thử lại sau {delays[attempt]:.0f}s.",
                        flush=True,
                    )
                    time.sleep(delays[attempt])


def metric_task(
    input_path: Path,
    output_path: Path,
    *,
    tokenization: str,
    family: str,
    batch_size: int,
    selected_key: Optional[str] = None,
) -> Dict[str, Any]:
    _prepare_heavy_asset(selected_key)
    result = run_metric_scores(
        input_path,
        output_path,
        tokenization=tokenization,
        selected_families=[family],
        selected_keys=[selected_key] if selected_key else None,
        batch_size=batch_size,
    )
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    expected = (
        [selected_key]
        if selected_key
        else [spec.key for spec in paper_metric_specs() if spec.family == family]
    )
    missing = sorted(set(expected) - set(payload.get("scores", {})))
    if missing:
        raise RuntimeError(
            f"Metric task chưa hoàn tất {missing}: {payload.get('errors', {})}"
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metric shard utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--input", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument(
        "--tokenization", choices=("syllable", "underthesea"), required=True
    )
    worker.add_argument("--family", required=True)
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--selected-key", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "worker":
        result = metric_task(
            args.input,
            args.output,
            tokenization=args.tokenization,
            family=args.family,
            batch_size=args.batch_size,
            selected_key=args.selected_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
