"""Reuse archived metric results from the repository in the main notebook.

The cache loader is deliberately read-only with respect to the source archive:
it validates and extracts the archive into the current run directory, then
points ``NotebookExperiment`` at the extracted scored JSONL files.  Metric
workers are not started in this mode.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import csv
from pathlib import Path
from zipfile import ZipFile


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_source_metric_cache(
    experiment,
    cache_dir: Path,
    *,
    branches=("zero_shot", "few_shot"),
    tokenizations=("syllable", "underthesea"),
) -> dict:
    """Load archived results and attach them to ``experiment``.

    The source archives must contain scored JSONL and metric score JSON for
    every requested branch/tokenization.  A missing or malformed cache raises
    an explicit error so a normal ``Run All`` cannot silently recreate scores.
    Set the notebook's rerun flag when a fresh metric run is intended.
    """

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục metric cache: {cache_dir}")

    archive_paths = {}
    for branch in branches:
        candidates = sorted(cache_dir.glob(f"metric-results-{branch}-*.zip"))
        if not candidates:
            raise FileNotFoundError(
                f"Không tìm thấy archive metric cho {branch} trong {cache_dir}"
            )
        archive_paths[branch] = candidates[-1]

    extracted_root = experiment.output / "source-metric-cache"
    extracted_root.mkdir(parents=True, exist_ok=True)
    result_root = experiment.metrics_dir / "fast" / "results"
    result_root.mkdir(parents=True, exist_ok=True)

    # The human score file is shared by both branch archives.  Prefer the
    # first branch and verify that both archives agree on its row count.
    human_paths = []
    report = {"status": "cached", "source_dir": str(cache_dir), "branches": {}}
    for branch, archive in archive_paths.items():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        branch_root = extracted_root / branch
        marker = branch_root / ".archive-sha256"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != digest:
            if branch_root.exists():
                shutil.rmtree(branch_root)
            branch_root.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, branch_root)
            marker.write_text(digest + "\n", encoding="utf-8")

        branch_result = {"archive": str(archive), "tokenizations": {}}
        for tokenization in tokenizations:
            scored = branch_root / "results" / f"scored_{branch}_{tokenization}.jsonl"
            scores = branch_root / "results" / f"scores_{branch}_{tokenization}.json"
            human = branch_root / "results" / f"scored_human_{tokenization}.jsonl"
            human_scores = branch_root / "results" / f"scores_human_{tokenization}.json"
            if (
                not scored.is_file()
                or not scores.is_file()
                or not human.is_file()
                or not human_scores.is_file()
            ):
                raise FileNotFoundError(
                    f"Cache thiếu output {branch}/{tokenization}: {scored}, {scores}, "
                    f"{human}, hoặc {human_scores}"
                )
            payload = _json(scores)
            metric_count = len(payload.get("scores", {}))
            row_count = _line_count(scored)
            if metric_count == 0 or row_count == 0:
                raise ValueError(f"Cache rỗng: {scores}")
            for values in payload["scores"].values():
                if len(values) != row_count:
                    raise ValueError(
                        f"Số điểm không khớp số dòng trong cache: {scores}"
                    )
            human_row_count = _line_count(human)
            human_payload = _json(human_scores)
            if not human_payload.get("scores") or any(
                len(values) != human_row_count
                for values in human_payload["scores"].values()
            ):
                raise ValueError(
                    f"Số điểm human không khớp số dòng trong cache: {human_scores}"
                )
            destination = result_root / f"scores_{branch}_{tokenization}.json"
            shutil.copy2(scores, destination)
            shutil.copy2(
                human_scores,
                result_root / f"scores_human_{tokenization}.json",
            )
            # ``correlate`` validates score lengths against these paths.
            experiment.datasets[branch] = scored
            human_paths.append(human)
            branch_result["tokenizations"][tokenization] = {
                "rows": row_count,
                "metrics": metric_count,
                "scores": str(destination),
            }

        human_rows = _line_count(human_paths[-1])
        expected_human_rows = (
            _line_count(experiment.human_file)
            if getattr(experiment, "human_file", None)
            else len(experiment.human_rows)
        )
        if human_rows != expected_human_rows:
            raise ValueError(
                f"Cache human có {human_rows} dòng, input hiện tại có "
                f"{expected_human_rows} dòng: {human_paths[-1]}"
            )
        report["branches"][branch] = branch_result

    # The archived human score vectors are stored in both archives.  Use the
    # first one as the dataset used by ``correlate`` and keep the clean human
    # file from ``experiment.prepare`` for the human correlation calculation.
    experiment.datasets["human"] = human_paths[0]
    report["human_rows"] = _line_count(human_paths[0])
    report["status"] = "ready"
    return report


def load_baseline_csv_cache(
    experiment,
    csv_path: Path,
    *,
    branch: str = "rule_based",
    tokenization: str = "syllable",
) -> dict:
    """Attach a pre-scored rule-based baseline CSV without running workers."""

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy baseline cache: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "id",
        "source_zh",
        "reference_vi",
        "prediction_vi",
        "damage_level",
        "prompt_type",
        "model_name",
        "backend",
        "temperature",
        "operation",
        "status",
    }
    metric_columns = [name for name in (rows[0] if rows else {}) if name.startswith("metric__")]
    if not rows or not required.issubset(rows[0]) or not metric_columns:
        raise ValueError(f"Baseline cache thiếu schema hoặc metric columns: {csv_path}")
    keys = {(row["id"], int(row["damage_level"])) for row in rows}
    if len(rows) != 1800 or len(keys) != 1800:
        raise ValueError(f"Baseline cache phải có 1.800 khóa (id, level): {csv_path}")
    per_id = {}
    for row in rows:
        per_id.setdefault(row["id"], []).append(int(row["damage_level"]))
    if len(per_id) != 300 or any(sorted(levels) != list(range(6)) for levels in per_id.values()):
        raise ValueError(f"Baseline cache phải có 300 câu × level 0..5: {csv_path}")

    extracted = experiment.output / "source-metric-cache" / branch
    extracted.mkdir(parents=True, exist_ok=True)
    dataset_path = extracted / f"scored_{branch}_{tokenization}.jsonl"
    score_path = experiment.metrics_dir / "fast" / "results" / f"scores_{branch}_{tokenization}.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    scores = {name.removeprefix("metric__"): [] for name in metric_columns}
    for row in rows:
        item = {
            "id": row["id"],
            "source_zh": row["source_zh"],
            "reference_vi": row["reference_vi"],
            "prediction_vi": row["prediction_vi"],
            "damage_level": int(row["damage_level"]),
            "prompt_type": row["prompt_type"],
            "model_name": row["model_name"],
            "backend": row["backend"],
            "temperature": float(row["temperature"]),
            "operation": row["operation"],
            "status": row["status"],
        }
        normalized.append(item)
        for name in metric_columns:
            scores[name.removeprefix("metric__")].append(float(row[name]))
    dataset_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in normalized),
        encoding="utf-8",
    )
    score_path.write_text(
        json.dumps({"scores": scores}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    experiment.datasets[branch] = dataset_path
    return {
        "status": "ready",
        "branch": branch,
        "tokenization": tokenization,
        "rows": len(normalized),
        "metrics": len(scores),
        "source": str(csv_path),
        "scores": str(score_path),
    }
