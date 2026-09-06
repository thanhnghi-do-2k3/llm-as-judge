"""Offline smoke test only; this file is not production code and never calls Gemini.

Run from ``src/meta-judge`` with ``PYTHONPATH=src``. It exercises the same
JSONL contract and Eq. 1-3 algorithm with deterministic fake model/metric
outputs, so developers can verify the wiring before installing heavy metrics
or spending API quota.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vn_meta_judge.data.loaders import BenchmarkRow, save_jsonl
from vn_meta_judge.evaluation.correlations import write_correlation_report


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vn-meta-mock-") as directory:
        root = Path(directory)
        human_rows = [
            {"id": f"id-{i}", "source_zh": "中文", "reference_vi": "Câu tham chiếu", "prediction_vi": "Câu hệ thống", "human_score": float(i + 1)}
            for i in range(6)
        ]
        synthetic_rows = [
            {"id": f"id-{i}", "source_zh": "中文", "reference_vi": "Câu tham chiếu", "prediction_vi": f"Câu hư level {i}", "damage_level": i}
            for i in range(6)
        ]
        save_jsonl(human_rows, root / "human.jsonl")
        save_jsonl(synthetic_rows, root / "synthetic.jsonl")
        (root / "human_scores.json").write_text(json.dumps({"scores": {f"metric_{i}": [1, 2, 3, 4, 5, 6] for i in range(3)}}), encoding="utf-8")
        (root / "synthetic_scores.json").write_text(json.dumps({"scores": {f"metric_{i}": [6, 5, 4, 3, 2, 1] for i in range(3)}}), encoding="utf-8")
        report = write_correlation_report(root / "human.jsonl", root / "human_scores.json", root / "synthetic.jsonl", root / "synthetic_scores.json", root / "report.json")
        assert report["meta"]["common_metrics"] == 3
        print(json.dumps({"status": "PASS", "common_metrics": report["meta"]["common_metrics"], "report": str(root / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
