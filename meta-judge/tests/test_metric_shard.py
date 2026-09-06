import json
import tempfile
import types
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

from vn_meta_judge.data.loaders import load_jsonl, save_jsonl
from vn_meta_judge.config import paper_metric_specs
from vn_meta_judge.metrics.shard import (
    _prepare_heavy_asset,
    finalize_shard_scores,
    prepare_shard_inputs,
)


class MetricShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_bleurt_prefetch_selects_requested_config_name(self):
        selected = next(
            spec
            for spec in paper_metric_specs()
            if spec.family == "BLEURT"
            and spec.kwargs["checkpoint"] == "bleurt-tiny-128"
        )
        evaluate = types.ModuleType("evaluate")
        evaluate.load = Mock()
        hf_metrics = types.ModuleType("metrics.hf_metrics")
        hf_metrics._find_bleurt_local = Mock(side_effect=FileNotFoundError)
        with patch.dict(
            sys.modules,
            {"evaluate": evaluate, "metrics.hf_metrics": hf_metrics},
        ):
            _prepare_heavy_asset(selected.key)
        evaluate.load.assert_called_once_with(
            "bleurt",
            config_name="bleurt-tiny-128",
        )

    def test_combines_clean_human_and_preserves_synthetic_order(self):
        human = self.root / "human_all.jsonl"
        synthetic = self.root / "few_shot.jsonl"
        save_jsonl(
            [
                {
                    "id": "h-1",
                    "source_zh": "中文",
                    "reference_vi": "Tham chiếu",
                    "prediction_vi": "Dự đoán một",
                    "human_score": 90.0,
                    "quality_flags": [],
                },
                {
                    "id": "h-2",
                    "source_zh": "中文二",
                    "reference_vi": "Tham chiếu hai",
                    "prediction_vi": "Dự đoán hai",
                    "human_score": 80.0,
                    "quality_flags": ["excluded"],
                },
            ],
            human,
        )
        synthetic_rows = [
            {
                "id": "s-1",
                "source_zh": "中文",
                "reference_vi": "Câu tham chiếu",
                "prediction_vi": f"Câu mức {level}",
                "damage_level": level,
                "prompt_type": "few_shot",
                "status": "ok",
            }
            for level in reversed(range(6))
        ]
        save_jsonl(synthetic_rows, synthetic)

        combined = self.root / "combined.jsonl"
        manifest_path = self.root / "manifest.json"
        manifest = prepare_shard_inputs(
            synthetic,
            combined,
            manifest_path,
            branch_name="few_shot",
            human_path=human,
        )

        self.assertEqual(manifest["combined_rows"], 7)
        self.assertEqual(manifest["datasets"][0]["selected_input_indexes"], [0])
        self.assertEqual(
            [row["damage_level"] for row in load_jsonl(combined)[1:]],
            list(reversed(range(6))),
        )

        checkpoint = self.root / "metric.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "scores": {
                        "metric-a": [float(index) for index in range(7)],
                        "metric-b": [float(index + 10) for index in range(7)],
                    },
                    "errors": {},
                    "timings": {"metric-a": 1.0, "metric-b": 2.0},
                }
            ),
            encoding="utf-8",
        )
        result = finalize_shard_scores(
            manifest_path,
            [checkpoint],
            self.root / "results",
            tokenization="syllable",
            expected_metric_keys=["metric-a", "metric-b"],
        )

        self.assertEqual(result["missing_metrics"], [])
        scored = load_jsonl(self.root / "results/scored_few_shot_syllable.jsonl")
        self.assertEqual([row["input_index"] for row in scored], list(range(6)))
        self.assertEqual(
            [row["damage_level"] for row in scored], list(reversed(range(6)))
        )
        self.assertEqual(scored[0]["metric_scores"]["metric-a"], 1.0)

    def test_rejects_wrong_prompt_and_incomplete_levels(self):
        synthetic = self.root / "few_shot.jsonl"
        row = {
            "id": "s-1",
            "source_zh": "中文",
            "reference_vi": "Câu tham chiếu",
            "prediction_vi": "Câu dự đoán",
            "damage_level": 0,
            "prompt_type": "zero_shot",
            "status": "ok",
        }
        save_jsonl([row], synthetic)
        with self.assertRaisesRegex(ValueError, "prompt_type"):
            prepare_shard_inputs(
                synthetic,
                self.root / "combined.jsonl",
                self.root / "manifest.json",
                branch_name="few_shot",
            )

        row["prompt_type"] = "few_shot"
        save_jsonl([row], synthetic)
        with self.assertRaisesRegex(ValueError, "0..5"):
            prepare_shard_inputs(
                synthetic,
                self.root / "combined.jsonl",
                self.root / "manifest.json",
                branch_name="few_shot",
            )

    def test_selects_inclusive_sentence_range_and_keeps_source_row_indexes(self):
        synthetic = self.root / "few_shot.jsonl"
        rows = []
        for sentence in range(1, 4):
            for level in range(6):
                rows.append(
                    {
                        "id": f"s-{sentence}",
                        "source_zh": f"中文{sentence}",
                        "reference_vi": f"Tham chiếu {sentence}",
                        "prediction_vi": f"Dự đoán {sentence}-{level}",
                        "damage_level": level,
                        "prompt_type": "few_shot",
                        "status": "ok",
                    }
                )
        # A failure outside the requested range must not block scoring that range.
        rows[0]["status"] = "error"
        rows[0]["prediction_vi"] = ""
        save_jsonl(rows, synthetic)

        manifest = prepare_shard_inputs(
            synthetic,
            self.root / "combined.jsonl",
            self.root / "manifest.json",
            branch_name="few_shot",
            sentence_start=2,
            sentence_end=3,
        )

        selected = manifest["datasets"][-1]
        self.assertEqual(selected["selected_rows"], 12)
        self.assertEqual(selected["selected_input_indexes"], list(range(6, 18)))
        self.assertEqual(
            manifest["sentence_range"],
            {
                "sentence_start": 2,
                "sentence_end": 3,
                "selected_sentence_count": 2,
                "total_sentence_count": 3,
                "is_full_input": False,
                "selection_order": "first appearance of id in source JSONL",
            },
        )
        self.assertEqual(
            [row["id"] for row in load_jsonl(self.root / "combined.jsonl")],
            ["s-2"] * 6 + ["s-3"] * 6,
        )

        checkpoint = self.root / "metric-range.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "scores": {"metric-a": [float(i) for i in range(12)]},
                    "errors": {},
                    "timings": {},
                }
            ),
            encoding="utf-8",
        )
        finalize_shard_scores(
            self.root / "manifest.json",
            [checkpoint],
            self.root / "results",
            tokenization="syllable",
            expected_metric_keys=["metric-a"],
        )
        scored = load_jsonl(self.root / "results/scored_few_shot_syllable.jsonl")
        self.assertEqual(
            [row["input_index"] for row in scored], list(range(6, 18))
        )

    def test_rejects_sentence_range_outside_input(self):
        synthetic = self.root / "few_shot.jsonl"
        save_jsonl(
            [
                {
                    "id": "s-1",
                    "source_zh": "中文",
                    "reference_vi": "Tham chiếu",
                    "prediction_vi": f"Dự đoán {level}",
                    "damage_level": level,
                    "prompt_type": "few_shot",
                    "status": "ok",
                }
                for level in range(6)
            ],
            synthetic,
        )
        with self.assertRaisesRegex(ValueError, "file có 1 câu"):
            prepare_shard_inputs(
                synthetic,
                self.root / "combined.jsonl",
                self.root / "manifest.json",
                branch_name="few_shot",
                sentence_start=2,
                sentence_end=2,
            )

if __name__ == "__main__":
    unittest.main()
