import json
import csv
import tempfile
import unittest
from pathlib import Path

from vn_meta_judge.config import paper_metric_specs
from vn_meta_judge.data.loaders import BenchmarkRow, load_benchmark_xlsx, save_jsonl
from vn_meta_judge.evaluation.correlations import compute_meta_correlation
from vn_meta_judge.evaluation.score_export import export_translations_csv
from vn_meta_judge.generation.api_runner import generate_damage_with_backend
from vn_meta_judge.generation.rule_based import generate_rule_baseline
from vn_meta_judge.preprocessing.vietnamese import strip_vietnamese_diacritics, strip_vietnamese_tone


class VietnamesePipelineTests(unittest.TestCase):
    def test_paper_metric_design_has_28_unique_configs(self):
        specs = paper_metric_specs()
        self.assertEqual(len(specs), 28)
        self.assertEqual(len({spec.key for spec in specs}), 28)

    def test_tone_removal_preserves_vowel_shape(self):
        self.assertEqual(strip_vietnamese_tone("tiếng Việt mà"), "tiêng Viêt ma")
        self.assertEqual(strip_vietnamese_tone("mười"), "mươi")
        self.assertEqual(strip_vietnamese_diacritics("mười"), "muoi")

    def test_rule_baseline_has_six_levels_per_reference(self):
        rows = [BenchmarkRow("id-1", "中文", "tiếng Việt mà", metadata={"stt": 1})]
        output = generate_rule_baseline(rows, seed=42)
        self.assertEqual(len(output), 6)
        self.assertEqual([item["damage_level"] for item in output], list(range(6)))
        self.assertEqual(output[0]["prediction_vi"], "tiếng Việt mà")

    def test_meta_correlation_uses_common_metric_keys(self):
        values = {f"m{i}": {"pearson": i / 10, "spearman": i / 10, "kendall": i / 10} for i in range(3)}
        result = compute_meta_correlation(values, values)
        self.assertEqual(result["common_metrics"], 3)
        self.assertAlmostEqual(result["meta_spearman"], 1.0)

    def test_jsonl_round_trip(self):
        rows = [BenchmarkRow("id-1", "中文", "Tiếng Việt")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            save_jsonl(rows, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["id"], "id-1")

    def test_api_runner_parallel_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "references.jsonl"
            output_path = root / "damage.jsonl"
            save_jsonl([BenchmarkRow("id-1", "中文", "Tiếng Việt")], input_path)
            calls = []

            def caller(messages, model, tokens):
                calls.append(messages[-1]["content"])
                return "Kết quả"

            result = generate_damage_with_backend(
                input_path,
                output_path,
                "mock-model",
                "zero_shot",
                caller,
                workers=4,
                checkpoint_every=2,
            )
            self.assertEqual(result["rows_written"], 6)
            self.assertEqual(result["workers"], 4)
            calls.clear()
            resumed = generate_damage_with_backend(
                input_path,
                output_path,
                "mock-model",
                "zero_shot",
                caller,
                workers=4,
            )
            self.assertEqual(resumed["rows_written"], 6)
            self.assertEqual(calls, [])

    def test_loader_supports_current_primary_score_columns(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame([
                    {"STT": 1, "Cau_nguon_ZH": "中文", "Cau_tham_chieu_VI": "Câu một"},
                    {"STT": 2, "Cau_nguon_ZH": "中文二", "Cau_tham_chieu_VI": "Câu hai"},
                ]).to_excel(writer, sheet_name="Du_lieu_goc", index=False)
                pd.DataFrame([
                    {"STT": 1, "Nguon_ZH": "中文", "Tham_chieu_VI": "Câu một", "Ban_dich_He_A": "A1", "Ban_dich_He_B": "B1", "Diem_A_chinh": 90, "Diem_B_chinh": 80},
                    {"STT": 2, "Nguon_ZH": "中文二", "Tham_chieu_VI": "Câu hai", "Ban_dich_He_A": "A2", "Ban_dich_He_B": "B2", "Diem_A_chinh": None, "Diem_B_chinh": None},
                ]).to_excel(writer, sheet_name="Cham_diem", index=False)
            rows = load_benchmark_xlsx(path, expected_rows=2)
            self.assertEqual([row.human_score for row in rows], [90.0, 80.0, None, None])
            self.assertEqual(rows[0].quality_flags, [])
            self.assertEqual(rows[2].quality_flags, ["invalid_score_A"])

    def test_loader_supports_single_sheet_google_export(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single-sheet.xlsx"
            pd.DataFrame([
                {"STT": 1, "Nguon_ZH": "中文", "Tham_chieu_VI": "Câu một", "Ban_dich_He_A": "A1", "Ban_dich_He_B": "B1", "Diem_A_final": 91, "Diem_B_final": 81},
                {"STT": 2, "Nguon_ZH": "中文二", "Tham_chieu_VI": "Câu hai", "Ban_dich_He_A": "A2", "Ban_dich_He_B": "B2", "Diem_A_final": 92, "Diem_B_final": 82},
            ]).to_excel(path, sheet_name="Cham_diem", index=False)
            rows = load_benchmark_xlsx(path, expected_rows=2)
            self.assertEqual([row.human_score for row in rows], [91.0, 81.0, 92.0, 82.0])
            self.assertEqual(rows[0].quality_flags, [])

    def test_translation_export_is_reusable_and_complete(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xlsx_path = root / "scores.xlsx"
            output_path = root / "translations.csv"
            pd.DataFrame([
                {"STT": 1, "Nguon_ZH": "中文", "Tham_chieu_VI": "Câu một", "Ban_dich_He_A": "A1", "Ban_dich_He_B": "B1", "Diem_A_final": 91, "Diem_B_final": 81},
                {"STT": 2, "Nguon_ZH": "中文二", "Tham_chieu_VI": "Câu hai", "Ban_dich_He_A": "A2", "Ban_dich_He_B": "B2", "Diem_A_final": 92, "Diem_B_final": 82},
            ]).to_excel(xlsx_path, sheet_name="Cham_diem", index=False)

            result = export_translations_csv(xlsx_path, output_path, expected_rows=2)
            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.DictReader(handle))

            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["system_a_rows"], 2)
            self.assertEqual(result["system_b_rows"], 2)
            self.assertEqual(exported[0]["Nguon_ZH"], "中文")
            self.assertEqual(exported[1]["Ban_dich_He_B"], "B2")


if __name__ == "__main__":
    unittest.main()
