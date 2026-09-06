from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from vn_meta_judge.data.loaders import save_jsonl
from vn_meta_judge.notebook_workflow import NotebookExperiment


class DamageReuseTests(unittest.TestCase):
    def test_external_damage_is_selected_without_api_or_output_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            translations = root / "translations.csv"
            pd.DataFrame(
                {
                    "STT": [1],
                    "ID_cau_VLSP": ["sample-1"],
                    "Nguon_ZH": ["中文"],
                    "Tham_chieu_VI": ["Câu tham chiếu"],
                    "Ban_dich_He_A": ["Bản dịch A"],
                    "Ban_dich_He_B": ["Bản dịch B"],
                }
            ).to_csv(translations, index=False)
            experiment = NotebookExperiment(
                root,
                Path(__file__).resolve().parents[1],
                "reuse-test",
                model="test-model",
                limit=None,
                api_keys=["configured-but-unused"],
            )
            experiment.load_translations(translations, expected_rows=1)
            experiment.prepare()

            damage_files = {}
            for branch in ["zero_shot", "few_shot"]:
                path = root / "repo" / f"{branch}.jsonl"
                rows = [
                    {
                        "run_key": f"sample-1|{branch}|{level}",
                        "id": "sample-1",
                        "source_zh": "中文",
                        "reference_vi": "Câu tham chiếu",
                        "prediction_vi": f"Câu mức {level}",
                        "damage_level": level,
                        "prompt_type": branch,
                        "model_name": "test-model",
                        "backend": "gemini",
                        "temperature": 0.0,
                        "status": "ok",
                        "error": None,
                    }
                    for level in range(6)
                ]
                save_jsonl(rows, path)
                damage_files[branch] = path

            generated = experiment.generation_dir / "zero_shot.jsonl"
            generated.write_text("keep generated checkpoint\n", encoding="utf-8")
            before = generated.read_bytes()
            with patch(
                "vn_meta_judge.notebook_workflow.call_gemini",
                side_effect=AssertionError("API must not run"),
            ):
                result = experiment.generate(
                    regenerate_damage=False,
                    damage_files=damage_files,
                )

            self.assertEqual(result["zero_shot"]["status"], "ready")
            self.assertEqual(
                experiment.datasets["zero_shot"],
                damage_files["zero_shot"].resolve(),
            )
            self.assertEqual(generated.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
