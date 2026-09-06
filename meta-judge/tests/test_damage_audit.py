import json
import tempfile
import unittest
from pathlib import Path

from vn_meta_judge.generation.audit import audit_damage_file


class DamageAuditTests(unittest.TestCase):
    def test_audit_detects_complete_six_level_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "damage.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for level in range(6):
                    handle.write(json.dumps({"run_key": f"id|{level}", "id": "id", "damage_level": level, "prediction_vi": "v"}, ensure_ascii=False) + "\n")
            result = audit_damage_file(path)
            self.assertTrue(result["pass_basic_schema"])
            self.assertEqual(result["level_counts"], {str(level): 1 for level in range(6)})

    def test_audit_does_not_reject_natural_deixis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "damage.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for level in range(6):
                    text = "Đây là một câu tiếng Việt." if level == 0 else "v"
                    handle.write(json.dumps({"run_key": f"id|{level}", "id": "id", "damage_level": level, "prediction_vi": text}, ensure_ascii=False) + "\n")
            result = audit_damage_file(path)
            self.assertEqual(result["meta_text_rows"], 0)

    def test_audit_rejects_mixed_explicit_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for level in range(6):
                    handle.write(json.dumps({
                        "run_key": f"id|{level}", "id": "id", "damage_level": level,
                        "prediction_vi": "v", "backend": "gemini" if level < 3 else "deepseek",
                    }) + "\n")
            result = audit_damage_file(path)
            self.assertTrue(result["mixed_backends"])
            self.assertFalse(result["pass_basic_schema"])


if __name__ == "__main__":
    unittest.main()
