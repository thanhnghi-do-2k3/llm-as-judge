import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics.hf_metrics import _find_bleurt_local


class BleurtCachePathTests(unittest.TestCase):
    def test_finds_current_evaluate_metrics_cache_layout(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = (
                Path(root)
                / "metrics"
                / "bleurt"
                / "bleurt-tiny-128"
                / "run-hash"
                / "downloads"
                / "extracted"
                / "archive-hash"
                / "bleurt-tiny-128"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "bleurt_config.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"HF_HOME": root}):
                result = _find_bleurt_local("bleurt-tiny-128")
            self.assertEqual(Path(result), checkpoint)

    def test_keeps_legacy_datasets_cache_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = (
                Path(root)
                / "datasets"
                / "downloads"
                / "extracted"
                / "archive-hash"
                / "BLEURT-20-D12"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "bleurt_config.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"HF_HOME": root}):
                result = _find_bleurt_local("BLEURT-20-D12")
            self.assertEqual(Path(result), checkpoint)


if __name__ == "__main__":
    unittest.main()
