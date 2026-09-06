from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import pandas as pd

from vn_meta_judge.translation_workflow import translate_dataset


class TranslationRequestPoolTests(unittest.TestCase):
    def test_request_caller_handles_system_a_without_direct_keys(self):
        frame = pd.DataFrame(
            {
                "STT": [1, 2],
                "Cau_nguon_ZH": ["中文一", "中文二"],
                "Cau_tham_chieu_VI": ["Một", "Hai"],
            }
        )
        request_caller = Mock(return_value="Bản dịch từ pool")
        target_caller = Mock(side_effect=lambda text: "B " + text)

        with tempfile.TemporaryDirectory() as folder:
            result = translate_dataset(
                frame,
                Path(folder),
                api_keys=[],
                model="test",
                request_caller=request_caller,
                target_caller=target_caller,
            )

        self.assertEqual(request_caller.call_count, 2)
        self.assertEqual(target_caller.call_count, 2)
        self.assertEqual(result.Ban_dich_He_A.tolist(), ["Bản dịch từ pool"] * 2)

    def test_source_workers_use_request_pool_concurrently(self):
        frame = pd.DataFrame(
            {
                "STT": list(range(1, 7)),
                "Cau_nguon_ZH": [f"中文{i}" for i in range(6)],
                "Cau_tham_chieu_VI": [f"Câu {i}" for i in range(6)],
            }
        )
        active = 0
        peak = 0
        lock = threading.Lock()

        def request_caller(messages, model, tokens):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return "Bản dịch từ pool"

        with tempfile.TemporaryDirectory() as folder:
            translate_dataset(
                frame,
                Path(folder),
                api_keys=[],
                model="test",
                request_caller=request_caller,
                target_caller=lambda text: "B " + text,
                source_workers=3,
                target_workers=1,
            )

        self.assertEqual(peak, 3)


if __name__ == "__main__":
    unittest.main()
