import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from vn_meta_judge.notebook_workflow import GeminiRequestPool


class GeminiRequestPoolTests(unittest.TestCase):
    def test_standard_and_high_quota_profiles_run_in_one_pool(self):
        calls = []

        def caller(messages, model, key, tokens):
            calls.append(key)
            return key

        pool = GeminiRequestPool(
            standard_keys=["standard-one", "standard-two"],
            high_quota_keys=["high-one"],
            standard_workers=1,
            standard_min_interval_seconds=0,
            high_quota_workers_per_key=2,
            high_quota_min_interval_seconds=0,
            caller=caller,
        )

        results = [pool([], "model", 10) for _ in range(4)]

        self.assertEqual(pool.worker_count, 3)
        self.assertEqual(
            results,
            ["standard-one", "high-one", "high-one", "standard-two"],
        )
        self.assertEqual(calls, results)

    def test_key_cannot_belong_to_both_profiles(self):
        with self.assertRaisesRegex(ValueError, "không được nằm đồng thời"):
            GeminiRequestPool(
                standard_keys=["duplicate"],
                high_quota_keys=["duplicate"],
            )

    def test_high_quota_lanes_allow_bounded_parallel_requests(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def caller(messages, model, key, tokens):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return key

        pool = GeminiRequestPool(
            high_quota_keys=["high-one"],
            high_quota_workers_per_key=3,
            high_quota_min_interval_seconds=0,
            caller=caller,
        )
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(lambda _: pool([], "model", 10), range(6)))

        self.assertEqual(results, ["high-one"] * 6)
        self.assertEqual(peak, 3)


if __name__ == "__main__":
    unittest.main()
