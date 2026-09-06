import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vn_meta_judge.metrics.comet_cache import (
    comet_repo_id,
    download_comet_checkpoint,
)


class FakeResponse:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)


class FakeHubError(Exception):
    def __init__(self, message, status_code, retry_after=None):
        super().__init__(message)
        self.response = FakeResponse(status_code, retry_after)


class CometCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.snapshot = Path(self.temporary.name)
        checkpoint = self.snapshot / "checkpoints" / "model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

    def test_normalizes_short_model_name(self):
        self.assertEqual(
            comet_repo_id("eamt22-cometinho-da"),
            "Unbabel/eamt22-cometinho-da",
        )
        self.assertEqual(
            comet_repo_id("Unbabel/wmt22-comet-da"),
            "Unbabel/wmt22-comet-da",
        )

    @patch("huggingface_hub.snapshot_download")
    def test_prefers_complete_local_snapshot(self, snapshot_download):
        snapshot_download.return_value = str(self.snapshot)
        result = download_comet_checkpoint("Unbabel/wmt22-comet-da")
        self.assertEqual(Path(result).name, "model.ckpt")
        snapshot_download.assert_called_once_with(
            repo_id="Unbabel/wmt22-comet-da",
            local_files_only=True,
        )

    @patch("huggingface_hub.snapshot_download")
    def test_retries_hub_429_and_uses_retry_after(self, snapshot_download):
        snapshot_download.side_effect = [
            FileNotFoundError("not cached"),
            FakeHubError("maximum queue size reached", 429, retry_after=7),
            str(self.snapshot),
        ]
        sleeps = []
        result = download_comet_checkpoint(
            "wmt22-comet-da",
            retry_delays=(1,),
            sleep_fn=sleeps.append,
        )
        self.assertEqual(Path(result).name, "model.ckpt")
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(snapshot_download.call_count, 3)

    @patch("huggingface_hub.snapshot_download")
    def test_preserves_non_retryable_hub_error(self, snapshot_download):
        snapshot_download.side_effect = [
            FileNotFoundError("not cached"),
            FakeHubError("repository not found", 404),
        ]
        with self.assertRaisesRegex(
            RuntimeError,
            "Unbabel/wmt22-comet-da.*repository not found",
        ):
            download_comet_checkpoint("wmt22-comet-da", sleep_fn=lambda _: None)
        self.assertEqual(snapshot_download.call_count, 2)


if __name__ == "__main__":
    unittest.main()
