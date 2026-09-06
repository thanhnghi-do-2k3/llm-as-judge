"""Reliable Hugging Face checkpoint resolution for COMET models."""

from pathlib import Path
import time
from typing import Callable, Optional, Sequence


DEFAULT_RETRY_DELAYS = (15.0, 30.0, 60.0)


def comet_repo_id(model_name: str) -> str:
    """Normalize legacy short names to the public Unbabel Hub namespace."""
    name = model_name.strip()
    if not name:
        raise ValueError("COMET model name must not be empty.")
    return name if "/" in name else f"Unbabel/{name}"


def _checkpoint_from_snapshot(snapshot_dir: str) -> str:
    root = Path(snapshot_dir)
    preferred = root / "checkpoints" / "model.ckpt"
    if preferred.is_file():
        return str(preferred)
    checkpoints = sorted((root / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) == 1:
        return str(checkpoints[0])
    raise FileNotFoundError(
        f"COMET snapshot {root} does not contain one usable checkpoint."
    )


def _http_status(error: Exception) -> Optional[int]:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _retry_after(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(error: Exception) -> bool:
    status = _http_status(error)
    return status is None or status == 429 or status >= 500


def download_comet_checkpoint(
    model_name: str,
    *,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Resolve a COMET checkpoint, preferring cache and retrying Hub overloads.

    COMET 2.2.x catches every Hugging Face error and reports the misleading
    message "model not supported". Calling the Hub directly keeps the original
    error visible and lets transient 429/5xx responses recover.
    """
    from huggingface_hub import snapshot_download

    repo_id = comet_repo_id(model_name)
    try:
        snapshot = snapshot_download(repo_id=repo_id, local_files_only=True)
        return _checkpoint_from_snapshot(snapshot)
    except Exception:
        pass

    attempts = len(retry_delays) + 1
    for attempt in range(1, attempts + 1):
        try:
            snapshot = snapshot_download(repo_id=repo_id, max_workers=4)
            return _checkpoint_from_snapshot(snapshot)
        except Exception as error:
            if attempt == attempts or not _is_retryable(error):
                raise RuntimeError(
                    f"Không tải được COMET model {repo_id}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            hinted = _retry_after(error)
            delay = max(float(retry_delays[attempt - 1]), hinted or 0.0)
            print(
                f"Hugging Face tạm lỗi khi tải {repo_id}; "
                f"thử lại {attempt + 1}/{attempts} sau {delay:.0f}s."
            )
            sleep_fn(delay)

    raise AssertionError("Unreachable COMET download state.")
