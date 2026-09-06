from __future__ import annotations

from pathlib import Path


def load_local_env() -> None:
    """Load only the ignored coursework .env, without replacing shell vars."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=project_root / ".env", override=False)
