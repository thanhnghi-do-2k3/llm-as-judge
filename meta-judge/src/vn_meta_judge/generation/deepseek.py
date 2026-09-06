from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .api_runner import generate_damage_with_backend
from .env import load_local_env


DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def call_deepseek(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    max_output_tokens: int = 400,
) -> str:
    """Call DeepSeek's OpenAI-compatible chat-completions endpoint.

    This adapter is intentionally explicit.  It is an optional generator
    ablation, not an automatic fallback for a failed Gemini request.
    """

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, AttributeError) as exc:
                raise RuntimeError(f"Unexpected DeepSeek response: {data}") from exc
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"DeepSeek network error: {exc}") from exc
        time.sleep(2**attempt)
    raise RuntimeError("DeepSeek request failed after retries")


def generate_damage(
    input_path: Path,
    output_path: Path,
    model: str,
    prompt_type: str,
    limit: Optional[int] = None,
    max_output_tokens: int = 400,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
) -> Dict[str, Any]:
    """Generate a separate DeepSeek ablation branch with resume support."""

    load_local_env()
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in src/meta-judge/.env or the shell before running the optional DeepSeek branch."
        )
    return generate_damage_with_backend(
        input_path=input_path,
        output_path=output_path,
        model=model,
        prompt_type=prompt_type,
        caller=lambda messages, selected_model, tokens: call_deepseek(
            messages, selected_model, api_key, base_url=base_url, max_output_tokens=tokens
        ),
        limit=limit,
        max_output_tokens=max_output_tokens,
        backend_name="deepseek",
    )
