from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .api_runner import generate_damage_with_backend
from .env import load_local_env


def parse_api_keys(raw: Optional[str]) -> List[str]:
    """Parse a JSON array or comma/newline/semicolon-separated Gemini key list."""

    if not raw:
        return []
    text = str(raw).strip()
    if not text:
        return []

    values: Iterable[Any]
    try:
        parsed = json.loads(text)
        values = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        values = re.split(r"[,;\n]+", text)

    keys: List[str] = []
    for value in values:
        key = str(value).strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def resolve_api_keys(
    api_key: Optional[str] = None,
    api_keys: Optional[Iterable[str]] = None,
) -> List[str]:
    """Resolve and de-duplicate explicit and environment-provided Gemini keys."""

    load_local_env()
    candidates: List[str] = []
    if api_key:
        candidates.append(api_key)
    if isinstance(api_keys, str):
        candidates.extend(parse_api_keys(api_keys))
    elif api_keys:
        candidates.extend(str(value) for value in api_keys)
    candidates.extend(parse_api_keys(os.environ.get("GEMINI_API_KEYS")))
    candidates.extend(parse_api_keys(os.environ.get("GEMINI_API_KEY")))

    resolved: List[str] = []
    for value in candidates:
        key = value.strip()
        if key and key not in resolved:
            resolved.append(key)
    return resolved


def _request(url: str, payload: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Gemini network error: {exc}") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError("Gemini request failed after retries")


def call_gemini(messages: List[Dict[str, str]], model: str, api_key: str, max_output_tokens: Optional[int] = 400) -> str:
    system = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
    user = "\n\n".join(item["content"] for item in messages if item["role"] != "system")
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0},
    }
    if max_output_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
    data = _request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", payload)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


def generate_damage(
    input_path: Path,
    output_path: Path,
    model: str,
    prompt_type: str,
    limit: Optional[int] = None,
    max_output_tokens: int = 400,
    api_key: Optional[str] = None,
    api_keys: Optional[Iterable[str]] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Generate/resume one prompt branch. Use --limit for the 20/50-row pilot."""

    keys = resolve_api_keys(api_key=api_key, api_keys=api_keys)
    if not keys:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GEMINI_API_KEYS in src/meta-judge/.env or the shell before calling Gemini."
        )

    key_lock = threading.Lock()
    key_index = 0

    def call_with_key_pool(messages: List[Dict[str, str]], selected_model: str, tokens: int) -> str:
        nonlocal key_index
        with key_lock:
            selected_key = keys[key_index % len(keys)]
            key_index += 1
        return call_gemini(messages, selected_model, selected_key, max_output_tokens=tokens)

    result = generate_damage_with_backend(
        input_path=input_path,
        output_path=output_path,
        model=model,
        prompt_type=prompt_type,
        caller=call_with_key_pool,
        limit=limit,
        max_output_tokens=max_output_tokens,
        backend_name="gemini",
        workers=workers,
    )
    result["api_key_count"] = len(keys)
    return result
