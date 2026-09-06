from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..data.loaders import load_jsonl, save_jsonl
from .prompts import render_messages


ApiCaller = Callable[[List[Dict[str, str]], str, int], str]


def generate_damage_with_backend(
    input_path: Path,
    output_path: Path,
    model: str,
    prompt_type: str,
    caller: ApiCaller,
    limit: Optional[int] = None,
    max_output_tokens: int = 400,
    backend_name: str = "api",
    workers: int = 1,
    checkpoint_every: int = 10,
) -> Dict[str, Any]:
    """Generate one isolated damage branch with resumable row checkpoints.

    ``caller`` is deliberately injected so the orchestration is identical for
    Gemini and optional ablation backends.  The model/backend metadata is
    persisted in every row; this makes accidental cross-backend concatenation
    visible during audit.
    """

    raw = load_jsonl(input_path)
    rows = raw[:limit] if limit is not None else raw
    output: List[Dict[str, Any]] = load_jsonl(output_path) if output_path.exists() else []
    original_output = list(output)
    existing_backends = {str(item.get("backend")) for item in output if item.get("backend")}
    if existing_backends and existing_backends != {backend_name}:
        raise RuntimeError(
            f"Refusing to mix backends in {output_path}: found {sorted(existing_backends)}, requested {backend_name}. "
            "Use a separate output file."
        )
    existing_models = {str(item.get("model_name")) for item in output if item.get("model_name")}
    if existing_models and existing_models != {model}:
        raise RuntimeError(
            f"Refusing to mix models in {output_path}: found {sorted(existing_models)}, requested {model}. "
            "Use a separate output file."
        )
    existing = {item["run_key"]: item for item in output if "run_key" in item}
    requested: List[Tuple[Dict[str, Any], int, str]] = []
    for row in rows:
        for level in range(6):
            run_key = f"{row['id']}|{prompt_type}|{level}"
            requested.append((row, level, run_key))

    pending = [
        item for item in requested
        if not (
            item[2] in existing
            and existing[item[2]].get("status") == "ok"
            and existing[item[2]].get("backend") == backend_name
            and existing[item[2]].get("model_name") == model
        )
    ]

    def run_one(row: Dict[str, Any], level: int, run_key: str) -> Dict[str, Any]:
        messages = render_messages(row["source_zh"], row["reference_vi"], level, prompt_type)
        try:
            prediction = caller(messages, model, max_output_tokens).strip()
            status, error = "ok", None
        except Exception as exc:  # preserve row-level failures for resume/audit
            prediction, status, error = "", "error", str(exc)
        return {
            "run_key": run_key,
            "id": row["id"],
            "source_zh": row["source_zh"],
            "reference_vi": row["reference_vi"],
            "prediction_vi": prediction,
            "damage_level": level,
            "prompt_type": prompt_type,
            "model_name": model,
            "backend": backend_name,
            "temperature": 0.0,
            "status": status,
            "error": error,
        }

    def save_ordered() -> None:
        requested_keys = {run_key for _, _, run_key in requested}
        ordered = [existing[run_key] for _, _, run_key in requested if run_key in existing]
        ordered.extend(item for item in output if item.get("run_key") not in requested_keys)
        save_jsonl(ordered, output_path)

    worker_count = max(1, int(workers))
    checkpoint = max(1, int(checkpoint_every))
    completed = 0
    if worker_count == 1:
        for row, level, run_key in pending:
            existing[run_key] = run_one(row, level, run_key)
            completed += 1
            if completed % checkpoint == 0:
                save_ordered()
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(run_one, row, level, run_key) for row, level, run_key in pending]
            for future in as_completed(futures):
                item = future.result()
                existing[item["run_key"]] = item
                completed += 1
                if completed % checkpoint == 0:
                    save_ordered()
    save_ordered()
    requested_keys = {run_key for _, _, run_key in requested}
    output = [existing[run_key] for _, _, run_key in requested if run_key in existing]
    output.extend(item for item in original_output if item.get("run_key") not in requested_keys)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_requested": len(rows),
        "rows_expected": len(rows) * 6,
        "rows_written": len(output),
        "errors": sum(item.get("status") == "error" for item in output),
        "model": model,
        "backend": backend_name,
        "prompt_type": prompt_type,
        "temperature": 0.0,
        "workers": worker_count,
        "pending_before_run": len(pending),
    }
