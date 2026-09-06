from __future__ import annotations

import random
import re
from typing import Any, Dict, Iterable, List

from ..data.loaders import BenchmarkRow
from ..preprocessing.vietnamese import normalize_text, strip_vietnamese_diacritics


def _words(text: str) -> List[str]:
    return normalize_text(text).split(" ") if normalize_text(text) else []


def _drop_words(words: List[str], fraction: float) -> str:
    if len(words) < 3:
        return " ".join(words)
    keep = max(1, round(len(words) * (1 - fraction)))
    indexes = sorted(random.sample(range(len(words)), keep))
    return " ".join(words[index] for index in indexes)


def perturb_reference(reference: str, level: int, seed: int) -> Dict[str, Any]:
    """Deterministic mechanical baseline; it is not intended to mimic LLM semantics."""

    random.seed(seed)
    text = normalize_text(reference)
    words = _words(text)
    if level == 0:
        return {"prediction_vi": text, "operation": "identity"}
    if level == 1:
        tokens = text.split(" ")
        changed = 0
        for index, token in enumerate(tokens):
            candidate = strip_vietnamese_diacritics(token)
            if candidate != token:
                tokens[index] = candidate
                changed += 1
                if changed == 1:
                    break
        return {"prediction_vi": " ".join(tokens), "operation": "remove_one_tone"}
    if level == 2:
        tokens = [strip_vietnamese_diacritics(token) for token in words]
        if len(tokens) > 4:
            tokens.pop(len(tokens) // 2)
        return {"prediction_vi": " ".join(tokens), "operation": "remove_tones_and_drop_one_word"}
    if level == 3:
        return {"prediction_vi": _drop_words(words, 0.15), "operation": "word_dropout_15pct"}
    if level == 4:
        damaged = _drop_words(words, 0.30).split()
        if len(damaged) > 3:
            damaged[1], damaged[2] = damaged[2], damaged[1]
        return {"prediction_vi": " ".join(damaged), "operation": "word_dropout_30pct_and_swap"}
    damaged = _drop_words(words, 0.50)
    damaged = re.sub(r"[,.!?;:]+", "", damaged)
    return {"prediction_vi": damaged, "operation": "word_dropout_50pct_and_punctuation_remove"}


def generate_rule_baseline(rows: Iterable[BenchmarkRow], seed: int = 42) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        for level in range(6):
            result = perturb_reference(row.reference_vi, level, seed=seed + int(row.metadata.get("stt", 0)) + level)
            output.append({
                "id": row.id,
                "source_zh": row.source_zh,
                "reference_vi": row.reference_vi,
                "prediction_vi": result["prediction_vi"],
                "damage_level": level,
                "prompt_type": "rule_based",
                "model_name": "deterministic-rule-baseline",
                "backend": "rule_based",
                "temperature": 0.0,
                "operation": result["operation"],
            })
    return output
