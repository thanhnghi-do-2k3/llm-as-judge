from __future__ import annotations

import re
import unicodedata
from typing import Literal


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def strip_vietnamese_tone(text: str) -> str:
    """Remove tone marks while retaining Vietnamese vowel shape when possible."""

    tone_marks = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
    output = []
    for char in unicodedata.normalize("NFD", text or ""):
        if unicodedata.combining(char) and char in tone_marks:
            continue
        output.append(char)
    return unicodedata.normalize("NFC", "".join(output))


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove all combining marks (the plan's stronger ``mất dấu`` variant)."""

    decomposed = unicodedata.normalize("NFD", text or "")
    return unicodedata.normalize("NFC", "".join(char for char in decomposed if not unicodedata.combining(char)))


def tokenize_text(text: str, mode: Literal["syllable", "underthesea"] = "syllable") -> str:
    text = normalize_text(text)
    if mode == "syllable":
        return text
    if mode != "underthesea":
        raise ValueError(f"Unknown Vietnamese tokenization mode: {mode}")
    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise RuntimeError("Install underthesea to run --tokenization underthesea.") from exc
    return word_tokenize(text, format="text")
