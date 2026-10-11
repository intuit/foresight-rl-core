"""Pulling structured data out of unstructured model output.

Models return JSON wrapped in prose, fenced in markdown, or preceded by a
sentence of explanation despite being told not to. Every metric and benchmark
that wants an object hits the same problem, so the extraction lives here once.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_candidates(text: str) -> List[str]:
    """Return substrings of ``text`` that might parse as JSON, best first.

    Fenced blocks come first: when a model emits both an explanation and a
    fenced result, the fence is the answer.
    """
    if not text:
        return []
    candidates = [block.strip() for block in _FENCE_PATTERN.findall(text)]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start, end = text.find(open_char), text.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    candidates.append(text.strip())
    return [c for c in candidates if c]


def extract_json(text: str) -> Optional[Any]:
    """Return the first parseable JSON value in ``text``, or ``None``."""
    for candidate in json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first parseable JSON **object** in ``text``, or ``None``."""
    for candidate in json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


__all__ = ["extract_json", "extract_json_object", "json_candidates"]
