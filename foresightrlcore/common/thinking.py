"""Removing a reasoning model's inline scratchpad from its output.

A reasoning model narrates before it answers, inline and in the same string.
Anything downstream that reads the answer -- a multiple-choice extractor, an
exact-match comparison, a judge prompt -- reads the narration too unless it is
removed first, and the narration is precisely where the model writes the wrong
answers it considered and rejected. ``"the answer is A ... on reflection, B"``
scores as A.

Tags are configurable because they are not standardised. ``<think>`` is the
Qwen/DeepSeek convention and ``<reasoning>`` turns up elsewhere, but an
internal model is free to invent its own, and a caller that meets one should
not need a change here.

A near-duplicate of :func:`strip_thinking` lives in
``foresightrlcore/tasks/thinking_utils.py``. That copy is bundled into a
serverless reward Lambda at the archive root, where there is no
``foresightrlcore`` package to import from, so it cannot be replaced by a
re-export of this module.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Pattern, Sequence, Tuple

#: Tags stripped when a caller does not name its own.
DEFAULT_THINKING_TAGS: Tuple[str, ...] = ("think", "reasoning")

_PATTERN_CACHE: Dict[str, Tuple[Pattern[str], Pattern[str], Pattern[str]]] = {}


def _patterns(tag: str) -> Tuple[Pattern[str], Pattern[str], Pattern[str]]:
    """Return the (block, open, close) patterns for ``tag``, compiled once."""
    cached = _PATTERN_CACHE.get(tag)
    if cached is None:
        name = re.escape(tag)
        flags = re.IGNORECASE | re.DOTALL
        cached = (
            re.compile(rf"<{name}\b[^>]*>.*?</{name}\s*>", flags),
            re.compile(rf"<{name}\b[^>]*>", re.IGNORECASE),
            re.compile(rf"</{name}\s*>", re.IGNORECASE),
        )
        _PATTERN_CACHE[tag] = cached
    return cached


def normalize_tags(tags: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Accept tags written as ``think``, ``<think>`` or ``</think>`` alike."""
    if tags is None:
        return DEFAULT_THINKING_TAGS
    cleaned = []
    for tag in tags:
        name = str(tag).strip().strip("<>").strip("/").strip()
        if name and name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned)


def strip_thinking(text: str, tags: Optional[Sequence[str]] = None) -> str:
    """Return ``text`` with reasoning blocks for ``tags`` removed.

    Three shapes are handled, because all three occur in practice:

    * A balanced ``<tag>...</tag>`` block -- removed.
    * A closing tag with no opener -- some chat templates pre-fill the opening
      tag, so it is part of the prompt and never echoed. Everything up to the
      last closer is narration.
    * An opening tag with no closer -- generation hit its token cap mid-thought,
      so there is no answer at all. Dropping the remainder yields an empty
      string, which a metric reports as a parse failure. That is the honest
      outcome: grading the scratchpad of a truncated response would score the
      model on answers it was still arguing against.

    Args:
        text: Raw model output.
        tags: Tag names to strip, with or without angle brackets. Defaults to
            :data:`DEFAULT_THINKING_TAGS`.
    """
    if not text:
        return ""

    for tag in normalize_tags(tags):
        block, opener, closer = _patterns(tag)
        text = block.sub("", text)

        if closer.search(text) and not opener.search(text):
            text = closer.split(text)[-1]

        start = opener.search(text)
        if start and not closer.search(text, start.end()):
            text = text[: start.start()]

    return text.strip()


__all__ = ["DEFAULT_THINKING_TAGS", "normalize_tags", "strip_thinking"]
