"""Shared helpers for generation modes (ported from the reference).

Token budgets and the final-state heading-strip regex mirror the advice-core
orchestrator so candidate generation is faithful to how the reference produced
the ``state`` document.
"""

from __future__ import annotations

import re

# Token budgets per call (orchestrator SUMMARY/COMMENTARY/STATE_MAX_TOKENS).
SUMMARY_MAX_TOKENS = 16000
COMMENTARY_MAX_TOKENS = 2048
STATE_MAX_TOKENS = 16000

# Strip any "Recommendations / Next steps / Advice / Strategy" heading and
# everything after it. Ported verbatim from orchestrator._final_state_async.
_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s*"
    r"(recommendations?|next\s+steps"
    r"|action\s+items|strategy|strategies"
    r"|advice|advisory)\b.*$"
)


def strip_advisory_sections(text: str) -> str:
    """Remove advisory/recommendation sections from a final state report."""
    m = _HEADING_RE.search(text)
    if m:
        return text[: m.start()].rstrip() + "\n"
    return text


def cap_tokens(requested: int, model_max: int) -> int:
    """Clamp a per-call token budget to the model's configured maximum."""
    if model_max <= 0:
        return requested
    return min(requested, model_max)


__all__ = [
    "SUMMARY_MAX_TOKENS",
    "COMMENTARY_MAX_TOKENS",
    "STATE_MAX_TOKENS",
    "strip_advisory_sections",
    "cap_tokens",
]
