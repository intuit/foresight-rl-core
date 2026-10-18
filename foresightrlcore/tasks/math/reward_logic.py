"""Math final-answer accuracy scoring (stdlib + ``math_verify`` only).

Single source of truth for the DeepMath-style accuracy reward. Mirrors
TRL's ``accuracy_reward`` from the GRPO trainer docs
(https://huggingface.co/docs/trl/en/grpo_trainer): parse the model's
final answer and the ground-truth answer with ``math_verify`` and check
mathematical equivalence.

Kept transport-free and training-stack-free (no ``torch`` / ``trl``) so
it can be unit-tested locally and reused anywhere. The only third-party
dependency is ``math_verify`` (bundled in the qwen3 container recipe).
"""

from __future__ import annotations

import logging

from foresightrlcore.tasks.thinking_utils import strip_thinking_tags

logger = logging.getLogger(__name__)


def accuracy_score(completion_text: str, ground_truth: str) -> float:
    """Return ``1.0`` if the completion's final answer matches ``ground_truth``.

    Follows TRL's ``accuracy_reward`` recipe:

    1. Strip ``<think>…</think>`` reasoning (reused
       :func:`foresightrlcore.tasks.thinking_utils.strip_thinking_tags`).
    2. Parse both the prediction and the gold answer with
       ``math_verify.parse`` (understands ``\\boxed{}`` and LaTeX) and
       compare with ``math_verify.verify``.
    3. Fallback for non-mathematical answers (e.g. ``"Yes"``): when either
       side fails to parse to a math expression, compare normalized
       strings. This is a small robustness improvement over vanilla TRL,
       which scores such rows ``0``.

    Any parse/verify error is swallowed and scored ``0.0`` so a single bad
    completion never crashes the reward batch.
    """
    from math_verify import parse, verify

    prediction = strip_thinking_tags(completion_text or "")
    gold = (ground_truth or "").strip()

    try:
        gold_parsed = parse(gold)
        pred_parsed = parse(prediction)
        # math_verify.parse returns an empty list when nothing math-like is
        # found. When the gold answer is non-mathematical (plain text like
        # "Yes"), fall back to a normalized string comparison so those rows
        # can still earn reward.
        if gold_parsed and pred_parsed:
            return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception as exc:  # noqa: BLE001 - math_verify may raise on odd input
        logger.debug("math_verify failed, falling back to string match: %s", exc)

    return 1.0 if _normalize(prediction) == _normalize(gold) and gold else 0.0


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for the non-math string fallback."""
    return " ".join(text.lower().split())
