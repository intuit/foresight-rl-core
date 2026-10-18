"""Multiple-choice scoring.

Models answer multiple choice in every format imaginable -- a bare letter, a
bare index, a JSON object, a sentence quoting the option text. This metric
accepts the common ones and, crucially, distinguishes *unparseable* from
*wrong*: a harness that silently scores format failures as incorrect reports a
weak model when it actually has a prompt bug.

Gold answers are equally inconsistent: a 0-based index, a 1-based index, a
letter, or the option text. All four are understood.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from foresightrlcore.common.thinking import strip_thinking
from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction, SampleScore
from foresightrlcore.evaluate.metrics.text_match import normalize
from foresightrlcore.evaluate.parsing import extract_json_object
from foresightrlcore.evaluate.registry import METRICS

# "Answer: B", "**C**", "(d)", or a lone letter on its own line.
_LETTER_PATTERNS = (
    re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([a-z])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([a-z])\)?\s*[.):]?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\*\*\s*\(?([a-z])\)?\s*\*\*", re.IGNORECASE),
)
_JSON_KEYS = ("correct_answer", "answer", "choice", "selection", "label")

# `\boxed{B}` -- how the DeepSeek-R1 family states a final answer, inherited
# from the maths-reasoning data these models are distilled on. Without this a
# reasoning model's answer is unreadable and every record scores as a parse
# failure, which on a leaderboard is indistinguishable from a wrong answer.
# The inner alternation allows one level of nesting, for `\boxed{\text{B}}`.
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
# LaTeX dressing that turns up inside the box: `\text{B}`, `\mathbf{B}`, `$B$`.
_LATEX_WRAPPER = re.compile(r"\\(?:text|mathrm|mathbf|textbf|textit)\s*\{([^{}]*)\}")

# A label a model put in front of its answer: "D) ", "(d) ", "4. ", "2)".
# A separator is required so that an answer merely *starting* with a letter or
# a number ("10 years") is never mistaken for a labelled one.
_LABEL_PREFIX = re.compile(r"^\s*\(?([A-Za-z]|\d{1,2})\s*[.):\-]\s*")


def _letter_to_index(letter: str) -> int:
    return ord(letter.lower()) - ord("a")


def _extract_from_boxed(text: str) -> Optional[str]:
    """Pull the answer out of the last ``\\boxed{...}``, undressing any LaTeX.

    The *last* box, because a model that shows its working boxes intermediate
    results too, and the final answer is the one it stands behind.
    """
    matches = _BOXED.findall(text)
    if not matches:
        return None
    inner = matches[-1]
    inner = _LATEX_WRAPPER.sub(r"\1", inner)
    inner = inner.replace("$", "").replace("\\", "").strip()
    return inner or None


def _extract_from_json(text: str) -> Optional[str]:
    """Pull an answer out of a JSON object, including one inside a code fence."""
    parsed = extract_json_object(text)
    if parsed is None:
        return None
    for key in _JSON_KEYS:
        if parsed.get(key) is not None:
            return str(parsed[key])
    return None


def extract_choice(
    text: str, choices: List[str], *, strip_reasoning: bool = True
) -> Optional[str]:
    """Return the raw answer token a model selected, or ``None`` if unparseable.

    By default a reasoning model's scratchpad is removed first. A reasoning
    model is easy to evaluate without noticing it is one, and the failure is
    silent: every letter pattern here matches inside ``<think>the answer is A
    -- no, B</think>`` and returns the rejected answer. Stripping the two
    conventional tags costs a regex and needs no configuration.

    ``strip_reasoning=False`` turns that safety net off, and the runner passes
    it whenever the candidate configured ``thinking:`` either way -- because
    then the decision has already been made upstream. Stripping again would be
    merely wasteful if the answer is yes, but it would *override* a config that
    deliberately said no, and a metric quietly countermanding a config is worse
    than either behaviour on its own.

    Args:
        text: The model's response.
        choices: The option list, used to sanity-check a parsed letter.
        strip_reasoning: Whether to remove the conventional reasoning tags.
    """
    if not text or not text.strip():
        return None

    if strip_reasoning:
        text = strip_thinking(text)
    if not text:
        return None

    # Before the letter patterns: a boxed answer is explicit, whereas the
    # patterns below infer one from prose and can match a letter the model was
    # only discussing.
    boxed = _extract_from_boxed(text)
    if boxed is not None:
        return boxed

    from_json = _extract_from_json(text)
    if from_json is not None:
        return from_json

    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text)
        if match:
            letter = match.group(1)
            if not choices or _letter_to_index(letter) < len(choices):
                return letter

    # An exact quote of one option, when only one of them appears.
    normalized = normalize(text)
    hits = [c for c in choices if c and normalize(c) in normalized]
    if len(hits) == 1:
        return hits[0]

    stripped = text.strip()
    # isdecimal, not isdigit: circled and superscript digits pass isdigit but
    # int() rejects them, and a model does answer an ordering question "②①⑤".
    if stripped.isdecimal():
        return stripped
    return None


def resolve_gold_index(gold: Any, choices: List[str]) -> Optional[int]:
    """Normalise a gold answer to a 0-based choice index.

    A bare integer is ambiguous between 0-based and 1-based. It is read as
    0-based, except when that would fall outside the choice list and the
    1-based reading fits. A numeric *string* is tried as an index first and
    then as option text, so a benchmark whose options are themselves numbers
    still resolves.
    """
    if gold is None:
        return None

    if isinstance(gold, bool):
        return None
    if isinstance(gold, int):
        if 0 <= gold < len(choices):
            return gold
        if 1 <= gold <= len(choices):
            return gold - 1
        return None

    text = str(gold).strip()
    if text.isdecimal():
        # An integer is an index first -- that is the leaderboard convention.
        # Only when no index reading fits do we fall through and try to match
        # it as option *text*, which is what a numeric answer set looks like.
        index = resolve_gold_index(int(text), choices)
        if index is not None:
            return index
    elif len(text) == 1 and text.isalpha():
        index = _letter_to_index(text)
        return index if 0 <= index < len(choices) else None

    target = normalize(text)
    for index, choice in enumerate(choices):
        if normalize(choice) == target:
            return index

    # A prompt can ask for "the number or letter of the correct answer" while
    # rendering its options unlabelled -- several catalog benchmarks do, because
    # their data already carries markers for some rows. A model that complies
    # invents one and answers `D) <option text>`, which matches no option whole.
    # Take the text and discard the label: the text is the answer, the label is
    # decoration it was told to add. This runs only after a whole-string match
    # has failed, so options that carry their own markers still match on those.
    unlabelled = _LABEL_PREFIX.sub("", text, count=1)
    if unlabelled != text:
        stripped = normalize(unlabelled)
        for index, choice in enumerate(choices):
            if stripped and normalize(choice) == stripped:
                return index
    return None


class ChoiceMatchMetric:
    """Scores a multiple-choice answer, tracking parse failures separately."""

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        if prediction.failed:
            return SampleScore(detail={"error": prediction.error})

        choices = record.choices
        gold_index = resolve_gold_index(record.gold, choices)
        if gold_index is None:
            return SampleScore(
                parse_failed=True,
                detail={
                    "reason": "gold answer could not be resolved to a choice",
                    "gold": record.gold,
                },
            )

        raw = extract_choice(
            prediction.text,
            choices,
            strip_reasoning=not prediction.thinking_handled,
        )
        if raw is None:
            return SampleScore(
                parse_failed=True,
                detail={"reason": "no answer found in the response"},
            )

        predicted_index = resolve_gold_index(raw, choices)
        if predicted_index is None:
            return SampleScore(
                parse_failed=True,
                detail={"reason": "answer did not match any choice", "raw": raw},
            )

        correct = predicted_index == gold_index
        return SampleScore(
            value=1.0 if correct else 0.0,
            correct=correct,
            detail={
                "predicted_index": predicted_index,
                "gold_index": gold_index,
                "raw": raw,
            },
        )


@METRICS.register("choice_match")
def _make() -> ChoiceMatchMetric:
    return ChoiceMatchMetric()


__all__ = [
    "ChoiceMatchMetric",
    "extract_choice",
    "resolve_gold_index",
]
