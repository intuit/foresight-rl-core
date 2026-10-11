"""Multiple-choice benchmark type (record tier).

The workhorse: render the question and its options, ask for one answer, score
it. Everything about the rendering is a parameter, because leaderboard
benchmarks each have their own prompt conventions and the difference is worth
several points of measured accuracy.

Parameters (from ``params`` or top-level keys on the benchmark spec):

``instruction``
    Text appended after the choices. Defaults to a letter-answer instruction.
``system_prompt``
    Optional system message. Omitted entirely when unset, since some models
    behave differently with an empty system turn than with none.
``choice_labels``
    Label alphabet. Defaults to ``ABCDEFGH``.
``label_choices``
    Whether to prefix each option with a label. Set ``false`` for datasets
    whose option text already carries its own marker, which would otherwise be
    rendered twice (``A. A. ...``).
``number_of_choices``
    Truncate the option list to the first N.
``template``
    Full override, formatted with ``{question}``, ``{choices}``,
    ``{instruction}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from foresightrlcore.evaluate.interfaces import (
    PARSE_FAILED_SUFFIX,
    EvalRecord,
    Metric,
    Prediction,
    PromptRequest,
    SampleScore,
)
from foresightrlcore.evaluate.metrics.choice_match import ChoiceMatchMetric
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

logger = logging.getLogger(__name__)

DEFAULT_CHOICE_LABELS = "ABCDEFGH"
DEFAULT_INSTRUCTION = (
    "Answer with the letter of the correct choice only, and nothing else."
)
DEFAULT_TEMPLATE = "{question}\n\n{choices}\n\n{instruction}"


class McqBenchmark:
    """Renders a multiple-choice question and scores the selected option."""

    def __init__(
        self,
        *,
        metrics: Optional[Sequence[Metric]] = None,
        instruction: str = DEFAULT_INSTRUCTION,
        system_prompt: Optional[str] = None,
        choice_labels: str = DEFAULT_CHOICE_LABELS,
        label_choices: bool = True,
        number_of_choices: Optional[int] = None,
        template: str = DEFAULT_TEMPLATE,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.metrics: List[Metric] = list(metrics or [ChoiceMatchMetric()])
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.choice_labels = choice_labels
        self.label_choices = label_choices
        self.number_of_choices = number_of_choices
        self.template = template
        self.max_tokens = max_tokens

    def _visible_choices(self, record: EvalRecord) -> List[str]:
        choices = list(record.choices)
        if self.number_of_choices is not None:
            choices = choices[: self.number_of_choices]
        return choices

    def render_choices(self, record: EvalRecord) -> str:
        """Return the option block, one per line, labelled ``A.``/``B.``/..."""
        visible = self._visible_choices(record)
        if not self.label_choices:
            return "\n".join(visible)
        lines = []
        for index, choice in enumerate(visible):
            label = (
                self.choice_labels[index]
                if index < len(self.choice_labels)
                else str(index + 1)
            )
            lines.append(f"{label}. {choice}")
        return "\n".join(lines)

    def build_prompt(self, record: EvalRecord) -> PromptRequest:
        text = self.template.format(
            question=record.inputs.get("question", ""),
            choices=self.render_choices(record),
            instruction=self.instruction,
        )
        request = PromptRequest.user(text, system=self.system_prompt)
        if self.max_tokens is not None:
            request.max_tokens = self.max_tokens
        return request

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        """Score with the first metric; extra metrics land in ``detail``.

        The first metric is the benchmark's score. Additional ones are recorded
        alongside it so a run can report several views of the same answer
        without changing what the leaderboard number means.

        Each extra metric contributes its parse verdict as well as its value.
        Keeping only the value loses the one thing the extra metric is uniquely
        able to say: the reported score for the leaderboard suite comes from a
        containment rule that scores an unreadable answer 0 rather than
        flagging it, so on its own it reports no parse failures however many
        there were.
        """
        primary = self.metrics[0].score(record, prediction)
        for extra in self.metrics[1:]:
            verdict = extra.score(record, prediction)
            name = type(extra).__name__
            primary.detail[name] = verdict.value
            primary.detail[f"{name}{PARSE_FAILED_SUFFIX}"] = verdict.parse_failed
        return primary


@BENCHMARK_TYPES.register("mcq")
def _make(
    spec: Any = None, metrics: Optional[Sequence[Metric]] = None, **_: Any
) -> McqBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    known = {
        "instruction",
        "system_prompt",
        "choice_labels",
        "label_choices",
        "number_of_choices",
        "template",
        "max_tokens",
    }
    return McqBenchmark(
        metrics=metrics, **{k: v for k, v in params.items() if k in known}
    )


__all__ = [
    "DEFAULT_CHOICE_LABELS",
    "DEFAULT_INSTRUCTION",
    "DEFAULT_TEMPLATE",
    "McqBenchmark",
]
