"""Contracts for the evaluate operation's pluggable axes.

Stdlib only, so the package imports with no heavy dependencies.

The important design decision here is the **two tiers**. Benchmarks differ in
who owns the loop, and forcing both shapes through one interface is what makes
harnesses like ``lm-eval`` or an agentic scenario suite awkward to integrate:

* :class:`RecordBenchmark` -- the runner owns the loop. You get one record and
  return one score. Concurrency, retries, cost accounting, and checkpointing
  are handled for you. This is right for prompt-in / answer-out benchmarks.

* :class:`HarnessBenchmark` -- the benchmark owns the loop. You get a prepared
  workspace and a live endpoint, and you return the aggregate. This is right
  for agentic suites, external repositories, and anything that grades
  filesystem artifacts rather than text.

A benchmark declares its tier simply by which protocol it implements; the
runner detects it. There is no flag to set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from foresightrlcore.inference.interfaces import ChatMessage, InferenceEndpoint

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRecord:
    """One evaluation unit.

    Args:
        id: Unique record id, stable across runs -- it is the checkpoint key,
            so a changing id defeats resume.
        inputs: Named inputs the benchmark reads (``question``, ``context``,
            ``scenario_dir``, ...).
        gold: The expected answer, in whatever shape the metric compares.
        choices: Answer options for multiple-choice records; empty otherwise.
        meta: Free-form metadata (``category``, ``difficulty``, ``split``, ...).
            ``category`` is what stratified sampling keys on by default.
    """

    id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    gold: Any = None
    choices: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRequest:
    """What a record-tier benchmark wants sent to the model."""

    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # Left as None to defer to the model config, which is where sampling
    # normally belongs: it describes the candidate, not the question. A
    # benchmark sets these only when the task itself dictates them.
    repetition_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, text: str, *, system: Optional[str] = None) -> "PromptRequest":
        """Build a single-turn request, optionally with a system message."""
        messages: List[ChatMessage] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        return cls(messages=messages)


@dataclass
class Prediction:
    """What came back from the model for one :class:`PromptRequest`.

    ``text`` is what metrics score. For a reasoning model it has had the
    scratchpad removed, so ``raw_text`` keeps the response as it arrived --
    without it, a run that strips the wrong tags looks like a model that
    answered nothing, and there is no way to tell the two apart afterwards.

    ``thinking_handled`` says the candidate configured ``thinking:`` and the
    runner has therefore already decided whether to strip. A metric with its
    own reasoning-tag safety net must stand down when it sees this, or a
    config that deliberately kept its reasoning tokens would find them
    stripped anyway.
    """

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    raw_text: Optional[str] = None
    thinking_handled: bool = False
    finish_reason: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)

    @property
    def original_text(self) -> str:
        """The response before any stripping; ``text`` when none happened."""
        return self.text if self.raw_text is None else self.raw_text

    @property
    def truncated(self) -> bool:
        """True when generation stopped at the token budget, not at an answer.

        A truncated completion usually fails to parse, which scores the same as
        a wrong answer. The two need telling apart: one is the model being
        wrong, the other is ``max_tokens`` being too low, and only the second
        is fixed by rerunning.
        """
        return self.finish_reason == "length"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


#: Suffix under which a benchmark records a non-primary metric's parse verdict
#: in ``SampleScore.detail``, e.g. ``ChoiceMatchMetric_parse_failed``. The
#: runner aggregates these into a per-metric valid-answer rate.
PARSE_FAILED_SUFFIX = "_parse_failed"


@dataclass
class SampleScore:
    """A metric's verdict on one prediction.

    ``parse_failed`` is deliberately separate from ``correct``: a model that
    answered in the wrong format is a different failure from one that answered
    incorrectly, and conflating them hides harness bugs behind low scores.
    """

    value: float = 0.0
    correct: bool = False
    parse_failed: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessContext:
    """Everything a harness-tier benchmark is handed for one trial.

    Args:
        benchmark: Benchmark name.
        endpoint: A live OpenAI-compatible endpoint for the model under test.
        workspace: Writable scratch directory, unique to this trial.
        output_file: Path the benchmark should write its result JSON to. The
            runner reads it when the benchmark returns nothing itself.
        records: Loaded dataset records, when a loader was configured.
        source_dir: Root of the staged benchmark source tree, if any.
        params: ``BenchmarkSpec.type_params()`` -- forwarded verbatim.
        env: Environment for any subprocess, already carrying the endpoint
            contract vars.
        trial: 0-based trial index.
        timeout_seconds: Wall-clock budget for this trial.
        agent_runtime: Set when the candidate is driven as an agent.
        logger: Logger scoped to this benchmark.
    """

    benchmark: str
    endpoint: InferenceEndpoint
    workspace: Path
    output_file: Path
    records: Sequence[EvalRecord] = ()
    source_dir: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    trial: int = 0
    timeout_seconds: int = 3600
    agent_runtime: Optional[Any] = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("foresightrlcore.evaluate")
    )


@dataclass
class HarnessOutcome:
    """The aggregate a harness-tier benchmark reports for one trial."""

    score: float = 0.0
    n_samples: int = 0
    n_errors: int = 0
    sub_scores: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Protocols (contracts)
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetLoader(Protocol):
    """Turns a dataset spec into :class:`EvalRecord` objects."""

    def records(
        self, spec: Dict[str, Any], sample: Optional[Dict[str, Any]] = None
    ) -> Iterable[EvalRecord]:
        """Return records for ``spec``.

        Args:
            spec: The dataset entry (``loader``, ``path``, ``limit``, plus any
                loader-specific keys).
            sample: Optional sampling config (``size``, ``seed``,
                ``stratify_by``).
        """
        ...


@runtime_checkable
class RecordBenchmark(Protocol):
    """Tier 1: the runner drives the loop, one record at a time."""

    def build_prompt(self, record: EvalRecord) -> PromptRequest:
        """Return the request to send for ``record``."""
        ...

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        """Score ``prediction`` against ``record``."""
        ...


@runtime_checkable
class HarnessBenchmark(Protocol):
    """Tier 2: the benchmark drives its own loop and returns the aggregate."""

    def run(self, ctx: HarnessContext) -> HarnessOutcome:
        """Run the whole benchmark once and return its aggregate."""
        ...


@runtime_checkable
class Metric(Protocol):
    """Scores one prediction against one record."""

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        """Return a :class:`SampleScore`."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Writes an evaluate result somewhere -- a file, a service, a console."""

    def report(self, result: Any, output_dir: Path) -> None:
        """Emit ``result`` (an ``EvaluateResult``) to ``output_dir``."""
        ...


def is_harness_benchmark(obj: object) -> bool:
    """True when ``obj`` drives its own loop (tier 2)."""
    return callable(getattr(obj, "run", None))


def is_record_benchmark(obj: object) -> bool:
    """True when ``obj`` is driven record-by-record (tier 1)."""
    return callable(getattr(obj, "build_prompt", None)) and callable(
        getattr(obj, "score", None)
    )


__all__ = [
    "PARSE_FAILED_SUFFIX",
    "DatasetLoader",
    "EvalRecord",
    "HarnessBenchmark",
    "HarnessContext",
    "HarnessOutcome",
    "Metric",
    "Prediction",
    "PromptRequest",
    "RecordBenchmark",
    "Reporter",
    "SampleScore",
    "is_harness_benchmark",
    "is_record_benchmark",
]
