"""Result documents produced by an evaluate run.

:class:`BenchmarkResult` is the normalisation point: both benchmark tiers, and
any externally-authored benchmark, end up in this shape. It is the contract
reporters consume -- see ``docs/evaluate-extension-contract.md``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Reported as the score when a benchmark exceeded its failure tolerance. A
#: sentinel rather than 0.0, because "we could not measure this" and "the model
#: scored zero" must not average together.
INVALID_SCORE = -1.0


class SampleResult(BaseModel):
    """One scored record, for record-tier benchmarks.

    ``response`` and ``raw_response`` are populated only when
    ``evaluate.store_responses`` is set, because keeping every generation is
    expensive on a real run and pointless on a passing one. On a *failing* run
    they are the difference between a diagnosis and a guess: a score of zero
    can mean the model was wrong, that the answer was in a format the metric
    cannot read, or that reasoning-tag stripping removed the answer along with
    the narration -- and those three are indistinguishable from a number.
    ``raw_response`` is the text before stripping, and is set only when
    stripping changed it.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    model: str
    trial: int = 0
    record_id: str
    score: float = 0.0
    correct: bool = False
    parse_failed: bool = False
    error: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    response: str = ""
    raw_response: str = ""
    detail: Dict[str, Any] = Field(default_factory=dict)

    def resume_key(self) -> str:
        """Stable identifier used to skip already-completed work on resume."""
        return f"{self.benchmark}|{self.model}|{self.trial}|{self.record_id}"

    def to_flat_row(self) -> Dict[str, Any]:
        row = self.model_dump()
        detail = row.pop("detail", {})
        for key, value in detail.items():
            row[f"detail_{key}"] = value
        return row


class BenchmarkResult(BaseModel):
    """One ``(benchmark, model, trial)`` outcome.

    ``sub_scores`` is the extension point for benchmarks that report more than
    one number -- binary/fractional/rubric triples, per-category breakdowns, and
    so on. Nothing in the core interprets ``detail``; it round-trips to the
    report untouched.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    model: str
    trial: int = 0
    score: float = 0.0
    n_samples: int = 0
    n_errors: int = 0
    sub_scores: Dict[str, float] = Field(default_factory=dict)
    detail: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    @property
    def is_valid(self) -> bool:
        return self.score != INVALID_SCORE and not self.error

    @property
    def error_rate(self) -> float:
        total = self.n_samples + self.n_errors
        return (self.n_errors / total) if total else 0.0

    def resume_key(self) -> str:
        return f"{self.benchmark}|{self.model}|{self.trial}"

    def to_flat_row(self) -> Dict[str, Any]:
        row = self.model_dump()
        for key, value in row.pop("sub_scores", {}).items():
            row[f"sub_{key}"] = value
        row.pop("detail", None)
        return row


class AggregateScore(BaseModel):
    """A ``(benchmark, model)`` score aggregated across trials.

    With one trial this is just the score. With several it carries the
    t-distribution interval -- the honest summary for a benchmark whose
    scenarios are not deterministic.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    model: str
    trials: int = 1
    mean: float = 0.0
    stdev: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_level: float = 0.95
    scores: List[float] = Field(default_factory=list)
    sub_scores: Dict[str, float] = Field(default_factory=dict)
    n_samples: int = 0
    n_errors: int = 0
    cost_usd: float = 0.0
    valid: bool = True

    @property
    def ci_halfwidth(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


class ModelReportInfo(BaseModel):
    """How one candidate should be *presented*, as opposed to identified.

    Everything the runner keys on -- checkpoint rows, fingerprints, aggregate
    groups -- uses ``EvaluateModelConfig.name``, which for a locally served
    model is an S3 prefix. That is the correct identity and a terrible label: a
    leaderboard column headed by a 130-character URI is unreadable. Reporters
    render ``display_name`` and never the key, so presentation can change
    without invalidating a single checkpoint.

    ``leaderboard`` is free-form model metadata passed through to whichever sink
    wants it (today the Aleta reporter turns it into run tags). It is typed as
    ``str``-valued because tracking backends treat tags as strings.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str = ""
    leaderboard: Dict[str, str] = Field(default_factory=dict)


class EvaluateResult(BaseModel):
    """Top-level document written as ``evaluate_results.json``.

    Self-contained: carries the per-trial rows *and* the aggregated leaderboard,
    so the JSON alone reconstructs the run without reading the parquet.
    """

    model_config = ConfigDict(extra="forbid")

    job_name: str
    benchmarks: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    trials: int = 1
    started_at: str = ""
    finished_at: Optional[str] = None
    n_results: int = 0
    summary: Dict[str, Any] = Field(default_factory=dict)
    aggregates: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    model_info: Dict[str, ModelReportInfo] = Field(default_factory=dict)

    def label(self, model: str) -> str:
        """Return the display label for ``model``, falling back to its key."""
        info = self.model_info.get(model)
        return (info.display_name if info else "") or model


__all__ = [
    "INVALID_SCORE",
    "AggregateScore",
    "BenchmarkResult",
    "EvaluateResult",
    "ModelReportInfo",
    "SampleResult",
]
