"""Shared dataclasses and Protocols for the optimizer's pluggable axes.

These define the contracts every dataset / prompt set / generation mode / judge
must satisfy. Keeping them dependency-free (stdlib only) means the whole package
imports without network or heavy deps.

Ported from the bts-pipeline reference optimizer and generalized: the
``GenerationClient`` protocol is **synchronous** here (the foresight-rl-core LLM
client managers are sync + thread-pool, unlike the reference's ``async`` client).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

# Promoted to the generation layer so ``evaluate`` can reuse the clients without
# importing ``optimizer``; re-exported here so this module's surface is stable.
from foresightrlcore.generation.interfaces import CallUsage, GenerationClient

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One evaluation unit: inputs + the baseline reference output.

    Args:
        realm_id: Row identifier (company/realm id for BTS; any unique id for
            other use cases).
        epoch_date: As-of date label for the epoch (free-form string).
        inputs: Generation inputs keyed by name. Datasets/schemas populate
            whatever keys their modes read (e.g. ``evidence_bundle``,
            ``start_date``, ``end_date``).
        reference: Baseline reference text to compare candidates against (e.g.
            the Haiku ``state`` column for BTS). May be empty for use cases
            without a reference.
        meta: Free-form metadata (e.g. ``shard``, ``dataset`` name).
    """

    realm_id: str
    epoch_date: str
    inputs: Dict[str, Any]
    reference: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResult:
    """Output of a generation mode for one record x model.

    Args:
        text: Final generated text (post-processing applied).
        calls: Per-call usage records for cost/latency accounting.
        error: Error string if generation failed (text may be empty/partial).
    """

    text: str
    calls: List[CallUsage] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)


@dataclass
class JudgeScore:
    """A single judge's verdict for one generation result.

    The numeric scale follows the reference: judges emit an integer score in
    ``[1, 5]``. Deterministic judges map their boolean pass/fail onto the same
    scale (5 = pass, 1 = fail) so they aggregate uniformly with LLM judges.

    Args:
        name: Judge name (matches the registry key).
        score: Integer score in ``[1, 5]``.
        passed: Whether this judge considers the row acceptable on its own.
        reasoning: Short explanation (LLM judges) or detail (deterministic).
    """

    name: str
    score: int
    passed: bool
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Protocols (contracts)
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetProvider(Protocol):
    """Yields :class:`Record` objects for a configured dataset entry."""

    def records(self, spec: Dict[str, Any], sample: Dict[str, Any]) -> Iterable[Record]:
        """Return records.

        Args:
            spec: The dataset entry from config (``name``, ``path``, ``filter``,
                and any column-mapping keys).
            sample: Sampling config (``size``, ``seed``, ``stratify_by``).
        """
        ...


@runtime_checkable
class PromptSet(Protocol):
    """Renders prompt templates with placeholder substitution."""

    def render(self, section: str, key: str, **variables: Any) -> str:
        """Return the template at ``section.key`` with ``{var}`` substituted."""
        ...


@runtime_checkable
class GenerationMode(Protocol):
    """Defines a prompt-chain shape (which stages run) for one record x model."""

    def generate(
        self,
        record: Record,
        client: GenerationClient,
        prompt_set: PromptSet,
        *,
        max_tokens: int,
    ) -> GenResult:
        """Generate a candidate output for ``record`` using ``client``."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Scores a generation result against the record's reference."""

    def score(self, record: Record, gen: GenResult) -> JudgeScore:
        """Return a :class:`JudgeScore` for ``gen`` vs ``record.reference``."""
        ...


# Deterministic judges map boolean pass/fail onto the shared 1-5 scale.
DETERMINISTIC_PASS_SCORE = 5
DETERMINISTIC_FAIL_SCORE = 1


__all__ = [
    "Record",
    "CallUsage",
    "GenResult",
    "JudgeScore",
    "DatasetProvider",
    "PromptSet",
    "GenerationClient",
    "GenerationMode",
    "Judge",
    "DETERMINISTIC_PASS_SCORE",
    "DETERMINISTIC_FAIL_SCORE",
]
