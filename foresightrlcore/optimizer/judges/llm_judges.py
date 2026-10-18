"""LLM-as-judge judges built on ``resolve_llm_manager`` + structured output.

This replaces the reference's ``openevals`` + genos-specific ``ChatOpenAI`` with
the provider-agnostic foresight-rl-core client surface (bedrock **or** genos via
:func:`foresightrlcore.clients.llm.resolve_llm_manager`). The exact openevals
``CORRECTNESS_PROMPT`` / ``HALLUCINATION_PROMPT`` text is ported verbatim as
constants so judging is identical without the dependency.

The prompt is filled with ``str.format`` -- safe against literal ``{`` / ``}`` in
the substituted values because ``str.format`` only interprets braces in the
template, not in the arguments. A failed judge (after retries) yields score 1
with the error as reasoning, matching the reference error handling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.optimizer.interfaces import GenResult, JudgeScore, Record
from foresightrlcore.optimizer.registry import JUDGES

# Default judge model (reference DEFAULT_JUDGE_MODEL). Used only as documentation;
# the actual model is chosen by the configured judge_clients entry.
DEFAULT_JUDGE_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1-0"

# LLM judges (as opposed to deterministic ones). Kept as a module constant so
# the schema / validation can decide whether judge_clients is required.
LLM_JUDGE_NAMES = frozenset({"correctness", "hallucination"})


# --- Exact openevals prompt text (ported verbatim) -------------------------

CORRECTNESS_PROMPT = """You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

<Instructions>
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
</Instructions>

<Reminder>
  The goal is to evaluate factual correctness and completeness of the response.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Use the reference outputs below to help you evaluate the correctness of the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

HALLUCINATION_PROMPT = """You are an expert data labeler evaluating model outputs for hallucinations. Your task is to assign a score based on the following rubric:

<Rubric>
  A response without hallucinations:
  - Contains only verifiable facts that are directly supported by the input context
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains perfect accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
</Rubric>

<Instructions>
  - Read the input context thoroughly
  - Identify all claims made in the output
  - Cross-reference each claim with the input context
  - Note any unsupported or contradictory information
  - Consider the severity and quantity of hallucinations
</Instructions>

<Reminder>
  Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
</Reminder>

Use the following context to help you evaluate for hallucinations in the output:

<context>
{context}
</context>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

If available, you may also use the reference outputs below to help you identify hallucinations in the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""

_PROMPT_BY_JUDGE = {
    "correctness": CORRECTNESS_PROMPT,
    "hallucination": HALLUCINATION_PROMPT,
}


class EvaluationResult(BaseModel):
    """Structured judge output.

    NOTE: ``score`` deliberately has no ``ge``/``le`` constraint. Those emit
    JSON-schema ``minimum``/``maximum``, which Bedrock/GenOS structured output
    reject. The 1-5 range is conveyed via the description and enforced by
    clamping in :meth:`LlmJudge.score`.
    """

    score: int = Field(description="Score from 1 (worst) to 5 (best)")
    reasoning: str = Field(description="Brief explanation for the score")


class LlmJudge:
    """An LLM-as-judge wrapped to the optimizer ``Judge`` interface.

    Args:
        name: Judge name (``correctness`` / ``hallucination``).
        manager: An LLM client manager (bedrock/genos) exposing ``.get()`` and
            ``.force_refresh()``. Shared across judges that use the same client.
        threshold: Score (1-5) at/above which the row passes for this judge.
        max_retries: Retry budget per judge call.
        base_delay: Base backoff seconds.
    """

    def __init__(
        self,
        name: str,
        manager: Any,
        *,
        threshold: float = 3.0,
        max_retries: int = 10,
        base_delay: float = 3.0,
    ) -> None:
        if name not in _PROMPT_BY_JUDGE:
            raise ValueError(f"unknown LLM judge: {name!r}")
        if manager is None:
            raise ValueError(f"LLM judge {name!r} requires a client manager")
        self.name = name
        self._manager = manager
        self._prompt = _PROMPT_BY_JUDGE[name]
        self._threshold = float(threshold)
        self._max_retries = max_retries
        self._base_delay = base_delay

    def score(self, record: Record, gen: GenResult) -> JudgeScore:
        formatted = self._prompt.format(
            inputs=record.inputs.get("evidence_bundle", ""),
            outputs=gen.text or "",
            reference_outputs=record.reference or "",
            context="",
        )

        def _once() -> EvaluationResult:
            client = self._manager.get()
            structured = client.with_structured_output(EvaluationResult)
            result = structured.invoke([{"role": "user", "content": formatted}])
            if isinstance(result, EvaluationResult):
                return result
            if isinstance(result, dict):
                return EvaluationResult(
                    score=int(result.get("score", 1)),
                    reasoning=str(result.get("reasoning", "")),
                )
            return EvaluationResult(
                score=int(getattr(result, "score", 1)),
                reasoning=str(getattr(result, "reasoning", "")),
            )

        try:
            result = call_with_retry(
                _once,
                manager=self._manager,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                description=f"judge[{self.name}]",
            )
            score = max(1, min(5, int(result.score)))
            reasoning = result.reasoning
        except Exception as exc:  # noqa: BLE001 - mirror reference fallback
            score = 1
            reasoning = f"Evaluation failed: {type(exc).__name__}: {exc}"

        passed = score >= self._threshold
        return JudgeScore(
            name=self.name, score=score, passed=passed, reasoning=reasoning
        )


@JUDGES.register("correctness")
def _make_correctness(
    params: Optional[Dict[str, Any]] = None, *, manager: Any = None
) -> LlmJudge:
    params = params or {}
    return LlmJudge(
        "correctness", manager, threshold=float(params.get("threshold", 3.0))
    )


@JUDGES.register("hallucination")
def _make_hallucination(
    params: Optional[Dict[str, Any]] = None, *, manager: Any = None
) -> LlmJudge:
    params = params or {}
    return LlmJudge(
        "hallucination", manager, threshold=float(params.get("threshold", 3.0))
    )


__all__ = [
    "LlmJudge",
    "EvaluationResult",
    "CORRECTNESS_PROMPT",
    "HALLUCINATION_PROMPT",
    "DEFAULT_JUDGE_MODEL",
    "LLM_JUDGE_NAMES",
]
