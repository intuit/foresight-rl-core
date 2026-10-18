"""Scenario-graded benchmark (harness tier).

The in-package reference for the harness tier, and a useful benchmark in its
own right. A *scenario* asks for a structured result and is graded on several
independent checks, which is how agentic suites are scored in practice: not
"was the answer right" but "how many of the required facts came out right".

That distinction produces three numbers, all reported:

``binary``
    Fraction of scenarios where **every** check passed. The strict reading, and
    the headline score by default.
``fractional_micro``
    Passed checks / total checks across the suite. Weights big scenarios more.
``fractional_macro``
    Mean per-scenario pass fraction. Weights every scenario equally.

Each record's ``gold`` is a mapping of field name to expected value; the model
is asked for a JSON object and graded field by field. Records whose gold is a
scalar are graded as a single check named ``answer``.

Comparison is configurable per benchmark via ``grader``:

``normalized``
    Case- and whitespace-insensitive string equality (default).
``exact``
    Byte equality.
``numeric``
    Numeric equality within ``tolerance``, for money and quantities.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.evaluate.interfaces import (
    EvalRecord,
    HarnessContext,
    HarnessOutcome,
)
from foresightrlcore.evaluate.metrics.text_match import normalize
from foresightrlcore.evaluate.parsing import extract_json_object
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES
from foresightrlcore.inference.client import ChatClient

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Reply with a single JSON object holding exactly these fields: {fields}. "
    "Output only the JSON."
)

#: Per-scenario detail kept in the result. Capped so a large suite does not
#: turn its result document into the dataset.
MAX_SCENARIO_DETAIL = 200


def _grade_normalized(actual: Any, expected: Any, _tolerance: float) -> bool:
    return normalize(str(actual)) == normalize(str(expected))


def _grade_exact(actual: Any, expected: Any, _tolerance: float) -> bool:
    return str(actual) == str(expected)


def _grade_numeric(actual: Any, expected: Any, tolerance: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


GRADERS: Dict[str, Callable[[Any, Any, float], bool]] = {
    "normalized": _grade_normalized,
    "exact": _grade_exact,
    "numeric": _grade_numeric,
}


class ScenarioSuiteBenchmark:
    """Runs every scenario once and grades its checks.

    Args:
        grader: Key in :data:`GRADERS`.
        tolerance: Numeric grader tolerance.
        instruction: Appended to each scenario prompt; ``{fields}`` is filled
            with the expected field names.
        system_prompt: Optional system message.
        primary: Which sub-score becomes the headline ``score``.
        max_tokens: Output cap per scenario.
        concurrency: Scenarios evaluated in parallel.
        max_retries / base_delay: Retry policy for endpoint calls.
    """

    def __init__(
        self,
        *,
        grader: str = "normalized",
        tolerance: float = 0.01,
        instruction: str = DEFAULT_INSTRUCTION,
        system_prompt: Optional[str] = None,
        primary: str = "binary",
        max_tokens: int = 1024,
        concurrency: int = 4,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ) -> None:
        if grader not in GRADERS:
            raise ValueError(
                f"unknown grader {grader!r}; expected one of {sorted(GRADERS)}"
            )
        self.grader = grader
        self.tolerance = tolerance
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.primary = primary
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)
        self._max_retries = max_retries
        self._base_delay = base_delay

    # -- grading ----------------------------------------------------------

    @staticmethod
    def expected_fields(record: EvalRecord) -> Dict[str, Any]:
        """Return the record's checks as a field -> expected mapping."""
        if isinstance(record.gold, dict):
            return dict(record.gold)
        return {"answer": record.gold}

    def grade(
        self, record: EvalRecord, answer: Optional[Dict[str, Any]]
    ) -> Tuple[int, int, Dict[str, bool]]:
        """Return ``(passed, total, per-field verdicts)`` for one scenario.

        An unparseable answer fails every check rather than erroring: the model
        was asked for JSON and did not produce it, which is a real failure of
        the scenario, not of the harness.
        """
        expected = self.expected_fields(record)
        compare = GRADERS[self.grader]
        produced: Dict[str, Any] = answer or {}
        verdicts = {
            field: answer is not None
            and compare(produced.get(field), value, self.tolerance)
            for field, value in expected.items()
        }
        return sum(verdicts.values()), len(verdicts), verdicts

    # -- running ----------------------------------------------------------

    def _prompt(self, record: EvalRecord) -> str:
        question = record.inputs.get("question") or record.inputs.get("prompt") or ""
        fields = ", ".join(sorted(self.expected_fields(record)))
        return f"{question}\n\n{self.instruction.format(fields=fields)}"

    def _messages(self, record: EvalRecord) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prompt(record)})
        return messages

    def _run_one(self, client: ChatClient, record: EvalRecord) -> Dict[str, Any]:
        try:
            result = call_with_retry(
                lambda: client.chat(self._messages(record), max_tokens=self.max_tokens),
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                description=f"scenario {record.id}",
            )
        except Exception as exc:  # noqa: BLE001 - one bad scenario is not fatal
            logger.warning("scenario %s failed: %s", record.id, exc)
            return {"id": record.id, "error": f"{type(exc).__name__}: {exc}"}

        passed, total, verdicts = self.grade(record, extract_json_object(result.text))
        return {
            "id": record.id,
            "passed": passed,
            "total": total,
            "fields": verdicts,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    def run(self, ctx: HarnessContext) -> HarnessOutcome:
        records = list(ctx.records)
        if not records:
            return HarnessOutcome(error="scenario_suite benchmark got no records")

        client = ChatClient(ctx.endpoint, timeout=ctx.timeout_seconds)
        ctx.logger.info(
            "scenario_suite: %d scenarios, grader=%s", len(records), self.grader
        )

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            scenarios = list(
                pool.map(lambda record: self._run_one(client, record), records)
            )

        return self._aggregate(scenarios)

    def _aggregate(self, scenarios: List[Dict[str, Any]]) -> HarnessOutcome:
        graded = [s for s in scenarios if "error" not in s]
        errors = len(scenarios) - len(graded)

        if not graded:
            return HarnessOutcome(
                n_errors=errors,
                error="every scenario failed",
                detail={"scenarios": scenarios[:MAX_SCENARIO_DETAIL]},
            )

        checks_passed = sum(s["passed"] for s in graded)
        checks_total = sum(s["total"] for s in graded)
        sub_scores = {
            "binary": sum(
                1.0 for s in graded if s["total"] and s["passed"] == s["total"]
            )
            / len(graded),
            "fractional_micro": (checks_passed / checks_total) if checks_total else 0.0,
            "fractional_macro": sum(
                (s["passed"] / s["total"]) if s["total"] else 0.0 for s in graded
            )
            / len(graded),
        }

        return HarnessOutcome(
            score=sub_scores.get(self.primary, sub_scores["binary"]),
            n_samples=len(graded),
            n_errors=errors,
            sub_scores=sub_scores,
            detail={
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "scenarios": scenarios[:MAX_SCENARIO_DETAIL],
            },
            prompt_tokens=sum(s.get("prompt_tokens", 0) for s in graded),
            completion_tokens=sum(s.get("completion_tokens", 0) for s in graded),
        )


@BENCHMARK_TYPES.register("scenario_suite")
def _make(spec: Any = None, **_: Any) -> ScenarioSuiteBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    known = {
        "grader",
        "tolerance",
        "instruction",
        "system_prompt",
        "primary",
        "max_tokens",
        "concurrency",
    }
    return ScenarioSuiteBenchmark(**{k: v for k, v in params.items() if k in known})


__all__ = [
    "DEFAULT_INSTRUCTION",
    "GRADERS",
    "MAX_SCENARIO_DETAIL",
    "ScenarioSuiteBenchmark",
]
