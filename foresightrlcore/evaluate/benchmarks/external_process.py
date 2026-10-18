"""Run a benchmark that lives outside this package (harness tier).

This is the type that makes externally-authored benchmarks -- an agentic
scenario suite, a repository with its own runner, a hub adapter -- work without
being rewritten. The contract is deliberately small:

1. The core stages the benchmark's source, prepares a workspace, and brings up
   an OpenAI-compatible endpoint.
2. It runs ``command`` with the contract environment variables set.
3. The command writes JSON to ``$EVALUATE_OUTPUT_FILE`` and exits 0.

Accepted result JSON::

    {
      "score": 0.72,                       # required
      "n_samples": 155,
      "n_errors": 3,
      "sub_scores": {"binary": 0.61, "fractional": 0.83},
      "detail": {...}                      # preserved verbatim
    }

A bare number is accepted too, for a benchmark whose runner emits only a score.

Argument and environment values may reference the contract variables (``$VAR``
or ``${VAR}``); they are expanded before the process starts, so a benchmark's
own CLI flags can carry the workspace or output path.
"""

from __future__ import annotations

import json
import logging
import os
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from foresightrlcore.evaluate.interfaces import HarnessContext, HarnessOutcome
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

logger = logging.getLogger(__name__)

# Bytes of captured output kept on failure. Enough to diagnose, small enough
# that a chatty benchmark cannot bloat the result document.
OUTPUT_TAIL_CHARS = 4000


def expand(value: str, env: Dict[str, str]) -> str:
    """Expand ``$VAR`` / ``${VAR}`` in ``value`` from ``env``.

    Unknown names are left as-is rather than blanked, so a typo shows up in the
    failing command line instead of silently becoming an empty argument.
    """
    return string.Template(value).safe_substitute(env)


class ExternalProcessBenchmark:
    """Runs an external command and reads back its result JSON.

    Args:
        command: argv to execute.
        env: Extra environment layered over the contract variables.
        cwd: Working directory, relative to the staged source when relative.
            Defaults to the source tree, or the workspace when there is none.
        python: Interpreter substituted for a leading ``python`` in ``command``.
            Defaults to ``$EVALUATE_PYTHON``, which the runner sets to the
            benchmark's isolated interpreter when ``isolate: venv`` is used.
    """

    def __init__(
        self,
        command: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        python: Optional[str] = None,
    ) -> None:
        if not command:
            raise ValueError("external_process benchmark requires a non-empty command")
        self.command = list(command)
        self.env = dict(env or {})
        self.cwd = cwd
        self.python = python

    def _resolve_command(self, env: Dict[str, str]) -> List[str]:
        argv = [expand(arg, env) for arg in self.command]
        python = self.python or env.get("EVALUATE_PYTHON")
        if python and argv[0] in ("python", "python3"):
            argv[0] = python
        return argv

    def _resolve_cwd(self, ctx: HarnessContext) -> Path:
        base = ctx.source_dir or ctx.workspace
        if not self.cwd:
            return base
        candidate = Path(self.cwd)
        return candidate if candidate.is_absolute() else base / candidate

    def run(self, ctx: HarnessContext) -> HarnessOutcome:
        env = {
            **os.environ,
            **ctx.env,
            **{k: expand(v, ctx.env) for k, v in self.env.items()},
        }
        argv = self._resolve_command(env)
        cwd = self._resolve_cwd(ctx)

        ctx.logger.info(
            "benchmark %s trial %d: running %s (cwd=%s)",
            ctx.benchmark,
            ctx.trial,
            " ".join(argv),
            cwd,
        )

        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=ctx.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return HarnessOutcome(
                error=f"benchmark timed out after {ctx.timeout_seconds}s"
            )
        except (OSError, ValueError) as exc:
            return HarnessOutcome(error=f"could not start benchmark: {exc}")

        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-OUTPUT_TAIL_CHARS:]
            return HarnessOutcome(
                error=f"benchmark exited with code {completed.returncode}",
                detail={"stderr_tail": tail},
            )

        return read_outcome(ctx.output_file, stdout=completed.stdout)


def read_outcome(output_file: Path, *, stdout: str = "") -> HarnessOutcome:
    """Parse a benchmark's result file, falling back to its stdout.

    Falling back matters in practice: plenty of runners print their summary and
    never learned about an output path.
    """
    raw: Optional[str] = None
    if output_file.exists():
        raw = output_file.read_text(encoding="utf-8").strip()
    if not raw and stdout.strip():
        raw = _last_json_object(stdout)
    if not raw:
        return HarnessOutcome(
            error=f"benchmark produced no result at {output_file} and no JSON on stdout"
        )

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return HarnessOutcome(error=f"benchmark result is not valid JSON: {exc}")

    if isinstance(payload, (int, float)):
        return HarnessOutcome(score=float(payload), n_samples=1)
    if not isinstance(payload, dict):
        return HarnessOutcome(
            error=f"benchmark result must be an object or a number, "
            f"got {type(payload).__name__}"
        )
    if "score" not in payload:
        return HarnessOutcome(error="benchmark result JSON has no 'score' field")

    try:
        score = float(payload["score"])
    except (TypeError, ValueError):
        return HarnessOutcome(
            error=f"benchmark 'score' is not a number: {payload['score']!r}"
        )

    sub_scores = {
        str(k): float(v)
        for k, v in (payload.get("sub_scores") or {}).items()
        if isinstance(v, (int, float))
    }
    return HarnessOutcome(
        score=score,
        n_samples=int(payload.get("n_samples", 0) or 0),
        n_errors=int(payload.get("n_errors", 0) or 0),
        sub_scores=sub_scores,
        detail=payload.get("detail") or {},
        prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
        completion_tokens=int(payload.get("completion_tokens", 0) or 0),
    )


def _last_json_object(text: str) -> Optional[str]:
    """Return the last brace-delimited JSON object in ``text``, if any."""
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for index in range(end, -1, -1):
        char = text[index]
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return text[index : end + 1]
    return None


@BENCHMARK_TYPES.register("external_process")
def _make(spec: Any = None, **_: Any) -> ExternalProcessBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    return ExternalProcessBenchmark(
        list(getattr(spec, "command", []) or []),
        env=dict(getattr(spec, "env", {}) or {}),
        cwd=params.get("cwd"),
    )


__all__ = [
    "ExternalProcessBenchmark",
    "OUTPUT_TAIL_CHARS",
    "expand",
    "read_outcome",
]
