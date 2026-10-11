"""Generic JSONL converter (line-oriented loop used by ``rl-train serverless convert``).

Resolves the user's ``--converter`` (registry name or dotted path) or
``--mapping`` (YAML field-rename spec) to a callable, then iterates the
input JSONL, runs the callable per line, validates the result against
the requested ``--schema`` pydantic model, and writes one JSONL line per
output record.

Per-line errors (missing fields, schema-validation failures, mapper
exceptions) are caught and logged with a one-line reason; the loop
continues so a bad sample does not kill the whole conversion.

The mapper itself lives in :mod:`foresightrlcore.tasks.<task>.convert`
or — for the no-code path — is synthesized from a YAML mapping spec.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Type

from pydantic import BaseModel, ValidationError

from foresightrlcore.serverless import registry
from foresightrlcore.serverless.spec import Mapper, get_schema

logger = logging.getLogger(__name__)


@dataclass
class ConvertSummary:
    """Per-file conversion stats."""

    schema: str
    total_in: int = 0
    total_out: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mapper resolution
# ---------------------------------------------------------------------------


def resolve_mapper(
    converter: Optional[str] = None,
    mapping: Optional[str] = None,
    schema: str = "rlvr",
) -> Mapper:
    """Resolve ``--converter`` or ``--mapping`` to a callable mapper.

    Mutually exclusive — passing both raises ``ValueError``.
    """
    if converter and mapping:
        raise ValueError("Specify exactly one of --converter or --mapping (not both).")
    if not converter and not mapping:
        raise ValueError(
            "Either --converter (registry name or 'pkg.module:fn' dotted "
            "path) or --mapping (path to a YAML spec) is required."
        )

    if converter:
        mapper, supported = registry.resolve_converter(converter)
        if schema not in supported:
            raise ValueError(
                f"Converter {converter!r} supports schemas {sorted(supported)!r}, "
                f"but --schema={schema!r} was requested."
            )
        return mapper

    # mapping path
    assert mapping is not None  # for type checker
    return _build_yaml_mapper(mapping, schema=schema)


# ---------------------------------------------------------------------------
# YAML mapping spec (no-code path)
# ---------------------------------------------------------------------------


def _build_yaml_mapper(yaml_path: str, schema: str) -> Mapper:
    """Build a mapper from a simple YAML field-rename spec.

    Example YAML::

        schema: rlvr
        data_source: mailchimp
        ability: classification
        reward_style: rule
        prompt_from: prompt              # source key for the prompt field
        ground_truth_from: outcome       # source key for reward_model.ground_truth
        extra_info_passthrough:          # source keys to copy to extra_info
          - trace_id
          - task
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Using --mapping requires PyYAML. Install with `uv add pyyaml`."
        ) from exc

    spec = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(spec, dict):
        raise ValueError(
            f"YAML mapping at {yaml_path} must be a dict, got {type(spec)}"
        )

    declared_schema = spec.get("schema", "rlvr")
    if declared_schema != schema:
        raise ValueError(
            f"YAML mapping declares schema={declared_schema!r} but the "
            f"CLI requested --schema={schema!r}; the two must match."
        )

    if schema != "rlvr":
        raise NotImplementedError(
            "YAML --mapping currently only supports schema=rlvr. "
            f"For schema={schema!r}, write a Python mapper instead."
        )

    data_source = spec.get("data_source")
    if not data_source:
        raise ValueError("YAML mapping must set 'data_source'")
    ability = spec.get("ability", "generation")
    reward_style = spec.get("reward_style", "model")
    prompt_from = spec.get("prompt_from", "prompt")
    ground_truth_from = spec.get("ground_truth_from")
    if not ground_truth_from:
        raise ValueError("YAML mapping must set 'ground_truth_from'")
    passthrough: List[str] = list(spec.get("extra_info_passthrough", []))

    from foresightrlcore.serverless.spec import RewardModel, RLVRRecord

    def _mapper(record: Dict[str, Any]) -> Optional[RLVRRecord]:
        prompt = record.get(prompt_from)
        gt = record.get(ground_truth_from)
        if not prompt or not isinstance(gt, str) or not gt.strip():
            return None
        extra = {k: record[k] for k in passthrough if k in record}
        return RLVRRecord(
            data_source=data_source,
            prompt=(
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            ),
            ability=ability,
            reward_model=RewardModel(ground_truth=gt, style=reward_style),
            extra_info=extra if extra else None,
        )

    return _mapper


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def convert_file(
    input_path: str,
    output_path: str,
    mapper: Mapper,
    schema: str = "rlvr",
    max_errors_to_log: int = 5,
) -> ConvertSummary:
    """Run a JSONL conversion end-to-end."""
    model_cls = get_schema(schema)
    summary = ConvertSummary(schema=schema)

    pass_schema = _mapper_takes_schema_kwarg(mapper)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(input_path, "r", encoding="utf-8") as src,
        open(output_path, "w", encoding="utf-8") as dst,
    ):
        for line_no, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue
            summary.total_in += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _record_skip(summary, line_no, f"bad JSON: {exc}", max_errors_to_log)
                continue
            try:
                result = (
                    mapper(record, schema=schema) if pass_schema else mapper(record)
                )
            except Exception as exc:  # noqa: BLE001 — mappers can raise anything
                _record_skip(
                    summary,
                    line_no,
                    f"mapper raised {type(exc).__name__}: {exc}",
                    max_errors_to_log,
                )
                continue

            for produced in _normalize_output(result):
                if produced is None:
                    summary.skipped += 1
                    continue
                try:
                    validated = _coerce_to_model(produced, model_cls)
                except ValidationError as exc:
                    _record_skip(
                        summary,
                        line_no,
                        f"schema validation failed: {exc.errors()[:1]}",
                        max_errors_to_log,
                    )
                    continue
                except ValueError as exc:
                    _record_skip(
                        summary,
                        line_no,
                        f"schema validation failed: {exc}",
                        max_errors_to_log,
                    )
                    continue
                dst.write(validated.model_dump_json(exclude_none=True) + "\n")
                summary.total_out += 1

    _log_summary(summary, input_path, output_path)
    return summary


def _record_skip(
    summary: ConvertSummary, line_no: int, reason: str, max_errors_to_log: int
) -> None:
    summary.skipped += 1
    if len(summary.errors) < max_errors_to_log:
        summary.errors.append(f"line {line_no}: {reason}")


def _normalize_output(result: Any) -> Iterator[Any]:
    """Yield items from a mapper return — supports single, None, or iterable."""
    if result is None:
        yield None
        return
    if isinstance(result, BaseModel):
        yield result
        return
    if isinstance(result, dict):
        yield result
        return
    if hasattr(result, "__iter__"):
        yield from result
        return
    yield result


def _coerce_to_model(produced: Any, model_cls: Type[BaseModel]) -> BaseModel:
    if isinstance(produced, model_cls):
        return produced
    if isinstance(produced, BaseModel):
        raise ValueError(
            f"Mapper produced a {type(produced).__name__}, but "
            f"--schema={model_cls.__name__} was requested."
        )
    return model_cls.model_validate(produced)


def _mapper_takes_schema_kwarg(mapper: Callable) -> bool:
    try:
        sig = inspect.signature(mapper)
    except (TypeError, ValueError):
        return False
    return "schema" in sig.parameters


def _log_summary(summary: ConvertSummary, input_path: str, output_path: str) -> None:
    logger.info(
        "Convert %s -> %s (schema=%s): %d in, %d out, %d skipped",
        input_path,
        output_path,
        summary.schema,
        summary.total_in,
        summary.total_out,
        summary.skipped,
    )
    for err in summary.errors:
        logger.info("  %s", err)
    if summary.skipped > len(summary.errors):
        logger.info(
            "  ... and %d more (raise log level for full detail)",
            summary.skipped - len(summary.errors),
        )


__all__ = ["ConvertSummary", "convert_file", "resolve_mapper"]
