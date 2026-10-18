"""Validation for :class:`EvaluateJobConfig`.

Mirrors :mod:`foresightrlcore.config.optimizer_validation`: catch every
misconfiguration locally, before a job is submitted, because the cheapest
failure is the one that happens on a laptop.

Checks that matter most here are the ones a benchmark author gets wrong:
naming a benchmark that is not in the catalog, giving a harness-tier benchmark
no ``command``, pinning nothing in ``source``, or asking for GPU-backed local
serving on a CPU instance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from foresightrlcore.config.evaluate_schema import (
        BenchmarkSpec,
        EvaluateJobConfig,
    )

logger = logging.getLogger(__name__)

_VALID_BACKENDS = frozenset({"aws_training_job", "local"})
_GPU_INSTANCE_PREFIXES = ("ml.g", "ml.p")


def validate_evaluate_config(config: "EvaluateJobConfig") -> List[str]:
    """Run all evaluate validation checks. Returns a list of error strings."""
    errors: List[str] = []
    errors.extend(_validate_execution(config))
    errors.extend(_validate_output(config))
    errors.extend(_validate_models(config))
    errors.extend(_validate_benchmarks(config))
    errors.extend(_validate_reporters(config))
    errors.extend(_validate_infra_sizing(config))
    errors.extend(_validate_secrets(config))
    return errors


def _validate_execution(config: "EvaluateJobConfig") -> List[str]:
    exec_type = config.execution.type
    if exec_type not in _VALID_BACKENDS:
        return [
            f"evaluate does not support execution.type '{exec_type}'. "
            f"Supported backends: {sorted(_VALID_BACKENDS)}"
        ]
    return []


def _validate_output(config: "EvaluateJobConfig") -> List[str]:
    errors: List[str] = []
    if config.execution.type != "aws_training_job":
        return errors

    output_path = str(config.output.s3_path) if config.output.s3_path else ""
    if not output_path:
        errors.append(
            "evaluate on aws_training_job requires output.s3_path "
            "(results are written there)."
        )
    elif not output_path.startswith("s3://"):
        errors.append(
            f"AWS evaluate requires an s3:// URI for output.s3_path, "
            f"got: {output_path!r}"
        )

    errors.extend(_validate_checkpointing(config))
    return errors


def _validate_checkpointing(config: "EvaluateJobConfig") -> List[str]:
    """Say out loud whether this run will be able to resume.

    Resume is worth a line of output either way. An evaluate job runs for
    hours against paid endpoints, so whether its work survives an interruption
    is not a detail, and the previous silence on the subject is what let a
    13-hour run reach its end with nothing to resume from.
    """
    if not config.evaluate.checkpoint.enabled:
        logger.warning(
            "evaluate.checkpoint.enabled is false: this run cannot resume. If "
            "it is interrupted or exceeds job.max_runtime_hours (%sh), every "
            "generation is lost and a re-run starts from the first record.",
            config.job.max_runtime_hours,
        )
        return []

    if config.output.checkpoint_s3_path:
        logger.info(
            "Resume enabled: checkpoints at %s", config.output.checkpoint_s3_path
        )
        return []

    # Checkpointing is on but has nowhere durable to go. The schema derives a
    # path from output.s3_path, so reaching here means it could not -- and the
    # runner would still write checkpoints, to container disk that does not
    # outlive the job.
    logger.warning(
        "evaluate.checkpoint.enabled is true but no checkpoint location could "
        "be resolved, so checkpoints will be written to container disk and "
        "discarded when the job ends -- this run cannot resume. Set "
        "output.checkpoint_s3_path explicitly."
    )
    return []


def _validate_models(config: "EvaluateJobConfig") -> List[str]:
    import os

    errors: List[str] = []
    seen: set[str] = set()
    for m in config.evaluate.models:
        if m.name in seen:
            errors.append(f"duplicate model name: {m.name!r}")
        seen.add(m.name)

        if m.provider == "local" and m.serving is None:
            errors.append(
                f"model '{m.name}': provider=local requires a 'serving' block "
                f"(vLLM parameters)."
            )
        if m.provider == "bedrock" and not m.region:
            if not os.environ.get("AWS_DEFAULT_REGION"):
                logger.warning(
                    "model %r: provider=bedrock has no region and "
                    "AWS_DEFAULT_REGION is unset; the run will fail unless a "
                    "region is resolvable at runtime.",
                    m.name,
                )
        if m.provider == "echo":
            logger.warning(
                "model %r uses provider=echo: responses are synthetic and "
                "scores are meaningless. Use it for plumbing checks only.",
                m.name,
            )
        if (
            m.provider == "bedrock"
            and m.top_p is not None
            and "temperature" in m.model_fields_set
        ):
            logger.warning(
                "model %r sets both 'temperature' and 'top_p'. Anthropic "
                "models on Bedrock reject a request carrying both and every "
                "call will fail; drop one. Other Bedrock families accept it.",
                m.name,
            )
        if m.thinking_tags and m.is_agentic:
            logger.warning(
                "model %r sets both 'thinking' and 'agent_runtime'; the "
                "thinking tags are ignored. An agent owns its own loop and "
                "reports its own result, so nothing passes through the "
                "runner's scoring path where stripping happens.",
                m.name,
            )

    # Model groups must reference declared models.
    known = {m.name for m in config.evaluate.models}
    for group, members in config.evaluate.model_groups.items():
        missing = sorted(set(members) - known)
        if missing:
            errors.append(
                f"model_group '{group}' references unknown model(s) {missing}; "
                f"known models: {sorted(known)}"
            )
    return errors


def _validate_benchmarks(config: "EvaluateJobConfig") -> List[str]:
    """Resolve every benchmark against the catalog and check its shape."""
    from foresightrlcore.evaluate.catalog import load_catalog
    from foresightrlcore.evaluate.registry import BENCHMARK_TYPES
    from foresightrlcore.plugins import load_evaluate_plugins

    load_evaluate_plugins()

    errors: List[str] = []
    run = config.evaluate

    try:
        catalog = load_catalog(run.catalog_paths)
    except Exception as exc:  # noqa: BLE001 - surfaced as a config error
        return [f"failed to load the benchmark catalog: {exc}"]

    seen: set[str] = set()
    for entry in run.benchmarks:
        if entry.name in seen:
            errors.append(f"duplicate benchmark name: {entry.name!r}")
        seen.add(entry.name)

        # A benchmark is either fully defined inline (it names a type) or must
        # exist in the catalog.
        if entry.name not in catalog and entry.type is None:
            errors.append(
                f"benchmark {entry.name!r} is not in the catalog and defines no "
                f"'type'. Known catalog benchmarks: {sorted(catalog)}"
            )
            continue

        try:
            spec = (
                entry.merged_over(catalog[entry.name])
                if entry.name in catalog
                else entry
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a config error
            errors.append(f"benchmark {entry.name!r}: catalog merge failed: {exc}")
            continue

        errors.extend(_validate_benchmark_spec(config, spec, BENCHMARK_TYPES))

        try:
            run.resolve_models_for(entry)
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def _is_harness_type(
    spec: "BenchmarkSpec", type_name: str, benchmark_types: object
) -> bool:
    """Whether ``spec``'s type drives its own loop.

    The tier is a property of the constructed object rather than a declared
    flag, so validation settles it the same way the runner does. Anything that
    goes wrong here -- an unknown type, a factory that rejects this spec -- is
    reported elsewhere with more context, so it is treated as "no complaint".
    """
    from foresightrlcore.evaluate.interfaces import is_harness_benchmark

    if type_name not in benchmark_types:  # type: ignore[operator]
        return True
    try:
        factory = benchmark_types.get(type_name)  # type: ignore[attr-defined]
        return is_harness_benchmark(factory(spec=spec, metrics=[]))
    except Exception:  # noqa: BLE001 - not this check's failure to report
        return True


def _validate_benchmark_spec(
    config: "EvaluateJobConfig", spec: "BenchmarkSpec", benchmark_types: object
) -> List[str]:
    from foresightrlcore.evaluate.registry import METRICS

    errors: List[str] = []
    name = spec.name
    exec_type = config.execution.type

    type_name = spec.type or "mcq"
    if type_name not in benchmark_types:  # type: ignore[operator]
        known = benchmark_types.list()  # type: ignore[attr-defined]
        errors.append(
            f"benchmark {name!r}: unknown type {type_name!r}. Registered "
            f"types: {known}"
        )

    for metric in spec.metrics:
        if metric not in METRICS:
            errors.append(
                f"benchmark {name!r}: unknown metric {metric!r}. Registered "
                f"metrics: {METRICS.list()}"
            )

    # Harness-tier benchmarks that shell out need something to run.
    if type_name == "external_process" and not spec.command:
        errors.append(
            f"benchmark {name!r}: type=external_process requires a 'command' "
            f"(argv) to run."
        )

    # `isolate: venv` hands a benchmark its own interpreter -- as
    # `$EVALUATE_PYTHON`, and as the substitution for a leading `python` in
    # `command`. Both reach harness-tier benchmarks only, so asking for one
    # anywhere else pays for a venv nothing will ever use. A `plugin` harness
    # that shells out is as entitled to one as `external_process` is, which is
    # why this asks about the tier rather than about `command`.
    if spec.isolate == "venv" and not _is_harness_type(
        spec, type_name, benchmark_types
    ):
        errors.append(
            f"benchmark {name!r}: isolate=venv applies to harness-tier "
            f"benchmarks, which receive the interpreter as $EVALUATE_PYTHON. "
            f"Type {type_name!r} is scored record by record and never sees it."
        )

    if spec.source is not None:
        if spec.source.git and not spec.source.ref:
            logger.warning(
                "benchmark %r pins no source.ref; results will not be "
                "reproducible if the branch moves.",
                name,
            )
        if exec_type == "aws_training_job" and spec.source.path:
            errors.append(
                f"benchmark {name!r}: source.path is a local directory and is "
                f"not available inside a SageMaker job. Use source.git or "
                f"source.s3."
            )

    # Dataset paths must be reachable from wherever the job runs.
    if spec.dataset is not None:
        try:
            path = spec.dataset.resolved_path()
        except ValueError as exc:
            errors.append(f"benchmark {name!r}: {exc}")
            path = None
        if path and exec_type == "aws_training_job" and not path.startswith("s3://"):
            errors.append(
                f"benchmark {name!r}: aws_training_job requires an s3:// dataset "
                f"path, got: {path!r}."
            )

    if spec.prompts_path and exec_type == "aws_training_job":
        if not spec.prompts_path.startswith("s3://"):
            errors.append(
                f"benchmark {name!r}: aws_training_job requires an s3:// "
                f"prompts_path, got: {spec.prompts_path!r}."
            )

    # Judge metrics need a client mapping.
    if _uses_judge_metric(spec) and config.evaluate.judge_clients is None:
        errors.append(
            f"benchmark {name!r} uses a judge metric but no "
            f"evaluate.judge_clients mapping is configured."
        )

    return errors


def _uses_judge_metric(spec: "BenchmarkSpec") -> bool:
    return any("judge" in metric for metric in spec.metrics)


def _validate_reporters(config: "EvaluateJobConfig") -> List[str]:
    from foresightrlcore.evaluate.registry import REPORTERS
    from foresightrlcore.plugins import load_evaluate_plugins

    load_evaluate_plugins()

    errors: List[str] = []
    for spec in config.evaluate.reporters:
        if spec.type not in REPORTERS:
            errors.append(
                f"unknown reporter {spec.type!r}. Registered reporters: "
                f"{REPORTERS.list()}"
            )
    return errors


def _validate_infra_sizing(config: "EvaluateJobConfig") -> List[str]:
    """GPU only when serving locally; warn on wasteful GPU for API-only."""
    errors: List[str] = []
    if config.execution.type != "aws_training_job":
        return errors

    instance_type = config.resolved_instance_type()
    is_gpu = instance_type.startswith(_GPU_INSTANCE_PREFIXES)

    if config.requires_gpu():
        if not is_gpu:
            errors.append(
                f"a candidate uses provider=local (vLLM serving) which needs a "
                f"GPU instance, but resolved instance_type '{instance_type}' is "
                f"not a GPU family (ml.g*/ml.p*)."
            )
        c = config.container
        if c.recipe is None and c.image_uri is None and c.perf_test_image_uri is None:
            errors.append(
                "local serving requires a vLLM-capable GPU image: set "
                "container.recipe (e.g. qwen3_5), container.perf_test_image_uri, "
                "or container.image_uri."
            )
    elif is_gpu:
        logger.warning(
            "instance_type '%s' is a GPU instance but no candidate serves "
            "locally. This wastes GPU cost - a CPU instance (e.g. "
            "ml.m5.2xlarge) is sufficient for API-only runs.",
            instance_type,
        )
    return errors


def _validate_secrets(config: "EvaluateJobConfig") -> List[str]:
    """Warn if any plain env var looks like a secret; require a VPC for IDPS."""
    errors: List[str] = []
    secret_keywords = {"SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "API_KEY"}
    for key in config.environment:
        upper = key.upper()
        if any(kw in upper for kw in secret_keywords):
            logger.warning(
                "environment.%s looks like a secret. Move it to the secrets: "
                "section so it is resolved at runtime instead of baked into the "
                "SageMaker job metadata.",
                key,
            )

    if (
        config.execution.type == "aws_training_job"
        and config.secrets is not None
        and config.secrets.provider == "idps"
        and config.evaluate.vpc is None
    ):
        errors.append(
            "secrets.provider=idps on aws_training_job requires evaluate.vpc "
            "(subnets + security_groups) so the container can reach the Intuit "
            "internal PyPI to install idps-sdk. Add an evaluate.vpc block."
        )
    return errors


__all__ = ["validate_evaluate_config"]
