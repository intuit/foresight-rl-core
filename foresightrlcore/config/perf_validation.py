"""Validation for :class:`PerfTestJobConfig`.

Mirrors :mod:`foresightrlcore.config.validation` but tailored to the
perf-test operation -- there are no method/training fields, the execution
backend is constrained to ``aws_training_job`` or ``local``, and the
container must be a perf-test image (DJL inference) rather than a
pytorch-training DLC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresightrlcore.config.perf_schema import PerfTestJobConfig

logger = logging.getLogger(__name__)


_VALID_BACKENDS = frozenset({"aws_training_job", "local"})


def validate_perf_config(config: "PerfTestJobConfig") -> list[str]:
    """Run all perf-test validation checks. Returns a list of error strings."""
    errors: list[str] = []
    errors.extend(_validate_execution(config))
    errors.extend(_validate_output(config))
    errors.extend(_validate_container(config))
    errors.extend(_validate_run(config))
    errors.extend(_validate_secrets(config))
    return errors


def _validate_execution(config: "PerfTestJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type

    if exec_type not in _VALID_BACKENDS:
        errors.append(
            f"perf_test does not support execution.type '{exec_type}'. "
            f"Supported backends: {sorted(_VALID_BACKENDS)}"
        )
        return errors

    if exec_type == "local":
        local_cfg = config.execution.local or {}
        if not local_cfg.get("gpus"):
            errors.append(
                "Local backend requires execution.local.gpus to specify GPU IDs."
            )

    return errors


def _validate_output(config: "PerfTestJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type

    if exec_type == "aws_training_job":
        output_path = str(config.output.s3_path) if config.output.s3_path else ""
        if not output_path:
            errors.append(
                "perf_test on aws_training_job requires output.s3_path to be set "
                "(perf-test results are written there)."
            )
        elif not output_path.startswith("s3://"):
            errors.append(
                f"AWS perf_test requires an s3:// URI for output.s3_path, "
                f"got: {output_path!r}"
            )

    return errors


def _validate_container(config: "PerfTestJobConfig") -> list[str]:
    errors: list[str] = []
    c = config.container

    if c.recipe is None and c.image_uri is None and c.perf_test_image_uri is None:
        errors.append(
            "perf_test requires one of: container.recipe (resolves to a "
            "DJL inference image), container.perf_test_image_uri, or "
            "container.image_uri. None were set."
        )
        return errors

    if c.recipe is not None and c.perf_test_image_uri is None and c.image_uri is None:
        # When the recipe is set, the perf_test_image_uri should have been
        # resolved by recipes.resolve_container_recipe(). If it is still
        # missing, the recipe does not declare a perf-test variant.
        from foresightrlcore.config.recipes import CONTAINER_RECIPES

        recipe = CONTAINER_RECIPES.get(c.recipe)
        if recipe is not None and "perf_test_image_uri" not in recipe:
            errors.append(
                f"container.recipe '{c.recipe}' does not declare a "
                f"perf_test_image_uri. Either pick a recipe that supports "
                f"perf testing (qwen3, qwen3_5) or set "
                f"container.perf_test_image_uri / container.image_uri "
                f"explicitly."
            )

    return errors


def _validate_run(config: "PerfTestJobConfig") -> list[str]:
    errors: list[str] = []
    run = config.perf_test
    serving = run.serving
    gatling = run.gatling

    if serving.quantization is not None:
        allowed = {
            "awq",
            "gptq",
            "squeezellm",
            "marlin",
            "fp8",
            "bitsandbytes",
            "compressed-tensors",
        }
        if serving.quantization not in allowed:
            errors.append(
                f"run '{run.name}': unsupported "
                f"serving.quantization='{serving.quantization}'. "
                f"Must be one of: {sorted(allowed)}"
            )

    if (
        serving.speculative is not None
        and serving.speculative.method == "eagle3"
        and serving.speculative.model is None
    ):
        errors.append(
            f"run '{run.name}': speculative method 'eagle3' requires "
            f"serving.speculative.model to be set (draft model path)."
        )

    if not gatling.simulation_class:
        errors.append(f"run '{run.name}': gatling.simulation_class must not be empty.")

    if gatling.tps_levels and serving.max_num_seqs > 0:
        max_tps = max(gatling.tps_levels)
        if max_tps * 100 > serving.max_num_seqs * 10:
            # Not a hard error -- just a warning. The standard simulation
            # uses concurrent_users = tps * 100, which is fine in practice
            # because Gatling's throttle caps the actual RPS regardless.
            logger.warning(
                "run %r: max(tps_levels)=%d * 100 concurrent users may "
                "exceed serving.max_num_seqs=%d, which can cause request "
                "queuing. Tune max_num_seqs if you observe latency cliffs.",
                run.name,
                max_tps,
                serving.max_num_seqs,
            )

    return errors


def _validate_secrets(config: "PerfTestJobConfig") -> list[str]:
    """Mirror the secrets check from validate_config: warn if any plain env
    var name looks like a secret."""
    errors: list[str] = []

    secret_keywords = {"SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "API_KEY"}

    for key in config.environment:
        upper = key.upper()
        if any(kw in upper for kw in secret_keywords):
            logger.warning(
                "environment.%s looks like a secret. Move it to the "
                "secrets: section so it is resolved at runtime instead of "
                "baked into the SageMaker job metadata.",
                key,
            )

    # IDPS secret resolution pip-installs idps-sdk from the Intuit internal
    # PyPI at container runtime, which is only reachable from inside the VPC.
    # Mirrors the same rule in optimizer_validation.
    if (
        config.execution.type == "aws_training_job"
        and config.secrets is not None
        and config.secrets.provider == "idps"
        and config.perf_test.vpc is None
    ):
        errors.append(
            "secrets.provider=idps on aws_training_job requires perf_test.vpc "
            "(subnets + security_groups) so the container can reach the Intuit "
            "internal PyPI to install idps-sdk. Add a perf_test.vpc block."
        )

    return errors


__all__ = ["validate_perf_config"]
