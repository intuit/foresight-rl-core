"""Validation for :class:`OptimizerJobConfig`.

Mirrors :mod:`foresightrlcore.config.perf_validation` but tailored to the
optimizer operation: non-empty axes, LLM-judge client wiring, per-provider
requirements, per-backend path checks, and the GPU-only-when-serving-locally
infra sizing rule.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresightrlcore.config.optimizer_schema import OptimizerJobConfig

logger = logging.getLogger(__name__)

_VALID_BACKENDS = frozenset({"aws_training_job", "local"})
_GPU_INSTANCE_PREFIXES = ("ml.g", "ml.p")


def validate_optimizer_config(config: "OptimizerJobConfig") -> list[str]:
    """Run all optimizer validation checks. Returns a list of error strings."""
    errors: list[str] = []
    errors.extend(_validate_execution(config))
    errors.extend(_validate_output(config))
    errors.extend(_validate_models(config))
    errors.extend(_validate_judges(config))
    errors.extend(_validate_datasets(config))
    errors.extend(_validate_prompts(config))
    errors.extend(_validate_infra_sizing(config))
    errors.extend(_validate_secrets(config))
    return errors


def _validate_execution(config: "OptimizerJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type
    if exec_type not in _VALID_BACKENDS:
        errors.append(
            f"optimizer does not support execution.type '{exec_type}'. "
            f"Supported backends: {sorted(_VALID_BACKENDS)}"
        )
    return errors


def _validate_output(config: "OptimizerJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type

    if exec_type == "aws_training_job":
        output_path = str(config.output.s3_path) if config.output.s3_path else ""
        if not output_path:
            errors.append(
                "optimizer on aws_training_job requires output.s3_path "
                "(results are written there)."
            )
        elif not output_path.startswith("s3://"):
            errors.append(
                f"AWS optimizer requires an s3:// URI for output.s3_path, "
                f"got: {output_path!r}"
            )
        errors.extend(_validate_checkpointing(config))
    return errors


def _validate_checkpointing(config: "OptimizerJobConfig") -> list[str]:
    """Say out loud whether this run will be able to resume.

    An optimizer run is a long cross-product against paid endpoints, so
    whether its work survives an interruption deserves a line of output rather
    than being left for the operator to infer from a field they did not set.
    """
    if not config.optimizer.checkpoint.enabled:
        logger.warning(
            "optimizer.checkpoint.enabled is false: this run cannot resume. If "
            "it is interrupted or exceeds job.max_runtime_hours (%sh), every "
            "generation is lost and a re-run starts from the first row.",
            config.job.max_runtime_hours,
        )
        return []

    if config.output.checkpoint_s3_path:
        logger.info(
            "Resume enabled: checkpoints at %s", config.output.checkpoint_s3_path
        )
        return []

    logger.warning(
        "optimizer.checkpoint.enabled is true but no checkpoint location could "
        "be resolved, so checkpoints will be written to container disk and "
        "discarded when the job ends -- this run cannot resume. Set "
        "output.checkpoint_s3_path explicitly."
    )
    return []


def _validate_models(config: "OptimizerJobConfig") -> list[str]:
    errors: list[str] = []
    for m in config.optimizer.models:
        if m.provider == "local" and m.serving is None:
            errors.append(
                f"model '{m.name}': provider=local requires a 'serving' block "
                f"(vLLM parameters)."
            )
        if m.provider == "bedrock":
            import os

            if not m.region and not os.environ.get("AWS_DEFAULT_REGION"):
                logger.warning(
                    "model %r: provider=bedrock has no region and "
                    "AWS_DEFAULT_REGION is unset; the run will fail unless a "
                    "region is resolvable at runtime.",
                    m.name,
                )
    return errors


def _validate_judges(config: "OptimizerJobConfig") -> list[str]:
    errors: list[str] = []
    from foresightrlcore.optimizer.judges import is_llm_judge

    run = config.optimizer
    clients = run.judge_clients
    for j in run.judges:
        if is_llm_judge(j.name):
            if clients is None:
                errors.append(
                    f"judge '{j.name}' is an LLM judge but no "
                    f"optimizer.judge_clients mapping is configured."
                )
                continue
            try:
                clients.get(j.client)
            except ValueError:
                available = sorted(clients.root.keys())
                errors.append(
                    f"judge '{j.name}' references judge client "
                    f"'{j.client}' which is not in optimizer.judge_clients "
                    f"(available: {available})."
                )
    return errors


def _validate_datasets(config: "OptimizerJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type
    for ds in config.optimizer.datasets:
        path = ds.path
        if not path:
            errors.append(f"dataset '{ds.name}': a 'path' is required.")
            continue
        if exec_type == "aws_training_job" and not path.startswith("s3://"):
            errors.append(
                f"dataset '{ds.name}': aws_training_job requires an s3:// path "
                f"(mounted as an input channel), got: {path!r}."
            )
    return errors


def _prompt_set_ships_with_the_package(config: "OptimizerJobConfig") -> bool:
    """True when the prompt set's YAML lives inside a package staged into the job.

    The launcher copies ``foresightrlcore`` -- and each package named in
    ``container.extra_packages`` -- into the job's source directory. A prompt
    set whose YAML sits inside one of those package trees therefore arrives with
    the code and needs no upload.

    The test is deliberately against the *package* directories, not the repo
    root. In a source checkout everything is under the repo root, so a
    root-relative check would wrongly clear a prompt set whose YAML sits in an
    ``examples/`` directory next to a package rather than inside it -- present
    in the checkout, absent from the container.

    A name that cannot be resolved returns False, routing the user to the
    prompts_path error rather than crashing validation on an offline check.
    """
    import importlib
    from pathlib import Path

    import foresightrlcore
    from foresightrlcore.optimizer.prompts import make_prompt_set
    from foresightrlcore.plugins import load_optimizer_plugins

    load_optimizer_plugins()
    try:
        yaml_path = make_prompt_set(config.optimizer.prompt_set).path.resolve()
    except Exception:
        return False

    roots = [Path(foresightrlcore.__file__).resolve().parent]
    for extra in getattr(config.container, "extra_packages", []) or []:
        try:
            module = importlib.import_module(extra)
        except ImportError:
            continue
        roots.extend(Path(p).resolve() for p in getattr(module, "__path__", []))

    for root in roots:
        try:
            yaml_path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _validate_prompts(config: "OptimizerJobConfig") -> list[str]:
    """On aws_training_job a prompts YAML outside the package must come from S3.

    The launcher ships the installed packages into the container, not the repo,
    so a prompt_set that resolves to a path under an ``examples/`` directory
    does not exist there. Requiring an s3:// prompts_path keeps those prompts in
    S3 next to the dataset (no bundling, no code deploy).

    Prompt sets packaged inside ``foresightrlcore`` -- or inside an extension
    listed in ``container.extra_packages`` -- are exempt: the YAML travels with
    the code, and demanding an upload for a file already in the image would be
    busywork that also makes the built-in sets unusable remotely.
    """
    errors: list[str] = []
    if config.execution.type != "aws_training_job":
        return errors
    prompts_path = config.optimizer.prompts_path
    if not prompts_path:
        if _prompt_set_ships_with_the_package(config):
            return errors
        errors.append(
            f"optimizer on aws_training_job requires optimizer.prompts_path to "
            f"be an s3:// URI: prompt_set "
            f"{config.optimizer.prompt_set!r} is not packaged inside "
            f"foresightrlcore, so its YAML is not present in the container. "
            f"Upload it to S3 - e.g. next to the dataset - and point "
            f"prompts_path at it."
        )
    elif not prompts_path.startswith("s3://"):
        errors.append(
            f"optimizer on aws_training_job requires an s3:// prompts_path, got: "
            f"{prompts_path!r}."
        )
    return errors


def _validate_infra_sizing(config: "OptimizerJobConfig") -> list[str]:
    """GPU only when serving locally; warn on wasteful GPU for API-only."""
    errors: list[str] = []
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
    else:
        if is_gpu:
            logger.warning(
                "instance_type '%s' is a GPU instance but no candidate serves "
                "locally (all genos/bedrock). This wastes GPU cost - a CPU "
                "instance (e.g. ml.m5.2xlarge) is sufficient for API-only runs.",
                instance_type,
            )
    return errors


def _validate_secrets(config: "OptimizerJobConfig") -> list[str]:
    """Warn if any plain env var name looks like a secret; require a VPC for IDPS."""
    errors: list[str] = []
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

    # IDPS secret resolution pip-installs idps-sdk from the Intuit internal
    # PyPI at container runtime; that requires the job to run inside the Intuit
    # VPC. Without VpcConfig the install times out and the job fails.
    if (
        config.execution.type == "aws_training_job"
        and config.secrets is not None
        and config.secrets.provider == "idps"
        and config.optimizer.vpc is None
    ):
        errors.append(
            "secrets.provider=idps on aws_training_job requires optimizer.vpc "
            "(subnets + security_groups) so the container can reach the Intuit "
            "internal PyPI to install idps-sdk. Add an optimizer.vpc block."
        )
    return errors


__all__ = ["validate_optimizer_config"]
