"""Shared helpers for building SageMaker ``CreateTrainingJob`` requests.

Several launchers submit raw boto3 ``CreateTrainingJob`` requests
(``GRPOJobLauncher`` fallback, ``PerfTestSageMakerLauncher``,
``OptimizerSageMakerLauncher``). This module centralises settings that
must be applied consistently across every SageMaker training job so a
new launcher cannot silently drop a compliance-relevant field.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import os
import shutil
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence

logger = logging.getLogger(__name__)

ENCRYPTION_KWARG = "enable_inter_container_traffic_encryption"

#: Env var read by ``foresightrlcore.plugins.load_plugins`` inside the job to
#: import staged extension packages. The container has no installed metadata
#: for them, so entry-point discovery cannot find them on its own.
EXTRA_PACKAGES_ENV = "FORESIGHT_EXTRA_PACKAGES"

#: Patterns excluded from every source-staging copy. ``__pycache__`` is not
#: merely noise: CPython writes bytecode by creating ``<name>.pyc.<id>`` and
#: renaming it into place, so a concurrent interpreter compiling the same tree
#: makes ``copytree`` see a directory entry that is gone by the time it is
#: read. Walking the directory at all is therefore a race, and excluding it is
#: the only way to close it.
STAGE_IGNORE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo", ".git")

#: Ready-made ``copytree(ignore=...)`` callable for :data:`STAGE_IGNORE_PATTERNS`.
STAGE_IGNORE = shutil.ignore_patterns(*STAGE_IGNORE_PATTERNS)


def apply_inter_container_encryption(request: Dict[str, Any], enabled: bool) -> None:
    """Set in-transit (inter-container) traffic encryption on a request.

    Maps to SageMaker ``EnableInterContainerTrafficEncryption`` on a raw
    boto3 ``CreateTrainingJob`` request. Required by Intuit ACP policy
    ``IS031.1`` (``AwsSagemakerTrainingjob`` must be encrypted in-transit).

    The flag only affects multi-instance (distributed) jobs, but it is
    harmless for single-instance jobs and keeps them compliant. It does
    not require a VPC.

    Args:
        request: The ``CreateTrainingJob`` request dict, mutated in place.
        enabled: Whether to enable inter-container traffic encryption.
    """
    request["EnableInterContainerTrafficEncryption"] = enabled
    logger.info("Inter-container traffic encryption: %s", enabled)


def stage_extra_packages(source_dir: str, package_names: Sequence[str]) -> List[str]:
    """Copy extension packages into ``source_dir`` so the job can import them.

    Extension packages (e.g. ``foresightrl_intuit``) live outside the OSS
    core tree and are not pip-installed inside the container, so the
    launcher ships their source next to ``foresightrlcore``.

    Args:
        source_dir: Staging directory uploaded as ``sourcedir.tar.gz``.
        package_names: Importable package names to stage.

    Returns:
        The package names successfully staged, in input order.

    Raises:
        ValueError: If a requested package is not importable locally, or is
            not a regular (directory-backed) package.
    """
    staged: List[str] = []
    for name in package_names:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(
                f"extra_packages: cannot stage {name!r} — package not importable "
                f"in the launching environment. Install it first "
                f"(e.g. 'uv pip install -e intuit/')."
            )
        src = list(spec.submodule_search_locations)[0]
        shutil.copytree(
            src,
            os.path.join(source_dir, name),
            ignore=STAGE_IGNORE,
            dirs_exist_ok=True,
        )
        logger.info("Staged extension package %s from %s", name, src)
        staged.append(name)
    return staged


def extra_packages_env(package_names: Sequence[str]) -> Dict[str, str]:
    """Container env telling the runtime which staged packages to import."""
    if not package_names:
        return {}
    return {EXTRA_PACKAGES_ENV: ",".join(package_names)}


@contextmanager
def serverless_inter_container_encryption(enabled: bool) -> Iterator[None]:
    """Force in-transit encryption on SDK v3 serverless training jobs.

    SageMaker SDK v3's serverless trainers (``DPOTrainer``,
    ``RLVRTrainer``, ``RLAIFTrainer``) build their
    ``TrainingJob.create(...)`` arguments from a hardcoded dict inside
    ``train()`` and expose no constructor parameter for
    ``enable_inter_container_traffic_encryption`` — their ``**kwargs``
    are consumed by ``BaseTrainer``. Wrapping ``TrainingJob.create`` for
    the duration of the submit call is therefore the only way to set the
    flag required by Intuit ACP policy ``IS031.1``.

    When ``enabled`` is ``False`` nothing is patched, so opting out
    restores stock SDK behaviour and the parameter is never sent.

    Raises:
        RuntimeError: If the installed SDK's ``TrainingJob.create`` no
            longer accepts the parameter. Failing fast keeps an SDK
            upgrade from silently dropping the compliance flag.
    """
    if not enabled:
        logger.info("Inter-container traffic encryption (serverless): disabled")
        yield
        return

    from sagemaker.core.resources import TrainingJob

    bound_create = TrainingJob.create
    if ENCRYPTION_KWARG not in inspect.signature(bound_create).parameters:
        raise RuntimeError(
            f"sagemaker.core.resources.TrainingJob.create no longer accepts "
            f"{ENCRYPTION_KWARG!r}; cannot enforce in-transit encryption "
            f"(ACP IS031.1) on serverless jobs. Pin a compatible sagemaker "
            f"SDK, or set execution.serverless."
            f"encrypt_inter_container_traffic: false to opt out."
        )

    @functools.wraps(bound_create)
    def _create_with_encryption(*args: Any, **kwargs: Any) -> Any:
        kwargs[ENCRYPTION_KWARG] = True
        return bound_create(*args, **kwargs)

    original_attr = inspect.getattr_static(TrainingJob, "create")
    TrainingJob.create = _create_with_encryption
    logger.info("Inter-container traffic encryption (serverless): enabled")
    try:
        yield
    finally:
        TrainingJob.create = original_attr


__all__ = [
    "EXTRA_PACKAGES_ENV",
    "apply_inter_container_encryption",
    "extra_packages_env",
    "serverless_inter_container_encryption",
    "stage_extra_packages",
]
