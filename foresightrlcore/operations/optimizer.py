"""Optimizer operation -- validates config and dispatches to the backend.

Each invocation submits exactly one SageMaker training job (``aws_training_job``)
or runs the evaluation in-process (``local``). Returns a single-element list for
API symmetry with the other operations.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

from foresightrlcore.config.optimizer_validation import validate_optimizer_config
from foresightrlcore.operations import register_operation

if TYPE_CHECKING:
    from foresightrlcore.config.optimizer_schema import OptimizerJobConfig

logger = logging.getLogger(__name__)


class OptimizerOperation:
    """Top-level operation for the ``optimizer`` flow."""

    name = "optimizer"

    def run(self, config: "OptimizerJobConfig", wait: bool = False) -> List[str]:
        """Validate the config and run/submit the optimizer job."""
        errors = validate_optimizer_config(config)
        if errors:
            msg = "Optimizer config validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(msg)

        backend_type = config.execution.type
        logger.info("Optimizer execution backend: %s", backend_type)

        if backend_type == "aws_training_job":
            from foresightrlcore.execution.optimizer_launcher import (
                OptimizerSageMakerLauncher,
            )

            launcher = OptimizerSageMakerLauncher(config=config)
            return [launcher.launch(wait=wait)]

        if backend_type == "local":
            return self._run_local(config)

        raise ValueError(f"Optimizer does not support execution.type '{backend_type}'")

    def _run_local(self, config: "OptimizerJobConfig") -> List[str]:
        """Execute the optimizer via the in-process runner (dev / GPU box).

        Resolves secrets (if a ``secrets:`` section is present) and exports the
        non-secret ``environment`` vars so genos/bedrock auth works locally, then
        runs :class:`OptimizerRunner`. The returned identifier is the job name.
        """
        from pathlib import Path

        from foresightrlcore.optimizer.runner import run_optimizer

        # Export non-secret environment vars (e.g. GENOS_* non-secret values).
        for key, value in config.environment.items():
            os.environ.setdefault(key, value)

        # Resolve secrets from the configured provider, if any.
        if config.secrets is not None and config._source_path:
            from foresightrlcore.auth import resolve_secrets

            try:
                resolve_secrets(config._source_path)
            except Exception:  # noqa: BLE001 - surfaced clearly to the user
                logger.error("Local secret resolution failed", exc_info=True)
                raise

        local_cfg = config.execution.local or {}
        output_root = Path(
            local_cfg.get("output_dir")
            or config.output.s3_path
            or "/tmp/optimizer_output"
        )
        checkpoint_dir = local_cfg.get("checkpoint_dir")

        # For a local S3 output path we still write locally, then the runner
        # uploads. Use a local staging dir when output is an s3:// URI.
        if str(output_root).startswith("s3://"):
            output_dir = Path("/tmp/optimizer_output") / config.job.name
        else:
            output_dir = Path(output_root)

        run_optimizer(
            config,
            output_dir=output_dir,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else output_dir,
        )
        return [config.job.name]


register_operation("optimizer", OptimizerOperation)
