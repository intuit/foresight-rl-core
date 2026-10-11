"""Evaluate operation -- validates config and dispatches to the backend.

Submits exactly one SageMaker training job (``aws_training_job``) or runs the
benchmarks in-process (``local``). Returns a single-element list for API
symmetry with the other operations.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

from foresightrlcore.config.evaluate_validation import validate_evaluate_config
from foresightrlcore.operations import register_operation

if TYPE_CHECKING:
    from foresightrlcore.config.evaluate_schema import EvaluateJobConfig

logger = logging.getLogger(__name__)


class EvaluateOperation:
    """Top-level operation for the ``evaluate`` flow."""

    name = "evaluate"

    def run(self, config: "EvaluateJobConfig", wait: bool = False) -> List[str]:
        """Validate the config and run/submit the evaluate job."""
        errors = validate_evaluate_config(config)
        if errors:
            raise ValueError(
                "Evaluate config validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        backend_type = config.execution.type
        logger.info("Evaluate execution backend: %s", backend_type)

        if backend_type == "aws_training_job":
            from foresightrlcore.execution.evaluate_launcher import (
                EvaluateSageMakerLauncher,
            )

            return [EvaluateSageMakerLauncher(config=config).launch(wait=wait)]

        if backend_type == "local":
            return self._run_local(config)

        raise ValueError(f"Evaluate does not support execution.type '{backend_type}'")

    def _run_local(self, config: "EvaluateJobConfig") -> List[str]:
        """Run the benchmarks in-process (dev box / GPU box).

        Exports the non-secret ``environment`` vars and resolves ``secrets:``
        first, so an endpoint that needs provider credentials can authenticate
        exactly as it would inside a job.
        """
        from pathlib import Path

        from foresightrlcore.evaluate.runner import run_evaluate

        for key, value in config.environment.items():
            os.environ.setdefault(key, value)

        if config.secrets is not None and config._source_path:
            from foresightrlcore.auth import resolve_secrets

            try:
                resolve_secrets(config._source_path)
            except Exception:  # noqa: BLE001 - surfaced clearly to the user
                logger.error("Local secret resolution failed", exc_info=True)
                raise

        local_cfg = config.execution.local or {}
        output_root = (
            local_cfg.get("output_dir")
            or config.output.s3_path
            or "/tmp/evaluate_output"
        )
        # An s3:// output path still writes locally first; the runner uploads.
        output_dir = (
            Path("/tmp/evaluate_output") / config.job.name
            if str(output_root).startswith("s3://")
            else Path(output_root)
        )
        checkpoint_dir = local_cfg.get("checkpoint_dir")

        run_evaluate(
            config,
            output_dir=output_dir,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else output_dir,
        )
        return [config.job.name]


register_operation("evaluate", EvaluateOperation)
