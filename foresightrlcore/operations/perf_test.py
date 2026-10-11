"""Perf-test operation -- validates config and dispatches to the execution backend.

Each invocation submits exactly one SageMaker training job (or one local run).
The operation returns a single-element list for API symmetry with the training
operation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from foresightrlcore.config.perf_validation import validate_perf_config
from foresightrlcore.operations import register_operation

if TYPE_CHECKING:
    from foresightrlcore.config.perf_schema import PerfTestJobConfig

logger = logging.getLogger(__name__)


class PerfTestOperation:
    """Top-level operation for the ``perf_test`` flow."""

    name = "perf_test"

    def run(self, config: "PerfTestJobConfig", wait: bool = False) -> List[str]:
        """Validate the config and submit the perf-test job.

        Returns a single-element list containing the backend-specific job
        identifier.
        """
        errors = validate_perf_config(config)
        if errors:
            msg = "Perf-test config validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(msg)

        backend_type = config.execution.type
        logger.info("Perf-test execution backend: %s", backend_type)

        if backend_type == "aws_training_job":
            from foresightrlcore.execution.perf_test_launcher import (
                PerfTestSageMakerLauncher,
            )

            launcher = PerfTestSageMakerLauncher(config=config)
            return [launcher.launch(wait=wait)]

        if backend_type == "local":
            return self._run_local(config)

        raise ValueError(f"Perf-test does not support execution.type '{backend_type}'")

    def _run_local(self, config: "PerfTestJobConfig") -> List[str]:
        """Execute the run via the in-process runner.

        Used on developer GPU boxes where the model and prompts live on the
        local disk. The job identifier returned is the run name -- there is
        no external job system to track.
        """
        import os
        from pathlib import Path

        from foresightrlcore.perf_test.runner import PerfTestRunner

        local_cfg = config.execution.local or {}
        output_root = Path(
            local_cfg.get("output_dir")
            or config.output.s3_path
            or "/tmp/perf_test_output"
        )

        data_root = local_cfg.get("data_dir")
        if data_root is None:
            raise ValueError(
                "Local perf-test backend requires execution.local.data_dir "
                "(local directory mirroring gatling.data_s3_path)"
            )

        run = config.perf_test
        run_output = output_root / run.name
        run_output.mkdir(parents=True, exist_ok=True)
        os.makedirs(run_output, exist_ok=True)
        runner = PerfTestRunner(
            run,
            model_path=config.model.name,
            data_channel=str(data_root),
            output_dir=run_output,
        )
        runner.execute()
        return [run.name]


register_operation("perf_test", PerfTestOperation)
