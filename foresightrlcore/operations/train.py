"""Train operation — the primary operation for rl-train.

Orchestrates: config validation -> backend resolution -> method resolution -> launch.
"""

from __future__ import annotations

import logging
from typing import Any

from foresightrlcore.config.validation import validate_config
from foresightrlcore.execution import get_backend
from foresightrlcore.operations import register_operation

logger = logging.getLogger(__name__)


class TrainOperation:
    """Launch a training job on the resolved execution backend."""

    name = "train"

    def run(self, config: Any, wait: bool = False) -> str:
        errors = validate_config(config)
        if errors:
            msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

        backend_type = config.execution.type
        logger.info(f"Execution backend: {backend_type}")

        backend_cls = get_backend(backend_type)
        backend = backend_cls()

        backend_errors = backend.validate(config)
        if backend_errors:
            msg = f"Backend '{backend_type}' validation failed:\n" + "\n".join(
                f"  - {e}" for e in backend_errors
            )
            raise ValueError(msg)

        job_id = backend.launch(config, wait=wait)
        logger.info(f"Job launched: {job_id}")
        return job_id


register_operation("train", TrainOperation)
