"""Base protocol for execution backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutionBackend(Protocol):
    """Interface that every execution backend must implement.

    An ExecutionBackend is responsible for:
      1. Validating that the config is compatible with this backend
      2. Launching the training job on the target infrastructure
      3. Optionally reporting job status
    """

    backend_type: str

    def validate(self, config: Any) -> list[str]:
        """Validate backend-specific config. Returns list of errors (empty = valid)."""
        ...

    def launch(self, config: Any, wait: bool = False) -> str:
        """Launch a training job. Returns a job identifier string."""
        ...

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a launched job. Returns provider-specific status dict."""
        ...
