"""Base protocol for operations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Operation(Protocol):
    """Interface for top-level operations (train, eval, test, etc.)."""

    name: str

    def run(self, config: Any, wait: bool = False) -> str:
        """Execute the operation. Returns a job/run identifier."""
        ...
