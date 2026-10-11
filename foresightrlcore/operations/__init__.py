"""foresightrlcore.operations — operation registry (train, eval, test, etc.)."""

from __future__ import annotations

_OPERATIONS: dict[str, type] = {}


def register_operation(name: str, op_cls: type) -> None:
    """Register an operation class."""
    _OPERATIONS[name] = op_cls


def get_operation(name: str) -> type:
    """Retrieve a registered operation class by name."""
    if name not in _OPERATIONS:
        available = list(_OPERATIONS.keys())
        raise KeyError(f"Unknown operation '{name}'. Available: {available}")
    return _OPERATIONS[name]


def list_operations() -> list[str]:
    """Return names of all registered operations."""
    return list(_OPERATIONS.keys())
