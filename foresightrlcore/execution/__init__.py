"""foresightrlcore.execution — execution backend registry."""

from __future__ import annotations

_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_cls: type) -> None:
    """Register an execution backend class."""
    _BACKENDS[name] = backend_cls


def get_backend(name: str) -> type:
    """Retrieve a registered execution backend class by name."""
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise KeyError(f"Unknown execution backend '{name}'. Available: {available}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return list(_BACKENDS.keys())
