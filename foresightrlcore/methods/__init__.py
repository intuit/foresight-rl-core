"""foresightrlcore.methods — training method registry and handlers."""

from __future__ import annotations

_METHODS: dict[str, type] = {}

TRL_TRAINERS: dict[str, tuple[str, str]] = {
    "grpo": ("trl.GRPOTrainer", "trl.GRPOConfig"),
    "dpo": ("trl.DPOTrainer", "trl.DPOConfig"),
    "kto": ("trl.KTOTrainer", "trl.KTOConfig"),
}
"""Mapping of method type to ``(TrainerClass, ConfigClass)`` dotted paths.

Used by :class:`TrlMethodHandler` to lazily resolve TRL classes at runtime.
New methods can be added here without writing a dedicated handler subclass
(though a subclass is still recommended for method-specific validation).
"""


def register_method(name: str, handler_cls: type) -> None:
    """Register a method handler class."""
    _METHODS[name] = handler_cls


def get_method(name: str) -> type:
    """Retrieve a registered method handler class by name."""
    if name not in _METHODS:
        available = list(_METHODS.keys())
        raise KeyError(f"Unknown method '{name}'. Available: {available}")
    return _METHODS[name]


def list_methods() -> list[str]:
    """Return names of all registered methods."""
    return list(_METHODS.keys())
