"""Generic name -> factory registry with a decorator-based register API.

Every pluggable axis in the package is one of these: optimizer datasets, prompt
sets, modes and judges; evaluate benchmark types, dataset loaders, metrics and
reporters. Adding an implementation is a new class plus
``@REGISTRY.register("name")`` plus a config entry -- no core edits.

Usage::

    from foresightrlcore.common.registry import Registry

    MODES: "Registry[Callable[..., object]]" = Registry("mode")

    @MODES.register("summary_state")
    def _make() -> SummaryStateMode:
        return SummaryStateMode()

    mode = MODES.get("summary_state")()
    names = MODES.list()
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A tiny name -> object registry with a decorator-based register API.

    Args:
        kind: Human-readable label used in error messages (e.g. ``"mode"``).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers ``obj`` under ``name``.

        Raises:
            ValueError: If ``name`` is empty or already registered.
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"{self._kind} name must be a non-empty string")

        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"duplicate {self._kind} registration: {name!r}")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        """Return the object registered under ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind}: {name!r}. known: {sorted(self._items)}"
            )
        return self._items[name]

    def list(self) -> List[str]:
        """Return registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["Registry"]
