"""Pluggable task components (reward, callbacks, dataset)."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Dict, Protocol

logger = logging.getLogger(__name__)

_TASK_ALIASES: Dict[str, str] = {}


class TaskModule(Protocol):
    """Expected interface for task modules."""

    def get_reward_function(self, task_config: dict[str, Any]) -> Callable: ...
    def get_callbacks(self, task_config: dict[str, Any]) -> list: ...
    def get_dataset_loader(
        self, task_config: dict[str, Any], model_name: str
    ) -> Callable: ...


def register_task_alias(name: str, module_path: str) -> None:
    """Map a short task name (e.g. ``advice``) to a dotted module path."""
    _TASK_ALIASES[name] = module_path


def load_task_module(task_type: str) -> Any:
    """Dynamically import a task module by name or dotted path.

    Resolution order:
      1. File system path (``task.type = "/path/to/module.py"``)
      2. Dotted Python path (e.g. ``my_project.rewards.custom``)
      3. Built-in ``foresightrlcore.tasks.<task_type>``
    """
    if task_type.startswith("/") or task_type.endswith(".py"):
        return _load_from_path(task_type)

    from foresightrlcore.plugins import load_plugins

    load_plugins()

    if task_type in _TASK_ALIASES:
        return importlib.import_module(_TASK_ALIASES[task_type])

    if "." in task_type:
        try:
            return importlib.import_module(task_type)
        except ModuleNotFoundError:
            pass

    try:
        return importlib.import_module(f"foresightrlcore.tasks.{task_type}")
    except ModuleNotFoundError:
        pass

    raise ValueError(
        f"Unknown task type '{task_type}'. "
        f"Provide a built-in name, a dotted Python path, or a file path."
    )


def _load_from_path(file_path: str) -> Any:
    """Load a task module from a filesystem path."""
    import importlib.util

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Task module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("_external_task", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_task_components(
    task_type: str, task_config: dict[str, Any], model_name: str
) -> tuple[Callable, list, Callable]:
    """Load all task components: reward function, callbacks, and dataset loader."""
    module = load_task_module(task_type)
    reward_fn = module.get_reward_function(task_config)
    callbacks = module.get_callbacks(task_config)
    dataset_loader = module.get_dataset_loader(task_config, model_name)
    return reward_fn, callbacks, dataset_loader
