"""Training hooks — import and resolve custom callbacks, hooks, and trainer classes.

All hooks are specified as dotted Python paths in the YAML config under
``training.hooks``.  This module provides the loader utilities that the
method handler calls at training time.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic model/object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def import_dotted_path(dotted_path: str) -> Any:
    """Import a class or callable from a dotted path like ``'pkg.module.ClassName'``.

    The last component is treated as an attribute of the module formed by the
    preceding components.  For example ``'my_pkg.callbacks.EarlyStopping'``
    imports ``my_pkg.callbacks`` and returns ``EarlyStopping`` from it.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute does not exist on the module.
    """
    if "." not in dotted_path:
        raise ImportError(
            f"Invalid dotted path '{dotted_path}': must contain at least "
            f"one dot separating the module from the attribute "
            f"(e.g. 'my_pkg.module.ClassName')."
        )

    module_path, _, attr_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import module '{module_path}' "
            f"from dotted path '{dotted_path}': {exc}"
        ) from exc

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}' "
            f"(from dotted path '{dotted_path}')"
        )

    return getattr(module, attr_name)


def resolve_trainer_class(
    hooks: Any,
    default_cls: type,
) -> type:
    """Return the trainer class to use.

    If *hooks* is ``None`` or ``hooks.trainer_class`` is not set, returns
    *default_cls*.  Otherwise imports the custom class from the dotted path.
    """
    if hooks is None:
        return default_cls

    trainer_class_path: Optional[str] = _get_field(hooks, "trainer_class")
    if trainer_class_path is None:
        return default_cls

    cls = import_dotted_path(trainer_class_path)
    logger.info(f"Using custom trainer class: {trainer_class_path}")
    return cls


def resolve_callbacks(hooks: Any) -> list:
    """Import and instantiate extra ``TrainerCallback`` classes from hooks config.

    Returns an empty list when *hooks* is ``None`` or has no callbacks.
    """
    if hooks is None:
        return []

    callback_paths: Optional[list[str]] = _get_field(hooks, "callbacks")
    if not callback_paths:
        return []

    instances: list[Any] = []
    for path in callback_paths:
        cls = import_dotted_path(path)
        instances.append(cls())
        logger.info(f"Loaded hook callback: {path}")

    return instances


def resolve_pre_post_hooks(
    hooks: Any,
) -> tuple[Optional[Callable[..., None]], Optional[Callable[..., None]]]:
    """Import pre-train and post-train hook callables from hooks config.

    Returns ``(pre_train_fn, post_train_fn)``.  Either or both may be
    ``None`` if not configured.
    """
    if hooks is None:
        return None, None

    pre_fn: Optional[Callable[..., None]] = None
    post_fn: Optional[Callable[..., None]] = None

    pre_path: Optional[str] = _get_field(hooks, "pre_train")
    if pre_path is not None:
        pre_fn = import_dotted_path(pre_path)
        logger.info(f"Loaded pre_train hook: {pre_path}")

    post_path: Optional[str] = _get_field(hooks, "post_train")
    if post_path is not None:
        post_fn = import_dotted_path(post_path)
        logger.info(f"Loaded post_train hook: {post_path}")

    return pre_fn, post_fn
