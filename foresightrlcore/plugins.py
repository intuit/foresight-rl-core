"""Entry-point plugin loading for foresightrlcore extensions."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Set, Type, cast

if sys.version_info >= (3, 10):
    from importlib.metadata import EntryPoint
else:
    EntryPoint = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

_PLUGINS_LOADED = False
_OPTIMIZER_PLUGINS_LOADED = False
_EVALUATE_PLUGINS_LOADED = False

#: Set by the SageMaker launchers to the extension packages staged into the
#: job source dir (see ``execution.sagemaker_common.stage_extra_packages``).
EXTRA_PACKAGES_ENV = "FORESIGHT_EXTRA_PACKAGES"

_LLM_PROVIDERS: Dict[str, Callable[..., object]] = {}
_EXTRA_PACKAGES_LOADED: Set[str] = set()


def _iter_entry_points(group: str) -> List[EntryPoint]:
    eps = entry_points(group=group)
    return list(eps)


def _load_callable(entry: EntryPoint) -> None:
    try:
        fn = entry.load()
        if callable(fn):
            fn()
        else:
            logger.warning("Plugin entry %r is not callable", entry.name)
    except Exception:
        logger.warning("Failed to load plugin %r", entry.name, exc_info=True)


def _import_entry_modules(group: str) -> None:
    for ep in _iter_entry_points(group):
        try:
            value = ep.load()
            if isinstance(value, str):
                importlib.import_module(value)
            elif callable(value):
                result = value()
                if isinstance(result, str):
                    importlib.import_module(result)
        except Exception:
            logger.warning("Failed to import plugin module %r", ep.name, exc_info=True)


def load_extra_packages() -> None:
    """Import extension packages staged into the job by the launcher.

    Staged packages ship as plain source with no ``.dist-info``, so
    ``entry_points()`` cannot discover them. Import each one and call its
    ``_register()`` hook directly. Names are recorded before importing so a
    package that re-enters plugin loading cannot recurse.
    """
    raw = os.environ.get(EXTRA_PACKAGES_ENV, "")
    for name in (n.strip() for n in raw.split(",")):
        if not name or name in _EXTRA_PACKAGES_LOADED:
            continue
        _EXTRA_PACKAGES_LOADED.add(name)
        try:
            module = importlib.import_module(name)
        except Exception:
            logger.warning("Failed to import staged package %r", name, exc_info=True)
            continue
        register = getattr(module, "_register", None)
        if callable(register):
            try:
                register()
                logger.info("Registered plugins from staged package %s", name)
            except Exception:
                logger.warning(
                    "Failed to register staged package %r", name, exc_info=True
                )


def register_llm_provider(name: str, factory: Callable[..., object]) -> None:
    _LLM_PROVIDERS[name] = factory


def get_llm_provider(name: str) -> Callable[..., object]:
    if name not in _LLM_PROVIDERS:
        available = sorted(_LLM_PROVIDERS)
        raise ValueError(
            f"Unknown llm_clients provider: {name!r}. Available: {available}"
        )
    return _LLM_PROVIDERS[name]


def registered_llm_providers() -> List[str]:
    return sorted(_LLM_PROVIDERS)


def register_auth_provider(name: str, provider_cls: Type[object]) -> None:
    from foresightrlcore.auth import register_provider

    register_provider(name, provider_cls)


def load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return

    from foresightrlcore.clients.bedrock import BedrockClientManager
    from foresightrlcore.config.schema import LLMClientConfig

    def _bedrock_factory(cfg: LLMClientConfig) -> BedrockClientManager:
        assert cfg.bedrock is not None
        return BedrockClientManager(
            model_id=cfg.bedrock.model_id,
            region=cfg.bedrock.region,
            temperature=cfg.bedrock.temperature,
        )

    register_llm_provider("bedrock", cast(Callable[..., object], _bedrock_factory))

    import foresightrlcore.auth.local  # noqa: F401

    for ep in _iter_entry_points("foresightrlcore.plugins"):
        _load_callable(ep)

    _import_entry_modules("foresightrlcore.tasks")
    _import_entry_modules("foresightrlcore.llm_providers")
    _import_entry_modules("foresightrlcore.auth_providers")
    load_extra_packages()

    _PLUGINS_LOADED = True


def load_optimizer_plugins() -> None:
    global _OPTIMIZER_PLUGINS_LOADED
    if _OPTIMIZER_PLUGINS_LOADED:
        return

    import foresightrlcore.optimizer.datasets  # noqa: F401
    import foresightrlcore.optimizer.judges.llm_judges  # noqa: F401
    import foresightrlcore.optimizer.modes.single_call  # noqa: F401
    import foresightrlcore.optimizer.prompt_sets  # noqa: F401

    _import_entry_modules("foresightrlcore.optimizer.plugins")
    load_extra_packages()

    _OPTIMIZER_PLUGINS_LOADED = True


def load_evaluate_plugins() -> None:
    """Register the built-in evaluate axes, then any extension-provided ones.

    Importing the built-in modules is what runs their ``@REGISTRY.register``
    decorators. Extensions contribute through the
    ``foresightrlcore.evaluate.plugins`` entry-point group, or -- inside a
    remote job, where there is no installed metadata -- through
    :func:`load_extra_packages`.
    """
    global _EVALUATE_PLUGINS_LOADED
    if _EVALUATE_PLUGINS_LOADED:
        return

    import foresightrlcore.evaluate.benchmarks  # noqa: F401
    import foresightrlcore.evaluate.datasets  # noqa: F401
    import foresightrlcore.evaluate.metrics  # noqa: F401
    import foresightrlcore.evaluate.reporters  # noqa: F401

    _import_entry_modules("foresightrlcore.evaluate.plugins")
    load_extra_packages()

    _EVALUATE_PLUGINS_LOADED = True


def discover_task_packages() -> None:
    import pkgutil

    import foresightrlcore.tasks as tasks_pkg

    skip = {"thinking_utils", "advice_judge_logic", "advice"}
    for module_info in pkgutil.iter_modules(
        tasks_pkg.__path__, prefix=tasks_pkg.__name__ + "."
    ):
        leaf = module_info.name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or leaf in skip:
            continue
        try:
            importlib.import_module(module_info.name)
        except ImportError as exc:
            logger.info("discover_tasks: skipping %s (%s)", module_info.name, exc)

    _import_entry_modules("foresightrlcore.tasks")
