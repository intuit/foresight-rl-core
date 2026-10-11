"""In-package registries for serverless reward Lambdas, datasets, and converters.

Three registries live here. Each one is a flat ``dict[str, Spec]``,
populated at task-package import time via ``register_*`` helpers.

* :data:`LAMBDAS` — reward-function Lambdas. **Registry keys match the
  SageMaker AI Registry evaluator name exactly** (one identifier per
  reward function, not two). When ``custom_reward_function_name`` in a
  YAML config does not resolve to an existing AI Registry evaluator,
  the framework looks the same name up here and auto-creates the
  evaluator from the registered ``LambdaSpec``.
* :data:`DATASETS` — known datasets with descriptions. Informational
  today; future cross-validation hooks here.
* :data:`CONVERTERS` — record-level JSONL converters used by
  ``rl-train serverless convert``.

Resolution helpers
~~~~~~~~~~~~~~~~~~

* :func:`resolve_lambda` — given a lambda name, return the
  ``(main_file_path, [extra_file_paths])`` tuple consumed by
  ``aws_serverless._create_lambda_with_extras``.
* :func:`resolve_converter` — given a converter name **or** a dotted
  path ``pkg.module:callable``, return the underlying callable.

Discovery
~~~~~~~~~

Tasks register at package import time. The CLI calls
:func:`discover_tasks` lazily so a heavy task package (e.g. one that
pulls ``torch``/``transformers`` at import) is only loaded when its
registry entry is actually requested — except for the registry-listing
``rl-train serverless list`` command which forces full discovery.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lambda registry
# ---------------------------------------------------------------------------


DEFAULT_LAMBDA_RUNTIME = "python3.12"
"""Default Lambda runtime when a ``LambdaSpec`` does not set one.

The SageMaker SDK constant ``EVALUATOR_DEFAULT_RUNTIME`` is ``python3.9``
(end-of-life Dec 2025); we override it because every reward Lambda we
ship today targets Python 3.12 and the AWS-side production evaluators
(e.g. ``my-task-rlvr-judge``) are deployed on 3.12 too."""

DEFAULT_LAMBDA_TIMEOUT_S = 900
"""Default Lambda timeout in seconds (15 min — AWS's maximum).

Sized for Bedrock-judged rewards: a single Claude/Sonnet/Opus
``converse`` call routinely takes 5-15s, and a reward Lambda is invoked
in batched mode (10-32 samples per call). AWS Lambda's hard ceiling is
900s; we use the ceiling so a slow judge call cannot fail the run."""

DEFAULT_LAMBDA_MEMORY_MB = 1024
"""Default Lambda memory in MB.

Lambda CPU scales linearly with memory, so 1024 MB also buys ~6x the
CPU of the 128 MB default — a meaningful difference for the JSON
post-processing the judge wrappers do after the LLM call returns.
Matches typical Bedrock-judged reward Lambda deployments."""

MAX_LAMBDA_FUNCTION_NAME_LEN = 64
"""AWS Lambda ``FunctionName`` length cap (hard service limit)."""


@dataclass(frozen=True)
class LambdaSpec:
    """How to package and create a reward Lambda."""

    name: str
    main_module: str
    """Dotted Python module path; placed at the zip root as
    ``lambda_function.py`` (SageMaker SDK hardcodes the handler to
    ``lambda_function.lambda_handler``)."""

    extra_modules: Tuple[str, ...] = field(default_factory=tuple)
    """Additional dotted module paths bundled at the zip root using their
    basenames, so ``main_module`` can ``from <basename> import ...``."""

    env_defaults: Dict[str, str] = field(default_factory=dict)
    """Default environment variables for the Lambda function (set at
    create-time; not propagated as YAML hyperparameters)."""

    runtime: str = DEFAULT_LAMBDA_RUNTIME
    """AWS Lambda runtime identifier (e.g. ``python3.12``). Overrides the
    SageMaker SDK's ``EVALUATOR_DEFAULT_RUNTIME`` (which is
    ``python3.9``, scheduled for end-of-life Dec 2025)."""

    timeout_s: int = DEFAULT_LAMBDA_TIMEOUT_S
    """Per-invocation timeout in seconds. AWS's hard ceiling is 900s.
    Use the ceiling for Bedrock-judged rewards; lower it only for
    deterministic rule-based rewards that complete in <1s."""

    memory_mb: int = DEFAULT_LAMBDA_MEMORY_MB
    """Lambda memory in MB. Lambda CPU scales with memory, so this also
    controls effective CPU. 128 MB (the AWS default) is too slow for
    any non-trivial reward function."""

    description: str = ""


LAMBDAS: Dict[str, LambdaSpec] = {}


def register_lambda(
    name: str,
    main_module: str,
    extra_modules: Optional[List[str]] = None,
    env_defaults: Optional[Dict[str, str]] = None,
    runtime: str = DEFAULT_LAMBDA_RUNTIME,
    timeout_s: int = DEFAULT_LAMBDA_TIMEOUT_S,
    memory_mb: int = DEFAULT_LAMBDA_MEMORY_MB,
    description: str = "",
) -> LambdaSpec:
    """Register a reward Lambda under the given name.

    ``runtime``, ``timeout_s``, and ``memory_mb`` map to the AWS Lambda
    ``CreateFunction`` parameters of the same name; the defaults are
    sized for Bedrock-judged rewards (see the ``DEFAULT_LAMBDA_*``
    constants). Rule-based rewards that complete in <1s should override
    ``timeout_s`` and ``memory_mb`` downward to save cost.
    """
    spec = LambdaSpec(
        name=name,
        main_module=main_module,
        extra_modules=tuple(extra_modules or ()),
        env_defaults=dict(env_defaults or {}),
        runtime=runtime,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        description=description,
    )
    if name in LAMBDAS and LAMBDAS[name] != spec:
        logger.warning(
            "register_lambda: overwriting existing entry %r with a "
            "different LambdaSpec",
            name,
        )
    LAMBDAS[name] = spec
    return spec


def resolve_lambda(name: str) -> Tuple[str, List[str]]:
    """Return ``(main_file_path, [extra_file_paths])`` for the named lambda.

    Imports each module to locate its ``__file__`` on disk. Raises
    ``KeyError`` if the name is not registered.
    """
    if name not in LAMBDAS:
        raise KeyError(
            f"Lambda {name!r} is not registered. "
            f"Registered: {sorted(LAMBDAS)!r}. "
            "Either register one via foresightrlcore.serverless.registry."
            "register_lambda, or set "
            "execution.serverless.custom_reward_function_source explicitly."
        )
    spec = LAMBDAS[name]
    main_path = _module_file(spec.main_module)
    extra_paths = [_module_file(m) for m in spec.extra_modules]
    return main_path, extra_paths


def _module_file(dotted: str) -> str:
    """Return the source-file path for a dotted module path."""
    module = importlib.import_module(dotted)
    path = getattr(module, "__file__", None)
    if not path:
        raise ImportError(
            f"Module {dotted!r} has no __file__ attribute (built-in?). "
            "Cannot bundle into a Lambda zip."
        )
    return path


# ---------------------------------------------------------------------------
# Dataset registry (informational)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Known dataset metadata."""

    name: str
    description: str = ""
    suggested_validation_dataset: Optional[str] = None


DATASETS: Dict[str, DatasetSpec] = {}


def register_dataset(
    name: str,
    description: str = "",
    suggested_validation_dataset: Optional[str] = None,
) -> DatasetSpec:
    spec = DatasetSpec(
        name=name,
        description=description,
        suggested_validation_dataset=suggested_validation_dataset,
    )
    DATASETS[name] = spec
    return spec


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConverterSpec:
    """Record-level JSONL converter metadata."""

    name: str
    mapper: str  # dotted path "pkg.module:callable"
    schemas: frozenset
    description: str = ""


CONVERTERS: Dict[str, ConverterSpec] = {}


def register_converter(
    name: str,
    mapper: str,
    schemas: Optional[Set[str]] = None,
    description: str = "",
) -> ConverterSpec:
    """Register a record-level JSONL converter under the given name."""
    spec = ConverterSpec(
        name=name,
        mapper=mapper,
        schemas=frozenset(schemas or {"rlvr"}),
        description=description,
    )
    if name in CONVERTERS and CONVERTERS[name] != spec:
        logger.warning(
            "register_converter: overwriting existing entry %r with a "
            "different ConverterSpec",
            name,
        )
    CONVERTERS[name] = spec
    return spec


def resolve_converter(name_or_path: str) -> Tuple[Callable, frozenset]:
    """Resolve a converter name (registry) or dotted path to ``(callable, schemas)``.

    Dotted-path form is ``pkg.module:callable``. The returned ``schemas``
    set is ``frozenset({"rlvr"})`` by default for dotted-path imports
    (callers can pass ``--schema`` to validate against any registered
    schema).
    """
    if ":" in name_or_path:
        callable_ = _import_dotted_callable(name_or_path)
        return callable_, frozenset({"rlvr"})

    if name_or_path not in CONVERTERS:
        raise KeyError(
            f"No registered converter named {name_or_path!r}. "
            f"Registered: {sorted(CONVERTERS)!r}. "
            "Pass a dotted path 'pkg.module:fn' to use an unregistered "
            "converter, or register one via "
            "foresightrlcore.serverless.registry.register_converter."
        )
    spec = CONVERTERS[name_or_path]
    return _import_dotted_callable(spec.mapper), spec.schemas


def _import_dotted_callable(spec: str) -> Callable:
    """Import ``pkg.module:callable`` and return the callable."""
    if ":" not in spec:
        raise ValueError(
            f"Dotted callable spec must be 'pkg.module:callable', got {spec!r}"
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise AttributeError(
            f"Module {module_path!r} has no attribute {attr!r} "
            f"(loaded from spec {spec!r})"
        )
    return getattr(module, attr)


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

_DISCOVERY_DONE = False


def discover_tasks() -> None:
    """Import every ``foresightrlcore.tasks.<task>`` package once.

    Each task package's ``__init__.py`` is expected to call the
    ``register_*`` helpers above at import time, so this is sufficient
    to populate every registry. Idempotent.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return

    from foresightrlcore.plugins import discover_task_packages, load_plugins

    load_plugins()
    discover_task_packages()

    _DISCOVERY_DONE = True


def truncate_lambda_function_name(name: str) -> str:
    """Truncate ``name`` to ``MAX_LAMBDA_FUNCTION_NAME_LEN`` chars.

    AWS Lambda ``CreateFunction`` rejects names longer than 64
    characters. The auto-generated name shape from the backend is
    ``SageMaker-evaluator-<spec_name>-YYYYMMDD_HHMMSS`` (the prefix is
    20 chars, the timestamp suffix is 16 chars). If the resulting name
    exceeds the cap, this helper truncates from the **middle** of
    ``name`` — preserving both the ``SageMaker-evaluator-`` prefix and
    the timestamp suffix so the SDK convention is intact and the name
    stays unique on re-runs.

    Naming policy keeps the result deterministic (no random characters)
    so the same evaluator name + same minute always produces the same
    Lambda name — important for the registry-path smoke that asserts
    "byte-identical to production".
    """
    if len(name) <= MAX_LAMBDA_FUNCTION_NAME_LEN:
        return name
    overflow = len(name) - MAX_LAMBDA_FUNCTION_NAME_LEN
    midpoint = len(name) // 2
    start = midpoint - (overflow // 2)
    end = start + overflow
    truncated = name[:start] + name[end:]
    logger.warning(
        "Lambda FunctionName %r exceeds AWS's %d-char limit by %d; " "truncated to %r",
        name,
        MAX_LAMBDA_FUNCTION_NAME_LEN,
        overflow,
        truncated,
    )
    return truncated


__all__ = [
    "CONVERTERS",
    "ConverterSpec",
    "DATASETS",
    "DEFAULT_LAMBDA_MEMORY_MB",
    "DEFAULT_LAMBDA_RUNTIME",
    "DEFAULT_LAMBDA_TIMEOUT_S",
    "DatasetSpec",
    "LAMBDAS",
    "LambdaSpec",
    "MAX_LAMBDA_FUNCTION_NAME_LEN",
    "discover_tasks",
    "register_converter",
    "register_dataset",
    "register_lambda",
    "resolve_converter",
    "resolve_lambda",
    "truncate_lambda_function_name",
]
