"""Single, spawn-safe entry point for every ``vllm_mode = server`` launch.

Why this script exists
----------------------

Every backend that runs vLLM in server mode (SageMaker via ``entrypoint.sh``,
local-GPU via ``foresightrlcore.execution.local``, HyperPod EKS via
``foresightrlcore.execution.aws_hyperpod_eks.manifests``) routes through this
single module instead of invoking ``trl vllm-serve`` directly. Two reasons:

1. **Process-startup hygiene.** vLLM's tensor-parallel workers spawn child
   processes via Python's ``multiprocessing`` module. On H100/H200 (P5/P5en)
   the workers crash with::

       RuntimeError: Cannot re-initialize CUDA in forked subprocess.
       To use CUDA with multiprocessing, you must use the 'spawn' start method.

   unless both of the following are true **before** any module that imports
   ``torch.cuda`` (or transitively triggers ``cuInit()``) is loaded:

   - ``multiprocessing.set_start_method("spawn", force=True)`` has been called.
     Exporting ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` from a parent shell is
     *not* sufficient because the ``multiprocessing`` module caches its
     default context as soon as TRL's CLI entrypoint is imported.
   - ``CUDA_MODULE_LOADING=LAZY`` is in the environment so that libcuda does
     not eagerly load every kernel module at first ``cuInit()``. On H200 the
     eager path can segfault inside ``cuGetProcAddress_v2``.

2. **TRL source patching.** TRL's ``vllm-serve`` CLI does not expose
   ``reasoning_config``, ``compilation_config``, or ``disable_custom_all_reduce``,
   but vLLM's ``LLM(...)`` constructor accepts them via ``**kwargs``. Rather
   than fork TRL, ``patch_trl_vllm.patch(...)`` rewrites TRL's installed
   ``vllm_serve.py`` on disk to inject those kwargs. Doing the patch from
   inside the launcher (rather than from bash) guarantees the same code path
   runs in every backend.

Argument forwarding contract
----------------------------

Every CLI flag is forwarded verbatim to ``trl vllm-serve``. The launcher does
not parse, rewrite, or validate vLLM/TRL arguments — this keeps it resilient
to TRL upstream argument changes.

Environment-variable contract
-----------------------------

Spawn-safety defaults (always applied with ``setdefault`` so an operator can
override by exporting the variable upstream):

* ``VLLM_WORKER_MULTIPROC_METHOD=spawn``
* ``CUDA_MODULE_LOADING=LAZY``
* ``VLLM_USE_V1=0``
* ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``

Optional TRL patch opt-ins (read once at launch, all three are independent):

* ``VLLM_REASONING_CONFIG_JSON`` — JSON object with
  ``reasoning_start_str`` and ``reasoning_end_str`` keys. When set, the
  launcher injects ``reasoning_config=ReasoningConfig(...)`` into TRL's
  ``LLM(...)`` call.
* ``VLLM_COMPILATION_CONFIG_JSON`` — JSON object forwarded as
  ``compilation_config=CompilationConfig(**kwargs)``.
* ``VLLM_DISABLE_CUSTOM_ALL_REDUCE`` — set to ``"1"`` to inject
  ``disable_custom_all_reduce=True``.

If a patch opt-in is set but TRL's source has changed and the marker can no
longer be found, the launcher exits non-zero. This is intentional: a silent
feature regression on an upgrade is worse than a fail-fast crash at job start.

Usage (called from any backend)::

    # When the foresightrlcore package is installed (local backend):
    python3 -m foresightrlcore.runtime.scripts.vllm_launcher \\
        --model <path> --host 0.0.0.0 --port 8002 \\
        --tensor-parallel-size 4 --gpu-memory-utilization 0.85 [...]

    # When the script is shipped flat in a code tarball (SageMaker / EKS):
    python3 vllm_launcher.py --model ... [...]

Both shapes are exercised in production. The launcher resolves
``patch_trl_vllm`` via a dual-import strategy so both layouts work.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from typing import Any, Dict, Mapping, Optional

# Process-startup environment defaults applied before any vLLM/torch import.
# Each value is treated as a default — if the operator has already set the
# variable in the env, the existing value wins.
_REQUIRED_ENV_DEFAULTS: Mapping[str, str] = {
    # Force vLLM TP workers to use spawn instead of fork. Combined with
    # multiprocessing.set_start_method("spawn", force=True) below, this
    # eliminates the "Cannot re-initialize CUDA in forked subprocess" crash
    # on H100/H200 when the parent has any CUDA context (e.g. from a stray
    # `import torch` during dependency setup).
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # Defer kernel-module loading until first kernel launch. Eager loading
    # at cuInit() time can segfault inside cuGetProcAddress_v2 on H200.
    "CUDA_MODULE_LOADING": "LAZY",
    # vLLM 0.20.x ships both V0 and V1 engines. The V1 worker manager
    # forks more aggressively than V0 and is incompatible with our
    # entrypoint sequence (pip install + flash-attn rebuild can leave a
    # CUDA context in the parent shell). V0 is stable on the same path.
    "VLLM_USE_V1": "0",
    # vLLM's collective_rpc passes a few non-trivially-pickleable objects
    # (e.g. torch.dtype) between TP workers. The "insecure" path is the
    # only path that actually serializes them; the secure path deadlocks.
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}

# Env-var names that drive optional TRL source patching.
ENV_REASONING_CONFIG = "VLLM_REASONING_CONFIG_JSON"
ENV_COMPILATION_CONFIG = "VLLM_COMPILATION_CONFIG_JSON"
ENV_DISABLE_CUSTOM_ALL_REDUCE = "VLLM_DISABLE_CUSTOM_ALL_REDUCE"


def apply_env_defaults(
    env: dict[str, str] | None = None,
    defaults: Mapping[str, str] = _REQUIRED_ENV_DEFAULTS,
) -> dict[str, str]:
    """Apply spawn-safety env defaults without overriding operator values.

    Mutates ``env`` in place and returns it. When ``env`` is ``None`` the
    real ``os.environ`` is mutated (the production path). The dict
    indirection exists for unit testing.
    """
    target = os.environ if env is None else env
    for key, value in defaults.items():
        target.setdefault(key, value)
    return target  # type: ignore[return-value]


def _parse_json_env(name: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON-object env var, returning ``None`` if empty.

    Raises ``SystemExit`` with code 2 on malformed JSON — fail-fast matches
    the rest of the launcher: a broken opt-in is worse than a silent drop.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: env var {name} is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(parsed, dict):
        print(
            f"ERROR: env var {name} must be a JSON object, got {type(parsed).__name__}",
            file=sys.stderr,
        )
        sys.exit(2)
    return parsed


def apply_trl_patch_from_env() -> int:
    """Apply ``patch_trl_vllm.patch`` based on opt-in env vars.

    Returns 0 when no opt-ins are present (the no-op fast path), 0 when the
    patch succeeds (or is already applied), and a non-zero code propagated
    from ``patch_trl_vllm.patch`` when the patch was requested but cannot
    be applied. Callers should treat any non-zero return as fatal.
    """
    reasoning = _parse_json_env(ENV_REASONING_CONFIG)
    compilation = _parse_json_env(ENV_COMPILATION_CONFIG)
    disable_car = os.environ.get(ENV_DISABLE_CUSTOM_ALL_REDUCE, "0") == "1"

    if reasoning is None and compilation is None and not disable_car:
        return 0

    reasoning_start: Optional[str] = None
    reasoning_end: Optional[str] = None
    if reasoning is not None:
        reasoning_start = reasoning.get("reasoning_start_str")
        reasoning_end = reasoning.get("reasoning_end_str")
        if not (reasoning_start and reasoning_end):
            print(
                f"ERROR: {ENV_REASONING_CONFIG} must contain non-empty "
                "'reasoning_start_str' and 'reasoning_end_str' fields.",
                file=sys.stderr,
            )
            return 2

    # Import patch_trl_vllm with a dual-path strategy. Two deployment shapes
    # exist:
    #   1. Installed package — backends use `python -m foresightrlcore.runtime.
    #      scripts.vllm_launcher` and the package import works.
    #   2. Script-root copy — SageMaker and HyperPod EKS copy the contents of
    #      runtime/scripts/ to a flat code root (e.g. /opt/ml/code/) and run
    #      `python vllm_launcher.py ...`. In that layout `patch_trl_vllm.py`
    #      sits next to this file and the foresightrlcore package import
    #      doesn't resolve (`runtime/` is excluded from the package copy).
    # The import below tries (1) first; (2) is a fallback that adds the
    # launcher's own directory to sys.path before retrying.
    try:
        from foresightrlcore.runtime.scripts.patch_trl_vllm import patch
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from patch_trl_vllm import patch  # type: ignore[no-redef]

    return patch(
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        compilation_config=compilation,
        disable_custom_all_reduce=disable_car,
    )


def main(argv: list[str] | None = None) -> int:
    """Apply spawn-safety hygiene + TRL patch then hand off to TRL's main.

    Returns the exit code from TRL's main (or 0 if main returns None).
    Imports of TRL/vLLM are deferred until after env defaults, set_start_method,
    and the optional TRL patch have all been applied, so the spawn context is
    established and the patched source is what TRL imports.
    """
    apply_env_defaults()

    # force=True is required: multiprocessing may have already initialized
    # a default context elsewhere in the import chain. Calling without
    # force first is undefined behavior across Python minor versions.
    multiprocessing.set_start_method("spawn", force=True)

    patch_rc = apply_trl_patch_from_env()
    if patch_rc != 0:
        return patch_rc

    # Allow tests to inject argv. In production we forward sys.argv minus
    # the launcher script name so TRL's argparse sees a clean list.
    if argv is None:
        argv = sys.argv[1:]

    sys.argv = [sys.argv[0], *argv]

    from trl.scripts.vllm_serve import main as trl_main
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    rc = trl_main(script_args)
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
