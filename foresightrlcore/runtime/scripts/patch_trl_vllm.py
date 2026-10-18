"""Patch TRL's vllm_serve.py to inject vLLM engine config not exposed by TRL.

TRL's vllm-serve command creates vLLM via ``LLM(...)`` but doesn't expose
``reasoning_config`` or ``compilation_config`` as parameters. vLLM's ``LLM()``
constructor accepts both via ``**kwargs`` -> ``EngineArgs``, so we patch TRL's
source file in place to inject them into the ``LLM(...)`` call.

Three injections are supported:

  1. ``reasoning_config`` — enables ``ThinkingTokenBudgetLogitsProcessor`` by
     providing reasoning boundary tokens at the engine level.
  2. ``compilation_config`` — controls CUDA graph capture (sizes, warmups, mode)
     to avoid the graph-compilation hang on H100 while retaining partial CUDA
     graph speedup.
  3. ``disable_custom_all_reduce=True`` — workaround for vLLM 0.20.0's broken
     custom all-reduce kernel with ``TP>1``.

Behavior contract (matches the FILM-5307 port plan):

- The script is invoked from ``entrypoint.sh`` **only** when the user opts in
  via the relevant YAML fields (``infra.vllm.reasoning_*``,
  ``infra.vllm.compilation_config``) or the ``VLLM_DISABLE_CUSTOM_ALL_REDUCE=1``
  env var.
- If the user opted in but the patch cannot be applied (marker missing →
  most likely TRL upstream rewrote ``vllm_serve.py``), the script exits with a
  non-zero status. ``entrypoint.sh`` propagates the failure so the user sees the
  broken upgrade immediately at job start instead of getting a silent feature
  regression.
- If the source file is already patched (re-run, idempotent path), the script
  exits zero with an informational message.

Usage (called from ``entrypoint.sh``)::

    python3 patch_trl_vllm.py \
        --reasoning-start "<think>" --reasoning-end "</think>" \
        --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}' \
        [--disable-custom-all-reduce]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# The string we look for in TRL's vllm_serve.py to identify the LLM(...) call.
# If TRL upstream changes this line, the patch script will fail fast (when the
# user has opted into one of the new fields).
LLM_CALL_MARKER = "model_impl=script_args.vllm_model_impl,"


def _trl_version() -> str:
    """Installed trl version, or a placeholder if it cannot be determined."""
    try:
        from importlib.metadata import version

        return version("trl")
    except Exception:
        return "?"


def find_vllm_serve() -> Path:
    """Locate TRL's vllm_serve.py in the installed packages."""
    spec = importlib.util.find_spec("trl.scripts.vllm_serve")
    if spec is None or spec.origin is None:
        print(
            "ERROR: cannot find trl.scripts.vllm_serve module — is trl installed?",
            file=sys.stderr,
        )
        sys.exit(2)
    return Path(spec.origin)


def patch(
    reasoning_start: Optional[str],
    reasoning_end: Optional[str],
    compilation_config: Optional[Dict[str, Any]],
    disable_custom_all_reduce: bool,
) -> int:
    """Apply the requested injections to TRL's vllm_serve.py.

    Returns 0 on success (or already-applied no-op), non-zero on failure.
    """
    requested_reasoning = bool(reasoning_start and reasoning_end)
    requested_any = (
        requested_reasoning
        or compilation_config is not None
        or disable_custom_all_reduce
    )
    if not requested_any:
        print("No patch options requested; nothing to do.")
        return 0

    path = find_vllm_serve()
    source = path.read_text()

    if LLM_CALL_MARKER not in source:
        # Fail-fast: the user opted into at least one feature but TRL's source
        # doesn't have the expected marker. Most likely cause: TRL upstream
        # changed vllm_serve.py and the patch needs to be updated.
        print(
            f"ERROR: TRL {_trl_version()}'s vllm_serve.py at {path} does not "
            f"contain the expected marker {LLM_CALL_MARKER!r}. The patch cannot "
            "be applied. TRL 1.11 replaced the in-process LLM(...) call with an "
            "os.execve of `vllm serve`, so releases >=1.11 are not patchable; "
            "the container recipe must pin trl below that.",
            file=sys.stderr,
        )
        return 3

    needs_reasoning = requested_reasoning and "reasoning_config=" not in source
    needs_compilation = (
        compilation_config is not None and "compilation_config=" not in source
    )
    needs_disable_all_reduce = (
        disable_custom_all_reduce and "disable_custom_all_reduce=" not in source
    )

    if not (needs_reasoning or needs_compilation or needs_disable_all_reduce):
        print(f"Patch already applied to {path}; skipping (idempotent).")
        return 0

    # ── Inject imports for the dataclasses we reference below ──────────────
    import_target = "from vllm import LLM"
    idx = source.find(import_target)
    if idx != -1:
        line_start = source.rfind("\n", 0, idx) + 1
        indent = source[line_start:idx]

        imports_to_add = []
        if needs_reasoning and "from vllm.config import ReasoningConfig" not in source:
            imports_to_add.append(f"{indent}from vllm.config import ReasoningConfig")
        if (
            needs_compilation
            and "from vllm.config import CompilationConfig" not in source
        ):
            imports_to_add.append(f"{indent}from vllm.config import CompilationConfig")

        if imports_to_add:
            source = source.replace(
                import_target,
                import_target + "\n" + "\n".join(imports_to_add),
                1,
            )

    # ── Inject kwargs into the LLM(...) call ───────────────────────────────
    injections = [LLM_CALL_MARKER]

    if needs_reasoning:
        injections.append(
            f"            reasoning_config=ReasoningConfig("
            f'reasoning_start_str="{reasoning_start}", '
            f'reasoning_end_str="{reasoning_end}"),'
        )
        print(
            f"  -> injecting reasoning_config "
            f'(start="{reasoning_start}", end="{reasoning_end}")'
        )

    if needs_compilation:
        assert compilation_config is not None  # narrowed by needs_compilation
        cc_kwargs = ", ".join(f"{k}={v!r}" for k, v in compilation_config.items())
        injections.append(
            f"            compilation_config=CompilationConfig({cc_kwargs}),"
        )
        print(f"  -> injecting compilation_config ({compilation_config})")

    if needs_disable_all_reduce:
        injections.append("            disable_custom_all_reduce=True,")
        print("  -> injecting disable_custom_all_reduce=True")

    patched = source.replace(LLM_CALL_MARKER, "\n".join(injections))
    path.write_text(patched)
    print(f"Patched {path} successfully.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch TRL vllm_serve for reasoning + compilation config",
    )
    parser.add_argument(
        "--reasoning-start",
        default=None,
        help="Reasoning start string (e.g. '<think>')",
    )
    parser.add_argument(
        "--reasoning-end",
        default=None,
        help="Reasoning end string (e.g. '</think>')",
    )
    parser.add_argument(
        "--compilation-config",
        default=None,
        help="JSON dict for CompilationConfig kwargs "
        "(e.g. '{\"cudagraph_capture_sizes\":[1,2,4]}')",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Inject disable_custom_all_reduce=True into LLM()",
    )
    args = parser.parse_args()

    compilation_config: Optional[Dict[str, Any]] = None
    if args.compilation_config:
        try:
            compilation_config = json.loads(args.compilation_config)
        except json.JSONDecodeError as e:
            print(
                f"ERROR: invalid JSON for --compilation-config: {e}",
                file=sys.stderr,
            )
            sys.exit(2)

    print("Patching TRL vllm_serve.py ...")
    rc = patch(
        args.reasoning_start,
        args.reasoning_end,
        compilation_config,
        args.disable_custom_all_reduce,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
