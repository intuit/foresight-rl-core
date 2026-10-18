"""``vllm serve`` command/env construction and process supervision.

Lives in ``common`` because two layers need it and neither should depend on the
other: :mod:`foresightrlcore.generation.local_vllm_client` (optimizer-style
per-call generation) and :mod:`foresightrlcore.inference.local_vllm` (an
endpoint handed to an external benchmark).

Health checking uses ``urllib`` rather than ``requests`` so the inference
substrate stays stdlib-only.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from foresightrlcore.config.perf_schema import VllmServingConfig

logger = logging.getLogger(__name__)

DEFAULT_SERVED_NAME = "served-model"

#: Name a served LoRA adapter is registered and addressed under. vLLM exposes
#: each adapter as its own model id, so a request naming the base model gets
#: the base model -- unadapted, silently, and scoring like a failed fine-tune.
DEFAULT_ADAPTER_NAME = "adapter"


def build_serve_command(
    model_path: str,
    serving: "VllmServingConfig",
    *,
    lora_adapter: str | None = None,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> List[str]:
    """Build the ``vllm serve`` argv for ``model_path`` under ``serving``.

    Args:
        model_path: Base weights to serve.
        serving: vLLM serving parameters.
        lora_adapter: Optional LoRA adapter directory to attach to the base.
        adapter_name: Model id the adapter is addressed under.
    """
    cmd: List[str] = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(serving.port),
        "--tensor-parallel-size",
        str(serving.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
        "--max-num-seqs",
        str(serving.max_num_seqs),
        "--served-model-name",
        serving.served_model_name or DEFAULT_SERVED_NAME,
    ]
    if serving.max_model_len is not None:
        cmd += ["--max-model-len", str(serving.max_model_len)]
    if serving.trust_remote_code:
        cmd.append("--trust-remote-code")
    if serving.enforce_eager:
        cmd.append("--enforce-eager")
    if serving.dtype:
        cmd += ["--dtype", serving.dtype]
    if serving.quantization:
        cmd += ["--quantization", serving.quantization]
    if serving.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if serving.speculative is not None:
        cmd += [
            "--speculative-config",
            json.dumps(
                serving.speculative.model_dump(exclude_none=True),
                separators=(",", ":"),
            ),
        ]
    if lora_adapter:
        cmd += ["--enable-lora", "--lora-modules", f"{adapter_name}={lora_adapter}"]
        if serving.max_lora_rank is not None:
            cmd += ["--max-lora-rank", str(serving.max_lora_rank)]
    for flag, value in serving.extra_args.items():
        # An empty value means a store_true option, which argparse rejects if
        # handed an empty string. Without this there is no way to reach vllm's
        # store_true flags at all.
        cmd += [f"--{flag}"] if value == "" else [f"--{flag}", str(value)]
    return cmd


def build_serve_env(serving: "VllmServingConfig") -> Dict[str, str]:
    """Return the process environment for ``vllm serve``."""
    env = os.environ.copy()
    for key, value in serving.vllm_env.items():
        env[key] = value
    if serving.gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = serving.gpu_devices
    return env


def health_ok(port: int, *, timeout: float = 2.0) -> bool:
    """True when ``GET /health`` on ``port`` answers 200."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as resp:
            return bool(200 <= resp.status < 300)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def terminate_process_group(
    proc: "subprocess.Popen[bytes]", *, grace_seconds: int = 30
) -> None:
    """SIGTERM the process group, escalating to SIGKILL after ``grace_seconds``."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        logger.warning("could not signal process group cleanly: %s", exc)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("process did not exit in %ds; sending SIGKILL", grace_seconds)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def wait_until_healthy(
    proc: "subprocess.Popen[bytes]",
    port: int,
    *,
    timeout: float,
    poll_interval: float = 5.0,
) -> float:
    """Block until the server answers ``/health``; return seconds elapsed.

    Raises:
        RuntimeError: If the process exits first, or the timeout expires.
    """
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} "
                f"before becoming healthy"
            )
        if health_ok(port):
            return time.monotonic() - start
        time.sleep(poll_interval)
    raise RuntimeError(f"vllm serve did not become healthy within {timeout}s")


__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "DEFAULT_SERVED_NAME",
    "build_serve_command",
    "build_serve_env",
    "health_ok",
    "terminate_process_group",
    "wait_until_healthy",
]
