"""A locally served model exposed as an :class:`InferenceEndpoint`.

``vllm serve`` already speaks OpenAI, so there is nothing to translate -- this
adapter only owns the process lifecycle and hands back the endpoint once
``/health`` answers.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from foresightrlcore.common.vllm_process import (
    DEFAULT_ADAPTER_NAME,
    DEFAULT_SERVED_NAME,
    build_serve_command,
    build_serve_env,
    terminate_process_group,
    wait_until_healthy,
)
from foresightrlcore.inference.interfaces import InferenceEndpoint

if TYPE_CHECKING:
    from foresightrlcore.config.perf_schema import VllmServingConfig

logger = logging.getLogger(__name__)

# vLLM ignores the key, but clients insist on one being present.
LOCAL_API_KEY = "EMPTY"


class VllmServerEndpoint:
    """Owns a ``vllm serve`` subprocess and exposes it as an endpoint.

    Args:
        model_path: Local path, HF id, or S3-staged directory to serve.
        serving: vLLM serving parameters (port, TP size, dtype, ...).
        model_name: Logical candidate name used in reports. Defaults to
            ``model_path``.
        lora_adapter: Optional LoRA adapter directory to serve on top of
            ``model_path``. When set, the endpoint's model id becomes the
            adapter's, so callers reach the adapted model rather than the base.
    """

    def __init__(
        self,
        model_path: str,
        serving: "VllmServingConfig",
        *,
        model_name: Optional[str] = None,
        lora_adapter: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.serving = serving
        self.model_name = model_name or model_path
        self.lora_adapter = lora_adapter
        # With an adapter loaded, vLLM answers to both the base name and the
        # adapter name. Addressing the base would quietly return unadapted
        # completions, so the adapter's id wins whenever there is one.
        self.served_model_name = (
            DEFAULT_ADAPTER_NAME
            if lora_adapter
            else (serving.served_model_name or DEFAULT_SERVED_NAME)
        )
        self._proc: Optional["subprocess.Popen[bytes]"] = None

    def start(self) -> InferenceEndpoint:
        """Start the server and block until it is healthy."""
        if self._proc is not None:
            return self.endpoint

        cmd = build_serve_command(
            self.model_path, self.serving, lora_adapter=self.lora_adapter
        )
        logger.info("starting vllm serve: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=build_serve_env(self.serving),
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )
        elapsed = wait_until_healthy(
            self._proc, self.serving.port, timeout=self.serving.startup_timeout
        )
        logger.info(
            "vllm serve healthy after %.1fs; endpoint %s", elapsed, self.base_url
        )
        return self.endpoint

    def stop(self) -> None:
        """Stop the server. Safe to call more than once."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        logger.info("stopping vllm serve (PID %s)", proc.pid)
        terminate_process_group(proc)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.serving.port}/v1"

    @property
    def endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(
            base_url=self.base_url,
            api_key=LOCAL_API_KEY,
            model_id=self.served_model_name,
            model_name=self.model_name,
        )

    def __enter__(self) -> InferenceEndpoint:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


__all__ = ["LOCAL_API_KEY", "VllmServerEndpoint"]
