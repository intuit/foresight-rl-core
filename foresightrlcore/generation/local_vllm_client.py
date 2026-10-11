"""Local-serving generation client: ``vllm serve`` + OpenAI-compatible calls.

Reuses the perf_test vLLM lifecycle pattern (subprocess + ``/health`` poll,
SIGTERM on the process group at shutdown), then talks to the server with
``langchain_openai.ChatOpenAI(base_url="http://localhost:PORT/v1")``. Only one
local model is resident at a time -- the runner brackets a local candidate's
batch with :meth:`start` / :meth:`close`.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, List, Optional, Tuple

from foresightrlcore.common.vllm_process import (
    DEFAULT_SERVED_NAME as _DEFAULT_SERVED_NAME,
)
from foresightrlcore.common.vllm_process import (
    build_serve_command,
    build_serve_env,
)
from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.generation.interfaces import CallUsage
from foresightrlcore.generation.langchain_client import (
    LangChainChatGenerationClient,
)

logger = logging.getLogger(__name__)


class _LocalChatManager:
    """Manager surface (``.get()`` / ``.force_refresh()``) over a local endpoint."""

    def __init__(self, port: int, served_model_name: str) -> None:
        self._port = port
        self._served_model_name = served_model_name
        self._client: Any = None

    def get(self) -> Any:
        if self._client is None:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self._served_model_name,
                temperature=0.0,
                api_key="EMPTY",
                base_url=f"http://localhost:{self._port}/v1",
            )
        return self._client

    def force_refresh(self) -> Any:
        self._client = None
        return self.get()


class LocalVllmGenerationClient:
    """Owns a ``vllm serve`` subprocess and generates against it.

    Args:
        model_path: Local path / HF id / S3-staged dir of the model to serve.
        serving: vLLM serving parameters.
        max_retries: Retry budget per generation call.
        base_delay: Base backoff seconds.
        truncate_prompt_tokens: If set, forwarded to vLLM as
            ``truncate_prompt_tokens`` so the server truncates each prompt to
            the last N tokens before generating. ``None`` (default) leaves
            prompts untouched.
    """

    def __init__(
        self,
        model_path: str,
        serving: VllmServingConfig,
        *,
        max_retries: int = 10,
        base_delay: float = 3.0,
        truncate_prompt_tokens: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.serving = serving
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self._proc: Optional[subprocess.Popen[bytes]] = None
        served_name = serving.served_model_name or _DEFAULT_SERVED_NAME
        self._manager = _LocalChatManager(serving.port, served_name)
        extra_body: Optional[dict] = (
            {"truncate_prompt_tokens": truncate_prompt_tokens}
            if truncate_prompt_tokens is not None
            else None
        )
        self._delegate = LangChainChatGenerationClient(
            self._manager,
            max_retries=max_retries,
            base_delay=base_delay,
            extra_body=extra_body,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> float:
        """Start ``vllm serve`` and block until ``/health`` returns 200."""
        cmd = self._build_command()
        env = self._build_env()
        logger.info("Starting vllm serve: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )

        from foresightrlcore.perf_test.runner import _vllm_health_ok

        start = time.monotonic()
        deadline = start + self.serving.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self._proc.returncode} "
                    f"before becoming healthy"
                )
            if _vllm_health_ok(self.serving.port):
                elapsed = time.monotonic() - start
                logger.info("vllm serve healthy after %.1fs", elapsed)
                return elapsed
            time.sleep(5)
        raise RuntimeError(
            f"vllm serve did not become healthy within "
            f"{self.serving.startup_timeout}s"
        )

    def _build_command(self) -> List[str]:
        return build_serve_command(self.model_path, self.serving)

    def _build_env(self) -> dict:
        return build_serve_env(self.serving)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self, prompt: str, *, tag: str, max_tokens: int
    ) -> Tuple[str, CallUsage]:
        return self._delegate.generate(prompt, tag=tag, max_tokens=max_tokens)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            logger.info("vllm serve already exited with code %s", proc.returncode)
            self._proc = None
            return
        logger.info("Stopping vllm serve (PID %s)", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("Could not signal vllm serve cleanly: %s", exc)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("vllm serve did not exit in 30s; sending SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._proc = None


__all__ = ["LocalVllmGenerationClient"]
