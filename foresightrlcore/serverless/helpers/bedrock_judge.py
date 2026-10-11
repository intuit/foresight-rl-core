"""Generic ``BedrockJudge`` — boto3 ``converse`` wrapper with retry policy.

Used by serverless reward Lambdas that score model outputs with an
Anthropic / Mistral / Meta model on Amazon Bedrock. Extracted from the
inlined transport in the advice Lambda so other tasks can reuse the
same retry + transient-error policy without copy-pasting.

Stdlib + boto3 only. Safe to import inside a Lambda zip (no
training-stack dependencies).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


DEFAULT_TRANSIENT_BEDROCK_ERRORS: FrozenSet[str] = frozenset(
    {
        "ThrottlingException",
        "ServiceQuotaExceededException",
        "ModelErrorException",
        "ModelTimeoutException",
        "InternalServerException",
        "ServiceUnavailableException",
    }
)


class JudgeError(RuntimeError):
    """Raised when a Bedrock judge call exhausts retries."""


@dataclass
class BedrockJudge:
    """Thread-safe, process-shared client wrapper around Bedrock ``converse``.

    Parameters
    ----------
    model_id:
        Bedrock model identifier (e.g. ``us.anthropic.claude-opus-4-5-...``).
    region:
        AWS region for the Bedrock runtime client.
    max_tokens, temperature:
        Forwarded to ``inferenceConfig``.
    max_retries, base_delay_s, max_delay_s:
        Exponential-backoff retry policy on transient errors.
    transient_error_codes:
        Bedrock ``ClientError`` codes treated as retryable. Defaults to
        :data:`DEFAULT_TRANSIENT_BEDROCK_ERRORS`.
    boto_config:
        Optional explicit ``botocore.config.Config``. Defaults to
        ``retries={mode=adaptive, max_attempts=3}``, 120s read, 10s connect.
    """

    model_id: str
    region: str
    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 10
    base_delay_s: float = 3.0
    max_delay_s: float = 60.0
    transient_error_codes: FrozenSet[str] = DEFAULT_TRANSIENT_BEDROCK_ERRORS
    boto_config: Optional[Config] = None

    _client: Any = None
    _client_lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # ``dataclass`` makes the lock attribute a class default; we need
        # one per-instance for true thread-safety.
        object.__setattr__(self, "_client_lock", threading.Lock())

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    config = self.boto_config or Config(
                        retries={"max_attempts": 3, "mode": "adaptive"},
                        read_timeout=120,
                        connect_timeout=10,
                    )
                    object.__setattr__(
                        self,
                        "_client",
                        boto3.client(
                            "bedrock-runtime",
                            region_name=self.region,
                            config=config,
                        ),
                    )
        return self._client

    def reset_client(self) -> None:
        """Drop the cached boto3 client. Mostly for tests."""
        with self._client_lock:
            object.__setattr__(self, "_client", None)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> str:
        """Call ``converse`` with retry, return the assistant text payload.

        Raises :class:`JudgeError` when all retries are exhausted.
        """
        client = self._get_client()
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"text": user_prompt}]},
        ]
        if extra_messages:
            messages.extend(extra_messages)

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                response = client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                )
                text = _extract_text_from_converse(response)
                if not text:
                    raise JudgeError("Bedrock converse returned empty text")
                return text
            except ClientError as exc:
                last_err = exc
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in self.transient_error_codes:
                    logger.warning(
                        "Bedrock judge call failed with non-transient %s: %s",
                        code,
                        exc,
                    )
                    break
                logger.warning(
                    "Bedrock judge attempt %d/%d failed (%s); will retry",
                    attempt + 1,
                    self.max_retries,
                    code,
                )
            except JudgeError as exc:
                last_err = exc
                logger.warning(
                    "Bedrock judge attempt %d/%d returned bad payload: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

            if attempt == self.max_retries - 1:
                break
            delay = self.base_delay_s * (2**attempt) * (0.5 + random.random())
            time.sleep(min(delay, self.max_delay_s))

        err_name = type(last_err).__name__ if last_err is not None else "unknown"
        err_msg = str(last_err)[:200] if last_err is not None else ""
        raise JudgeError(f"judge_failed: {err_name}: {err_msg}") from last_err


def _extract_text_from_converse(response: Dict[str, Any]) -> str:
    """Pull the assistant text out of a Bedrock ``converse`` response."""
    output = response.get("output", {})
    message = output.get("message", {}) if isinstance(output, dict) else {}
    content = message.get("content", []) if isinstance(message, dict) else []
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)
