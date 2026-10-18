"""Resilient call wrapper with exponential backoff + auth-refresh.

Ports the proven retry loop from ``tasks/advice/trainer.py`` ``_build_judge_call``:
retry up to ``max_retries`` times with exponential backoff, and call the client
manager's ``force_refresh()`` first when the error looks auth-related (expired
ticket / 401 / unauthorized). Unlike the advice reward, this raises the final
exception instead of returning a neutral fallback -- callers decide the fallback
(generation records an error row; LLM judges score 1).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_AUTH_ERROR_KEYWORDS = (
    "authentication",
    "expired",
    "ticket",
    "unauthorized",
    "401",
    "403",
    "forbidden",
)


class _Refreshable(Protocol):
    def force_refresh(self) -> object: ...


def _is_auth_error(message: str) -> bool:
    lowered = message.lower()
    return any(kw in lowered for kw in _AUTH_ERROR_KEYWORDS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    manager: Optional[_Refreshable] = None,
    max_retries: int = 10,
    base_delay: float = 3.0,
    description: str = "call",
) -> T:
    """Run ``fn`` with retries + exponential backoff; re-raise on exhaustion.

    Args:
        fn: Zero-arg callable performing one attempt.
        manager: Optional client manager with ``force_refresh()``; invoked
            before a retry when the error looks auth-related.
        max_retries: Total attempts.
        base_delay: Base seconds; delay for attempt ``i`` is ``base_delay * 2**i``.
        description: Label used in log messages.

    Returns:
        The value returned by ``fn``.

    Raises:
        Exception: The last exception raised by ``fn`` if all attempts fail.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transport may raise anything
            last_exc = exc
            error_str = str(exc)
            if attempt < max_retries - 1:
                if _is_auth_error(error_str) and manager is not None:
                    try:
                        manager.force_refresh()
                    except Exception:  # noqa: BLE001 - refresh best-effort
                        logger.warning("force_refresh() failed during retry")
                delay = base_delay * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d), retrying in %.0fs: %s",
                    description,
                    attempt + 1,
                    max_retries,
                    delay,
                    error_str[:200],
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "%s failed after %d attempts: %s",
                    description,
                    max_retries,
                    error_str[:200],
                )
    assert last_exc is not None
    raise last_exc


__all__ = ["call_with_retry"]
