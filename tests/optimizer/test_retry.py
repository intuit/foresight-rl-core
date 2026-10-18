"""call_with_retry: backoff, auth-refresh, and exhaustion behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from foresightrlcore.common.retry import call_with_retry


class _FakeManager:
    def __init__(self) -> None:
        self.refreshed = 0

    def force_refresh(self) -> object:
        self.refreshed += 1
        return object()


def test_returns_immediately_on_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    with patch("time.sleep") as sleep:
        assert call_with_retry(fn, max_retries=5, base_delay=0.0) == "ok"
    assert calls["n"] == 1
    sleep.assert_not_called()


def test_retries_then_succeeds():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "done"

    with patch("time.sleep"):
        assert call_with_retry(fn, max_retries=5, base_delay=0.0) == "done"
    assert attempts["n"] == 3


def test_auth_error_triggers_manager_refresh():
    mgr = _FakeManager()
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("401 Unauthorized: ticket expired")
        return "ok"

    with patch("time.sleep"):
        assert call_with_retry(fn, manager=mgr, max_retries=3, base_delay=0.0) == "ok"
    assert mgr.refreshed == 1


def test_non_auth_error_does_not_refresh():
    mgr = _FakeManager()

    def fn():
        raise ValueError("some other failure")

    with patch("time.sleep"):
        with pytest.raises(ValueError):
            call_with_retry(fn, manager=mgr, max_retries=2, base_delay=0.0)
    assert mgr.refreshed == 0


def test_raises_last_exception_on_exhaustion():
    def fn():
        raise RuntimeError("always fails")

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="always fails"):
            call_with_retry(fn, max_retries=3, base_delay=0.0)


def test_exponential_backoff_delays():
    def fn():
        raise RuntimeError("boom")

    with patch("time.sleep") as sleep:
        with pytest.raises(RuntimeError):
            call_with_retry(fn, max_retries=3, base_delay=1.0)
    # delays for attempts 0 and 1 (last attempt does not sleep): 1*2^0, 1*2^1
    delays = [c.args[0] for c in sleep.call_args_list]
    assert delays == [1.0, 2.0]


def test_force_refresh_failure_is_swallowed():
    class _BrokenManager:
        def force_refresh(self) -> object:
            raise RuntimeError("refresh boom")

    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("401 unauthorized")
        return "ok"

    # The manager's force_refresh raises, but the retry loop must swallow it
    # (best-effort) and still succeed on the next attempt.
    with patch("time.sleep"):
        result = call_with_retry(
            fn, manager=_BrokenManager(), max_retries=3, base_delay=0.0
        )
    assert result == "ok"
