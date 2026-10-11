"""Tests for foresightrlcore.runtime.scripts.vllm_launcher.

The launcher's job is process-startup hygiene: applying spawn-safety env
defaults and calling ``multiprocessing.set_start_method("spawn", force=True)``
before TRL imports ``torch``/``vllm``. These tests exercise that contract
without actually starting a vLLM server.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock

import pytest

from foresightrlcore.runtime.scripts import vllm_launcher


class TestApplyEnvDefaults:
    """``apply_env_defaults`` must be additive — never overwrite operator values."""

    def test_sets_all_defaults_when_env_is_empty(self) -> None:
        env: dict[str, str] = {}
        result = vllm_launcher.apply_env_defaults(env)
        assert result is env
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert env["CUDA_MODULE_LOADING"] == "LAZY"
        assert env["VLLM_USE_V1"] == "0"
        assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"

    def test_preserves_operator_overrides(self) -> None:
        env = {
            "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            "VLLM_USE_V1": "1",
        }
        vllm_launcher.apply_env_defaults(env)
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
        assert env["VLLM_USE_V1"] == "1"
        assert env["CUDA_MODULE_LOADING"] == "LAZY"
        assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"

    def test_custom_defaults_argument(self) -> None:
        env: dict[str, str] = {}
        vllm_launcher.apply_env_defaults(env, defaults={"FOO": "bar"})
        assert env == {"FOO": "bar"}

    def test_default_targets_real_environ_when_env_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        vllm_launcher.apply_env_defaults()
        import os

        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


class TestMain:
    """``main`` must apply hygiene before importing TRL, then forward argv."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strip launcher-managed env vars so tests start from a clean slate."""
        for key in (
            "VLLM_WORKER_MULTIPROC_METHOD",
            "CUDA_MODULE_LOADING",
            "VLLM_USE_V1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION",
            "VLLM_REASONING_CONFIG_JSON",
            "VLLM_COMPILATION_CONFIG_JSON",
            "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_calls_set_start_method_before_trl_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``set_start_method`` must run before any ``from trl....`` happens.

        We assert this by stubbing both, recording call order, and checking
        ``set_start_method`` ran first.
        """
        order: list[str] = []

        def fake_set_start_method(method: str, force: bool = False) -> None:
            order.append(f"set_start_method({method},force={force})")

        fake_trl = mock.MagicMock()
        fake_trl.main = mock.MagicMock(return_value=0)

        def fake_make_parser() -> Any:
            order.append("trl_imported")
            parser = mock.MagicMock()
            parser.parse_args_and_config.return_value = (mock.MagicMock(),)
            return parser

        fake_trl.make_parser = fake_make_parser

        monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", fake_trl)
        monkeypatch.setattr(
            vllm_launcher.multiprocessing,
            "set_start_method",
            fake_set_start_method,
        )

        rc = vllm_launcher.main(["--model", "fake"])
        assert rc == 0
        assert order[0] == "set_start_method(spawn,force=True)"
        assert "trl_imported" in order
        assert order.index("set_start_method(spawn,force=True)") < order.index(
            "trl_imported"
        )

    def test_forwards_argv_to_trl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """sys.argv tail (everything after the script name) must reach TRL."""
        captured: dict[str, Any] = {}

        fake_trl = mock.MagicMock()

        def fake_make_parser() -> Any:
            parser = mock.MagicMock()

            def parse() -> tuple[Any, ...]:
                captured["argv"] = list(sys.argv)
                return (mock.MagicMock(),)

            parser.parse_args_and_config = parse
            return parser

        fake_trl.make_parser = fake_make_parser
        fake_trl.main = mock.MagicMock(return_value=0)
        monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", fake_trl)
        monkeypatch.setattr(
            vllm_launcher.multiprocessing,
            "set_start_method",
            lambda *a, **kw: None,
        )

        argv = ["--model", "X", "--tensor-parallel-size", "4"]
        vllm_launcher.main(argv)
        # argv[0] is preserved (test runner script name); the tail is our argv.
        assert captured["argv"][1:] == argv

    def test_returns_zero_when_trl_main_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TRL's ``main`` historically returns ``None``; we coerce to 0."""
        fake_trl = mock.MagicMock()
        fake_trl.main = mock.MagicMock(return_value=None)
        parser = mock.MagicMock()
        parser.parse_args_and_config.return_value = (mock.MagicMock(),)
        fake_trl.make_parser = mock.MagicMock(return_value=parser)
        monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", fake_trl)
        monkeypatch.setattr(
            vllm_launcher.multiprocessing,
            "set_start_method",
            lambda *a, **kw: None,
        )

        assert vllm_launcher.main(["--model", "x"]) == 0

    def test_propagates_nonzero_exit_from_trl_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_trl = mock.MagicMock()
        fake_trl.main = mock.MagicMock(return_value=7)
        parser = mock.MagicMock()
        parser.parse_args_and_config.return_value = (mock.MagicMock(),)
        fake_trl.make_parser = mock.MagicMock(return_value=parser)
        monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", fake_trl)
        monkeypatch.setattr(
            vllm_launcher.multiprocessing,
            "set_start_method",
            lambda *a, **kw: None,
        )

        assert vllm_launcher.main(["--model", "x"]) == 7


class TestApplyTrlPatchFromEnv:
    """The launcher forwards env-driven opt-ins to ``patch_trl_vllm.patch``."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "VLLM_REASONING_CONFIG_JSON",
            "VLLM_COMPILATION_CONFIG_JSON",
            "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_noop_when_no_env_vars_set(self) -> None:
        """No opt-ins → patch is never invoked; return 0."""
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        with mock.patch.object(patch_trl_vllm, "patch") as patched:
            rc = vllm_launcher.apply_trl_patch_from_env()
        assert rc == 0
        patched.assert_not_called()

    def test_forwards_reasoning_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        monkeypatch.setenv(
            "VLLM_REASONING_CONFIG_JSON",
            '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
        )
        with mock.patch.object(patch_trl_vllm, "patch", return_value=0) as patched:
            assert vllm_launcher.apply_trl_patch_from_env() == 0
        patched.assert_called_once_with(
            reasoning_start="<think>",
            reasoning_end="</think>",
            compilation_config=None,
            disable_custom_all_reduce=False,
        )

    def test_forwards_compilation_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        monkeypatch.setenv(
            "VLLM_COMPILATION_CONFIG_JSON",
            '{"cudagraph_capture_sizes":[1,2,4]}',
        )
        with mock.patch.object(patch_trl_vllm, "patch", return_value=0) as patched:
            assert vllm_launcher.apply_trl_patch_from_env() == 0
        patched.assert_called_once_with(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
            disable_custom_all_reduce=False,
        )

    def test_forwards_disable_custom_all_reduce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        monkeypatch.setenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
        with mock.patch.object(patch_trl_vllm, "patch", return_value=0) as patched:
            assert vllm_launcher.apply_trl_patch_from_env() == 0
        patched.assert_called_once_with(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config=None,
            disable_custom_all_reduce=True,
        )

    def test_propagates_nonzero_from_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        monkeypatch.setenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
        with mock.patch.object(patch_trl_vllm, "patch", return_value=3):
            assert vllm_launcher.apply_trl_patch_from_env() == 3

    def test_malformed_json_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_REASONING_CONFIG_JSON", "{not-json")
        with pytest.raises(SystemExit) as exc_info:
            vllm_launcher.apply_trl_patch_from_env()
        assert exc_info.value.code == 2

    def test_json_must_be_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_COMPILATION_CONFIG_JSON", '["a","b"]')
        with pytest.raises(SystemExit) as exc_info:
            vllm_launcher.apply_trl_patch_from_env()
        assert exc_info.value.code == 2

    def test_reasoning_config_missing_required_keys_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "VLLM_REASONING_CONFIG_JSON", '{"reasoning_start_str":"<think>"}'
        )
        rc = vllm_launcher.apply_trl_patch_from_env()
        assert rc == 2

    def test_disable_value_other_than_1_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the literal string '1' opts in; '0', 'true', '' do not."""
        from foresightrlcore.runtime.scripts import patch_trl_vllm

        monkeypatch.setenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "true")
        with mock.patch.object(patch_trl_vllm, "patch") as patched:
            rc = vllm_launcher.apply_trl_patch_from_env()
        assert rc == 0
        patched.assert_not_called()


class TestMainAppliesPatchBeforeTrlImport:
    """``main`` runs the env-driven patch BEFORE importing TRL."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "VLLM_WORKER_MULTIPROC_METHOD",
            "CUDA_MODULE_LOADING",
            "VLLM_USE_V1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION",
            "VLLM_REASONING_CONFIG_JSON",
            "VLLM_COMPILATION_CONFIG_JSON",
            "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_main_aborts_when_patch_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the patch returns non-zero, ``main`` must NOT import TRL."""
        monkeypatch.setenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
        monkeypatch.setattr(
            vllm_launcher.multiprocessing,
            "set_start_method",
            lambda *a, **kw: None,
        )

        # If TRL is imported, this fake will be reached; the test asserts it
        # is NOT — by leaving the fake without a return_value the test would
        # blow up if main ever calls trl_main.
        fake_trl = mock.MagicMock()
        fake_trl.main = mock.MagicMock(
            side_effect=AssertionError("TRL must not be imported after a patch failure")
        )
        monkeypatch.setitem(sys.modules, "trl.scripts.vllm_serve", fake_trl)

        from foresightrlcore.runtime.scripts import patch_trl_vllm

        with mock.patch.object(patch_trl_vllm, "patch", return_value=3):
            rc = vllm_launcher.main(["--model", "x"])
        assert rc == 3
        fake_trl.main.assert_not_called()
