"""Tests for foresightrlcore.runtime.scripts.patch_trl_vllm.

The script monkey-patches TRL's installed ``vllm_serve.py``. To avoid touching
the real TRL install, these tests redirect the patcher's ``find_vllm_serve``
to a tempfile that simulates TRL's source layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foresightrlcore.runtime.scripts import patch_trl_vllm

# A minimal stand-in for TRL's vllm_serve.py that contains the marker.
FAKE_TRL_SOURCE_WITH_MARKER = """\
from vllm import LLM


def serve():
    llm = LLM(
        model=script_args.model,
        tensor_parallel_size=script_args.vllm_tensor_parallel_size,
        model_impl=script_args.vllm_model_impl,
        gpu_memory_utilization=script_args.vllm_gpu_memory_utilization,
    )
    return llm
"""

# A stand-in without the marker (simulates TRL upstream rewriting vllm_serve.py).
FAKE_TRL_SOURCE_WITHOUT_MARKER = """\
from vllm import LLM


def serve():
    llm = LLM(
        model=script_args.model,
        tensor_parallel_size=script_args.vllm_tensor_parallel_size,
    )
    return llm
"""


@pytest.fixture
def fake_vllm_serve(tmp_path, monkeypatch):
    """Write a fake vllm_serve.py and redirect find_vllm_serve to it."""

    def _make(source: str) -> Path:
        path = tmp_path / "vllm_serve.py"
        path.write_text(source)
        monkeypatch.setattr(patch_trl_vllm, "find_vllm_serve", lambda: path)
        return path

    return _make


class TestPatchNoOpPaths:
    """No-opt-in / no-op call paths."""

    def test_no_args_returns_zero(self, fake_vllm_serve):
        fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        assert rc == 0

    def test_only_reasoning_start_is_ignored(self, fake_vllm_serve):
        """Need BOTH start and end strings; passing just one is treated as no-op."""
        path = fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start="<think>",
            reasoning_end=None,
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        assert rc == 0
        assert "reasoning_config=" not in path.read_text()


class TestPatchSuccessPaths:
    """Happy-path injections produce the expected edits."""

    def test_reasoning_injection(self, fake_vllm_serve):
        path = fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start="<think>",
            reasoning_end="</think>",
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        assert rc == 0
        text = path.read_text()
        assert "from vllm.config import ReasoningConfig" in text
        assert (
            'reasoning_config=ReasoningConfig(reasoning_start_str="<think>", '
            'reasoning_end_str="</think>")'
        ) in text

    def test_compilation_injection(self, fake_vllm_serve):
        path = fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
            disable_custom_all_reduce=False,
        )
        assert rc == 0
        text = path.read_text()
        assert "from vllm.config import CompilationConfig" in text
        assert "compilation_config=CompilationConfig(" in text
        assert "cudagraph_capture_sizes=[1, 2, 4]" in text

    def test_disable_all_reduce_injection(self, fake_vllm_serve):
        path = fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config=None,
            disable_custom_all_reduce=True,
        )
        assert rc == 0
        assert "disable_custom_all_reduce=True" in path.read_text()


class TestPatchIdempotent:
    """Re-running the patcher must not duplicate injections."""

    def test_reapply_is_noop(self, fake_vllm_serve):
        path = fake_vllm_serve(FAKE_TRL_SOURCE_WITH_MARKER)
        rc1 = patch_trl_vllm.patch(
            reasoning_start="<think>",
            reasoning_end="</think>",
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        first = path.read_text()
        rc2 = patch_trl_vllm.patch(
            reasoning_start="<think>",
            reasoning_end="</think>",
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        second = path.read_text()
        assert rc1 == 0 and rc2 == 0
        assert first == second
        # Exactly one reasoning_config insertion.
        assert first.count("reasoning_config=") == 1


class TestPatchFailFast:
    """Marker-missing must fail non-zero when the user opted in."""

    def test_missing_marker_with_opt_in_fails(self, fake_vllm_serve):
        fake_vllm_serve(FAKE_TRL_SOURCE_WITHOUT_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start="<think>",
            reasoning_end="</think>",
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        assert rc != 0

    def test_missing_marker_without_opt_in_is_noop(self, fake_vllm_serve):
        fake_vllm_serve(FAKE_TRL_SOURCE_WITHOUT_MARKER)
        rc = patch_trl_vllm.patch(
            reasoning_start=None,
            reasoning_end=None,
            compilation_config=None,
            disable_custom_all_reduce=False,
        )
        assert rc == 0
