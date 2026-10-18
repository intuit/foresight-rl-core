"""Tests for foresightrlcore.methods.sft_verl — verl FSDP SFT method handler.

validate()/registration/helpers are exercised directly. run() itself needs
the verl/hydra GPU stack, which is intentionally not installed in the
lint/test environment (see foresightrlcore/methods/verl_sft_dataset.py
docstring) — but hydra's role here is limited to composing a plain
OmegaConf DictConfig (omegaconf itself IS a real dependency already), so
run()'s actual field-mapping logic is exercised in TestRun by faking
``hydra`` in ``sys.modules`` (same technique as
tests/methods/test_verl_sft_dataset.py's fake ``verl.utils.dataset.*``) and
stubbing out only the two pieces that need real GPU-stack packages:
``_ensure_flash_attn_compatible`` (real subprocess calls, tested on its own
in TestEnsureFlashAttnCompatible instead) and ``_run_verl_sft`` (needs the
real ``verl.trainer.sft_trainer.SFTTrainer``, tested on its own in
TestRunVerlSft by faking just that one import).
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.methods import get_method
from foresightrlcore.methods.sft_verl import (
    VerlSftMethodHandler,
    _cache_jsonl_as_parquet,
    _check_packed_data_matches_config,
    _ensure_flash_attn_compatible,
    _ensure_transformers_version_compatible,
    _glob_data_files,
    _torchrun_launch_id,
)


def _sft_verl_config_dict(**overrides: dict) -> dict:
    # Generic fake model name (matching tests/methods/test_trl_method_handler.py's
    # own "m"/"test-model" convention) — this handler is architecture-
    # agnostic; only a handful of dedicated tests below actually need a
    # real, profile-bearing model name (see TestRun's gemma4_moe profile
    # test), and using a real one as the *default* fixture for all ~90
    # generic tests obscured that.
    base = {
        "job": {"name": "sft-verl-test"},
        "method": {"type": "sft_verl"},
        "infra": {
            "vllm_mode": "none",
            "instance_type": "ml.p4de.24xlarge",
            "instance_count": 2,
        },
        "model": {"name": "org/fake-model"},
        "training": {
            "sft_verl": {
                "max_length": 18432,
                "ulysses_sequence_parallel_size": 4,
                "use_remove_padding": True,
            }
        },
        "data": {"train_path": "s3://bucket/data/train"},
        "output": {
            "s3_path": "s3://bucket/output",
            "checkpoint_s3_path": "s3://bucket/checkpoints",
        },
    }
    for key, value in overrides.items():
        base[key] = value
    return base


class TestVerlSftMethodHandler:
    def test_method_type_attribute(self):
        handler = VerlSftMethodHandler()
        assert handler.method_type == "sft_verl"

    def test_registered_under_sft_verl(self):
        import foresightrlcore.methods.sft_verl  # noqa: F401

        assert get_method("sft_verl") is VerlSftMethodHandler

    def test_validate_valid_config(self, tmp_yaml):
        content = yaml.dump(_sft_verl_config_dict())
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.validate(config) == []

    def test_validate_missing_sft_verl_section(self, tmp_yaml):
        d = _sft_verl_config_dict()
        del d["training"]
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("training.sft_verl" in e for e in errors)

    def test_validate_returns_error_when_training_section_is_none(self):
        """Not reachable via real YAML parsing (TrainingConfig always

        default-constructs), so exercised directly with a duck-typed stand-in
        — validate() only ever does getattr(config, "training", None).
        """
        handler = VerlSftMethodHandler()
        errors = handler.validate(SimpleNamespace(training=None))
        assert any("Training config section is missing" in e for e in errors)

    def test_validate_invalid_fsdp_strategy(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["fsdp_strategy"] = "bogus"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("fsdp_strategy" in e for e in errors)

    def test_validate_rejects_unsupported_lr_scheduler(self, tmp_yaml):
        """verl's FSDPOptimizerConfig hard-asserts lr_scheduler_type in

        ["constant", "cosine"] — anything else (e.g. "linear") must be
        rejected here rather than silently trained as "constant" by
        run()'s mapping.
        """
        d = _sft_verl_config_dict()
        d["training"]["lr_scheduler"] = "linear"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("lr_scheduler" in e for e in errors)

    def test_validate_accepts_constant_lr_scheduler(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["lr_scheduler"] = "constant"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert not any("lr_scheduler" in e for e in errors)

    def test_validate_unrecognized_gpu_family(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["infra"]["instance_type"] = "ml.t2.micro"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("Unrecognized GPU family" in e for e in errors)

    def test_validate_requires_vllm_mode_none(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["infra"]["vllm_mode"] = "server"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("vllm_mode: none" in e for e in errors)

    def test_validate_ulysses_requires_remove_padding(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 4
        d["training"]["sft_verl"]["use_remove_padding"] = False
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("use_remove_padding" in e for e in errors)

    def test_validate_remove_padding_requires_flash_attention_2(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = True
        d["model"]["attn_implementation"] = "sdpa"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("attn_implementation" in e for e in errors)

    def test_validate_requires_train_path(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["data"] = {}
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("data.train_path" in e for e in errors)

    def test_validate_requires_checkpoint_s3_path_for_training(self, tmp_yaml):
        """verl's SFTTrainer only ever writes checkpoints to

        output.checkpoint_s3_path (via SageMaker's CheckpointConfig sync)
        — it has no separate "save final model" step, so without this
        field a training job "succeeds" with zero durable output anywhere.
        """
        d = _sft_verl_config_dict()
        d["output"] = {"s3_path": "s3://bucket/output"}
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("output.checkpoint_s3_path" in e for e in errors)

    def test_build_trainer_args_is_empty_stub(self, tmp_yaml):
        content = yaml.dump(_sft_verl_config_dict())
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.build_trainer_args(config) == {}

    def test_validate_rejects_unknown_data_format(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["data_format"] = "parquet_columnar"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("data_format" in e for e in errors)

    def test_validate_accepts_messages_data_format(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["data_format"] = "messages"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.validate(config) == []

    def test_validate_merge_lora_only_requires_checkpoint_s3_path(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["merge_lora_only"] = True
        del d["output"]["checkpoint_s3_path"]
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("checkpoint_s3_path" in e for e in errors)

    def test_validate_merge_lora_only_passes_with_checkpoint_s3_path(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["merge_lora_only"] = True
        d["output"]["checkpoint_s3_path"] = "s3://bucket/checkpoints"
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.validate(config) == []

    def test_validate_merge_lora_only_skips_train_path_requirement(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["merge_lora_only"] = True
        d["output"]["checkpoint_s3_path"] = "s3://bucket/checkpoints"
        d["data"] = {}
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.validate(config) == []

    def test_merge_lora_only_and_merge_checkpoint_step_default(self, tmp_yaml):
        content = yaml.dump(_sft_verl_config_dict())
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.training.sft_verl is not None
        assert config.training.sft_verl.merge_lora_only is False
        assert config.training.sft_verl.merge_checkpoint_step is None

    def test_save_lora_only_defaults_to_true(self, tmp_yaml):
        content = yaml.dump(_sft_verl_config_dict())
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.training.sft_verl is not None
        assert config.training.sft_verl.save_lora_only is True

    def test_save_lora_only_can_be_disabled(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["save_lora_only"] = False
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.training.sft_verl is not None
        assert config.training.sft_verl.save_lora_only is False

    def test_validate_accepts_packed_data_format(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["data_format"] = "packed"
        # packed data's correctness depends on position_ids resets, which
        # are only honored on the non-remove-padding path — see
        # test_validate_packed_requires_no_remove_padding below.
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = False
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        assert handler.validate(config) == []

    def test_validate_packed_requires_no_remove_padding(self, tmp_yaml):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["data_format"] = "packed"
        d["training"]["sft_verl"]["use_remove_padding"] = True
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = VerlSftMethodHandler()
        errors = handler.validate(config)
        assert any("data_format='packed'" in e for e in errors)


class TestGlobDataFiles:
    def test_globs_jsonl_in_directory(self, tmp_path):
        (tmp_path / "b.jsonl").write_text("{}\n")
        (tmp_path / "a.jsonl").write_text("{}\n")
        (tmp_path / "notes.txt").write_text("ignore me")

        files = _glob_data_files(str(tmp_path))
        assert [p.split("/")[-1] for p in files] == ["a.jsonl", "b.jsonl"]

    def test_raises_when_no_files_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _glob_data_files(str(empty_dir))

    def test_glob_pattern_path(self, tmp_path):
        (tmp_path / "shard-0.jsonl").write_text("{}\n")
        (tmp_path / "shard-1.jsonl").write_text("{}\n")
        files = _glob_data_files(str(tmp_path / "shard-*.jsonl"))
        assert len(files) == 2

    def test_globs_parquet_extension(self, tmp_path):
        (tmp_path / "training_8192.idx.parquet").write_bytes(b"x")
        (tmp_path / "notes.jsonl").write_text("{}\n")

        files = _glob_data_files(str(tmp_path), extension="parquet")
        assert [p.split("/")[-1] for p in files] == ["training_8192.idx.parquet"]

    def test_filename_prefix_disambiguates_train_from_eval_in_shared_dir(
        self, tmp_path
    ):
        # rl-train tokenize-pack's real output layout: both splits land in
        # the same flat directory.
        (tmp_path / "training_8192.idx.parquet").write_bytes(b"x")
        (tmp_path / "validation_8192.idx.parquet").write_bytes(b"y")

        train_files = _glob_data_files(
            str(tmp_path), extension="parquet", filename_prefix="training_"
        )
        val_files = _glob_data_files(
            str(tmp_path), extension="parquet", filename_prefix="validation_"
        )

        assert [p.split("/")[-1] for p in train_files] == ["training_8192.idx.parquet"]
        assert [p.split("/")[-1] for p in val_files] == ["validation_8192.idx.parquet"]


class TestTorchrunLaunchId:
    def test_matches_current_process_ppid(self):
        assert _torchrun_launch_id() == str(os.getppid())

    def test_stable_across_repeated_calls(self):
        # Same real torchrun launch (this test process) must always
        # compute the same id — this is exactly the property the sentinel
        # paths rely on to be identical across all local ranks.
        assert _torchrun_launch_id() == _torchrun_launch_id()


class TestEnsureFlashAttnCompatible:
    """The LOCAL_RANK==0-does-the-work / others-wait-on-a-sentinel pattern.

    Sentinel/failure paths are namespaced by ``_torchrun_launch_id()``
    (``os.getppid()``, stable for this whole test process) rather than
    hardcoded — computed here the same way the real function does, so
    every test cleans up the exact paths it actually touches.
    """

    SENTINEL = f"/tmp/flash_attn_fix.{_torchrun_launch_id()}.done"
    FAILURE = f"/tmp/flash_attn_fix.{_torchrun_launch_id()}.failed"

    def _cleanup(self):
        for p in (self.SENTINEL, self.FAILURE):
            if os.path.exists(p):
                os.remove(p)

    def test_rank0_cleans_up_stale_sentinel_and_failure_before_running(
        self, monkeypatch
    ):
        self._cleanup()
        # Simulate leftovers from a previous, failed attempt on this node.
        with open(self.SENTINEL, "w") as f:
            f.write("stale")
        with open(self.FAILURE, "w") as f:
            f.write("stale")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: MagicMock())
        try:
            _ensure_flash_attn_compatible("https://example.com/wheel.whl")
            assert os.path.exists(self.SENTINEL)
            assert not os.path.exists(self.FAILURE)
        finally:
            self._cleanup()

    def test_stale_sentinel_from_a_different_launch_is_not_trusted(self, monkeypatch):
        """Regression guard for the actual race Bugbot flagged: a /tmp

        sentinel left by an EARLIER, unrelated job on a reused container
        (SageMaker Managed Warm Pools, or a repeated local-GPU run) must
        never be mistaken for THIS run's own fix having completed. Writes
        a "done" sentinel at a DIFFERENT (fake) launch id's path — not
        self.SENTINEL, which is keyed by this test process's own real
        os.getppid() — so a non-zero rank waiting on the REAL path must
        still time out instead of proceeding immediately.
        """
        self._cleanup()
        foreign_sentinel = "/tmp/flash_attn_fix.999999999.done"
        with open(foreign_sentinel, "w") as f:
            f.write("stale from a different job")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("FLASH_ATTN_FIX_TIMEOUT", "0")
        monkeypatch.setattr("time.sleep", lambda s: None)
        try:
            with pytest.raises(TimeoutError):
                _ensure_flash_attn_compatible("https://example.com/wheel.whl")
        finally:
            self._cleanup()
            os.remove(foreign_sentinel)

    def test_rank0_success_writes_sentinel_and_calls_pip_correctly(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv("LOCAL_RANK", "0")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            _ensure_flash_attn_compatible("https://example.com/wheel.whl")
            assert os.path.exists(self.SENTINEL)
            assert not os.path.exists(self.FAILURE)
            assert calls[0][0] == ["pip", "uninstall", "-y", "flash-attn"]
            assert calls[1][0][-1] == "https://example.com/wheel.whl"
            assert calls[1][1]["check"] is True
        finally:
            self._cleanup()

    def test_rank0_failure_writes_failure_file_and_reraises(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv("LOCAL_RANK", "0")

        def fake_run(cmd, **kwargs):
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            with pytest.raises(subprocess.CalledProcessError):
                _ensure_flash_attn_compatible("https://example.com/wheel.whl")
            assert os.path.exists(self.FAILURE)
            assert not os.path.exists(self.SENTINEL)
        finally:
            self._cleanup()

    def test_non_rank0_proceeds_immediately_once_sentinel_exists(self, monkeypatch):
        self._cleanup()
        with open(self.SENTINEL, "w") as f:
            f.write("ok")
        monkeypatch.setenv("LOCAL_RANK", "1")
        try:
            _ensure_flash_attn_compatible("https://example.com/wheel.whl")
        finally:
            self._cleanup()

    def test_non_rank0_raises_when_failure_file_present(self, monkeypatch):
        self._cleanup()
        with open(self.FAILURE, "w") as f:
            f.write("boom")
        monkeypatch.setenv("LOCAL_RANK", "1")
        try:
            with pytest.raises(RuntimeError, match="rank0 failed"):
                _ensure_flash_attn_compatible("https://example.com/wheel.whl")
        finally:
            self._cleanup()

    def test_non_rank0_times_out_when_rank0_never_finishes(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("FLASH_ATTN_FIX_TIMEOUT", "0")
        monkeypatch.setattr("time.sleep", lambda s: None)
        try:
            with pytest.raises(TimeoutError):
                _ensure_flash_attn_compatible("https://example.com/wheel.whl")
        finally:
            self._cleanup()

    def test_non_rank0_sleeps_between_polls_then_succeeds(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("FLASH_ATTN_FIX_TIMEOUT", "600")

        def fake_sleep(seconds):
            # Simulate rank0 finishing while we were asleep.
            with open(self.SENTINEL, "w") as f:
                f.write("ok")

        monkeypatch.setattr("time.sleep", fake_sleep)
        try:
            _ensure_flash_attn_compatible("https://example.com/wheel.whl")
            assert os.path.exists(self.SENTINEL)
        finally:
            self._cleanup()


class TestEnsureTransformersVersionCompatible:
    """Same LOCAL_RANK==0-does-the-work / others-wait-on-a-sentinel pattern

    as TestEnsureFlashAttnCompatible above, plus an early-return no-op
    check (the version comparison itself, before any rank branching). The
    installed version is faked via ``importlib.metadata.version`` (not by
    injecting a fake module into ``sys.modules``, unlike other GPU-stack
    fakes in this file) — this function deliberately never does
    ``import transformers`` itself (see its docstring for why: doing so
    would cache the pre-swap package in ``sys.modules`` for the rest of
    this process, defeating the whole point of the swap), so patching
    ``sys.modules`` wouldn't even be exercising the real code path.
    """

    SENTINEL = f"/tmp/transformers_version_fix.{_torchrun_launch_id()}.done"
    FAILURE = f"/tmp/transformers_version_fix.{_torchrun_launch_id()}.failed"

    def _cleanup(self):
        for p in (self.SENTINEL, self.FAILURE):
            if os.path.exists(p):
                os.remove(p)

    def _install_fake_transformers(self, monkeypatch, version: str):
        import importlib.metadata as metadata_module

        monkeypatch.setattr(metadata_module, "version", lambda name: version)

    def test_already_satisfies_is_a_noop(self, monkeypatch):
        self._cleanup()
        # Explicit, not relying on the ambient-env default (rank0) — the
        # version check now runs inside the rank0 branch, so an unrelated
        # test elsewhere that leaks a non-zero LOCAL_RANK into os.environ
        # would otherwise send this down the wait-and-timeout path instead.
        monkeypatch.setenv("LOCAL_RANK", "0")
        self._install_fake_transformers(monkeypatch, "5.15.0")
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
        try:
            _ensure_transformers_version_compatible("5.15.0")
            assert calls == []
            # rank0 still writes the sentinel even when no swap was needed
            # — every other rank waits on this same file regardless of
            # which branch rank0 took, so it must always signal "done".
            assert os.path.exists(self.SENTINEL)
        finally:
            self._cleanup()

    def test_newer_than_required_is_also_a_noop(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv("LOCAL_RANK", "0")
        self._install_fake_transformers(monkeypatch, "5.17.0")
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
        try:
            _ensure_transformers_version_compatible("5.15.0")
            assert calls == []
        finally:
            self._cleanup()

    def test_rank0_cleans_up_stale_sentinel_and_failure_before_running(
        self, monkeypatch
    ):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        with open(self.SENTINEL, "w") as f:
            f.write("stale")
        with open(self.FAILURE, "w") as f:
            f.write("stale")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: MagicMock())
        try:
            _ensure_transformers_version_compatible("5.15.0")
            assert os.path.exists(self.SENTINEL)
            assert not os.path.exists(self.FAILURE)
        finally:
            self._cleanup()

    def test_rank0_success_writes_sentinel_and_calls_pip_correctly(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        monkeypatch.setenv("LOCAL_RANK", "0")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            _ensure_transformers_version_compatible("5.15.0")
            assert os.path.exists(self.SENTINEL)
            assert not os.path.exists(self.FAILURE)
            # No --no-deps: a newer transformers hard-requires a newer
            # tokenizers too, checked eagerly at import time — skipping
            # deps here would leave the old, now-incompatible tokenizers
            # in place. transformers' own real dependency set is small and
            # framework-agnostic (no torch), so letting pip resolve it
            # normally can't disturb the pinned CUDA/torch/flash-attn stack.
            assert calls[0][0] == [
                "pip",
                "install",
                "--no-cache-dir",
                "transformers>=5.15.0",
            ]
            assert calls[0][1]["check"] is True
        finally:
            self._cleanup()

    def test_rank0_failure_writes_failure_file_and_reraises(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        monkeypatch.setenv("LOCAL_RANK", "0")

        def fake_run(cmd, **kwargs):
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            with pytest.raises(subprocess.CalledProcessError):
                _ensure_transformers_version_compatible("5.15.0")
            assert os.path.exists(self.FAILURE)
            assert not os.path.exists(self.SENTINEL)
        finally:
            self._cleanup()

    def test_non_rank0_proceeds_immediately_once_sentinel_exists(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        with open(self.SENTINEL, "w") as f:
            f.write("ok")
        monkeypatch.setenv("LOCAL_RANK", "1")
        try:
            _ensure_transformers_version_compatible("5.15.0")
        finally:
            self._cleanup()

    def test_non_rank0_never_queries_installed_version(self, monkeypatch):
        """Regression guard for a real SageMaker failure: a non-rank0 rank

        that independently queries the installed version races rank0's own
        `pip uninstall` + `pip install` — landing in the window where the
        old dist-info has been removed but the new one isn't written yet
        raises importlib.metadata.PackageNotFoundError. Only rank0 may ever
        query or mutate the version; every other rank must just wait on the
        sentinel, never calling importlib.metadata.version itself.
        """
        self._cleanup()

        import importlib.metadata as metadata_module

        def _boom(name):
            raise AssertionError(
                "non-rank0 must never query the installed transformers version"
            )

        monkeypatch.setattr(metadata_module, "version", _boom)
        with open(self.SENTINEL, "w") as f:
            f.write("ok")
        monkeypatch.setenv("LOCAL_RANK", "1")
        try:
            _ensure_transformers_version_compatible("5.15.0")
        finally:
            self._cleanup()

    def test_non_rank0_raises_when_failure_file_present(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        with open(self.FAILURE, "w") as f:
            f.write("boom")
        monkeypatch.setenv("LOCAL_RANK", "1")
        try:
            with pytest.raises(RuntimeError, match="rank0 failed"):
                _ensure_transformers_version_compatible("5.15.0")
        finally:
            self._cleanup()

    def test_non_rank0_times_out_when_rank0_never_finishes(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("TRANSFORMERS_VERSION_FIX_TIMEOUT", "0")
        monkeypatch.setattr("time.sleep", lambda s: None)
        try:
            with pytest.raises(TimeoutError):
                _ensure_transformers_version_compatible("5.15.0")
        finally:
            self._cleanup()

    def test_non_rank0_sleeps_between_polls_then_succeeds(self, monkeypatch):
        self._cleanup()
        self._install_fake_transformers(monkeypatch, "5.10.4")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("TRANSFORMERS_VERSION_FIX_TIMEOUT", "600")

        def fake_sleep(seconds):
            with open(self.SENTINEL, "w") as f:
                f.write("ok")

        monkeypatch.setattr("time.sleep", fake_sleep)
        try:
            _ensure_transformers_version_compatible("5.15.0")
            assert os.path.exists(self.SENTINEL)
        finally:
            self._cleanup()


class TestCacheJsonlAsParquet:
    """The same LOCAL_RANK==0 / manifest-sentinel pattern, but keyed by a

    hash of the input file list (so, unlike flash-attn's fixed path, each
    test's unique tmp_path input naturally gets its own cache directory —
    still cleaned up explicitly since the cache dir itself lives under the
    hardcoded /tmp/verl_sft_parquet_cache/, not tmp_path).
    """

    def _cache_dir_for(self, files: list[str]) -> str:
        import hashlib

        key = hashlib.sha1("\n".join(sorted(files)).encode()).hexdigest()[:16]
        return f"/tmp/verl_sft_parquet_cache/{key}"

    def _cleanup(self, files: list[str]):
        import shutil

        shutil.rmtree(self._cache_dir_for(files), ignore_errors=True)

    def test_empty_files_list_passes_through_unchanged(self):
        assert _cache_jsonl_as_parquet([]) == []

    def test_rank0_converts_jsonl_to_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"a": 2}\n')
        files = [str(f)]
        try:
            result = _cache_jsonl_as_parquet(files)
            assert len(result) == 1
            assert result[0].endswith(".parquet")
            df = pd.read_parquet(result[0])
            assert list(df["a"]) == [1, 2]
        finally:
            self._cleanup(files)

    def test_rank0_reuses_cache_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n')
        files = [str(f)]
        try:
            first = _cache_jsonl_as_parquet(files)
            mtime_first = os.path.getmtime(first[0])
            second = _cache_jsonl_as_parquet(files)
            assert second == first
            # Idempotent: the parquet file is not rewritten on the second call.
            assert os.path.getmtime(second[0]) == mtime_first
        finally:
            self._cleanup(files)

    def test_rank0_failure_writes_failure_file_and_reraises(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LOCAL_RANK", "0")
        missing_file = str(tmp_path / "does-not-exist.jsonl")
        files = [missing_file]
        try:
            with pytest.raises(FileNotFoundError):
                _cache_jsonl_as_parquet(files)
            cache_dir = self._cache_dir_for(files)
            assert os.path.exists(
                os.path.join(cache_dir, f"manifest.failed.{_torchrun_launch_id()}")
            )
            assert not os.path.exists(os.path.join(cache_dir, "manifest.done"))
        finally:
            self._cleanup(files)

    def test_non_rank0_reads_manifest_once_sentinel_exists(self, tmp_path, monkeypatch):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n')
        files = [str(f)]
        try:
            monkeypatch.setenv("LOCAL_RANK", "0")
            rank0_result = _cache_jsonl_as_parquet(files)

            monkeypatch.setenv("LOCAL_RANK", "1")
            rank1_result = _cache_jsonl_as_parquet(files)
            assert rank1_result == rank0_result
        finally:
            self._cleanup(files)

    def test_non_rank0_raises_when_failure_manifest_present(
        self, tmp_path, monkeypatch
    ):
        f = tmp_path / "data.jsonl"
        files = [str(f)]  # f deliberately never created -> rank0 fails
        try:
            monkeypatch.setenv("LOCAL_RANK", "0")
            with pytest.raises(FileNotFoundError):
                _cache_jsonl_as_parquet(files)

            monkeypatch.setenv("LOCAL_RANK", "1")
            with pytest.raises(RuntimeError, match="rank0 failed"):
                _cache_jsonl_as_parquet(files)
        finally:
            self._cleanup(files)

    def test_non_rank0_ignores_stale_failure_from_a_different_launch(
        self, tmp_path, monkeypatch
    ):
        """Regression guard for the actual race Bugbot flagged: unlike

        the flash-attn/transformers sentinels (whose failure file is
        namespaced by _torchrun_launch_id), this cache's failure marker
        used to be a plain, un-namespaced "manifest.failed" — a leftover
        from an EARLIER, unrelated job's genuine failure on different data
        that happened to reuse this same SageMaker channel path (warm-pool
        container reuse) would abort a healthy new run's non-zero ranks
        before rank0 even got a chance to clean it up and rebuild. Writes
        a "failed" marker at a DIFFERENT (fake) launch id's path — not the
        real one, which is keyed by this test process's own os.getppid()
        — so a non-zero rank waiting on the REAL path must not treat it as
        this run's failure (must keep polling/time out, not immediately
        raise "rank0 failed").
        """
        files = [str(tmp_path / "data.jsonl")]
        cache_dir = self._cache_dir_for(files)
        os.makedirs(cache_dir, exist_ok=True)
        stale_failure = os.path.join(cache_dir, "manifest.failed.999999999")
        with open(stale_failure, "w") as ff:
            ff.write("stale failure from a different, unrelated job")

        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("PARQUET_CACHE_TIMEOUT", "0")
        monkeypatch.setattr("time.sleep", lambda s: None)
        try:
            with pytest.raises(TimeoutError):
                _cache_jsonl_as_parquet(files)
        finally:
            self._cleanup(files)

    def test_non_rank0_times_out_when_rank0_never_finishes(self, tmp_path, monkeypatch):
        files = [str(tmp_path / "never.jsonl")]
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("PARQUET_CACHE_TIMEOUT", "0")
        monkeypatch.setattr("time.sleep", lambda s: None)
        try:
            with pytest.raises(TimeoutError):
                _cache_jsonl_as_parquet(files)
        finally:
            self._cleanup(files)

    def test_non_rank0_sleeps_between_polls_then_reads_manifest(
        self, tmp_path, monkeypatch
    ):
        import json

        from foresightrlcore.methods.sft_verl import _expected_parquet_paths

        # Real, node-shared-filesystem file — non-zero ranks now also
        # os.stat() it (to verify a manifest's content matches THIS run's
        # files, not a stale one; see _expected_parquet_paths), so it must
        # actually exist here, unlike before that check was added.
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n')
        files = [str(f)]
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("PARQUET_CACHE_TIMEOUT", "600")
        cache_dir = self._cache_dir_for(files)
        expected = _expected_parquet_paths(files, cache_dir)

        def fake_sleep(seconds):
            # Simulate rank0 finishing its cache build while we were asleep.
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "manifest.json"), "w") as mf:
                json.dump(expected, mf)
            with open(os.path.join(cache_dir, "manifest.done"), "w") as sf:
                sf.write("ok")

        monkeypatch.setattr("time.sleep", fake_sleep)
        try:
            result = _cache_jsonl_as_parquet(files)
            assert result == expected
        finally:
            self._cleanup(files)

    def test_non_rank0_ignores_stale_manifest_from_a_different_job(
        self, tmp_path, monkeypatch
    ):
        """Regression guard for the actual race Bugbot flagged: this

        cache_dir is keyed only by file PATH (not content), and SageMaker
        mounts every input channel at the same fixed local path regardless
        of job/dataset — so on a reused container (warm pool), a
        completely different, unrelated job's manifest can already sit at
        this exact cache_dir. A stale manifest listing a DIFFERENT
        content-hash parquet path (i.e. not matching this run's real
        os.stat()-derived expectation) must never be trusted — only once
        rank0 writes the correct one should this return.
        """
        import json

        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n')
        files = [str(f)]
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("PARQUET_CACHE_TIMEOUT", "600")
        cache_dir = self._cache_dir_for(files)
        os.makedirs(cache_dir, exist_ok=True)

        # A stale manifest from a different job's (different-content) data
        # at this same path — present BEFORE this call even starts.
        stale_path = os.path.join(cache_dir, "stale_from_another_job.parquet")
        with open(os.path.join(cache_dir, "manifest.json"), "w") as mf:
            json.dump([stale_path], mf)
        with open(os.path.join(cache_dir, "manifest.done"), "w") as sf:
            sf.write("ok")

        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) == 1:
                return  # first poll still sees the stale manifest
            # Simulate rank0 finishing its OWN (correct) rebuild.
            monkeypatch.setenv("LOCAL_RANK", "0")
            _cache_jsonl_as_parquet(files)
            monkeypatch.setenv("LOCAL_RANK", "1")

        monkeypatch.setattr("time.sleep", fake_sleep)
        try:
            result = _cache_jsonl_as_parquet(files)
            assert result != [stale_path]
            assert result[0].endswith(".parquet")
            assert len(sleep_calls) >= 1
        finally:
            self._cleanup(files)


class TestRunVerlSft:
    """_run_verl_sft mirrors verl's own run_sft(): init process group, fit,

    destroy in a finally. Faking just verl.trainer.sft_trainer/
    verl.utils.distributed (not installed) exercises that control flow for
    real, including the finally-runs-even-on-exception guarantee.
    """

    def _install_fake_verl_trainer(self, monkeypatch, *, fit_side_effect=None):
        calls = {"initialized": False, "destroyed": False, "fit_called": False}

        fake_trainer_instance = MagicMock()

        def fake_fit():
            calls["fit_called"] = True
            if fit_side_effect is not None:
                raise fit_side_effect

        fake_trainer_instance.fit = fake_fit

        def fake_sft_trainer_ctor(config):
            calls["config"] = config
            return fake_trainer_instance

        def fake_initialize():
            calls["initialized"] = True

        def fake_destroy():
            calls["destroyed"] = True

        sft_trainer_mod: Any = ModuleType("verl.trainer.sft_trainer")
        sft_trainer_mod.SFTTrainer = fake_sft_trainer_ctor
        distributed_mod: Any = ModuleType("verl.utils.distributed")
        distributed_mod.initialize_global_process_group = fake_initialize
        distributed_mod.destroy_global_process_group = fake_destroy

        monkeypatch.setitem(sys.modules, "verl.trainer.sft_trainer", sft_trainer_mod)
        monkeypatch.setitem(sys.modules, "verl.utils.distributed", distributed_mod)
        return calls

    def test_initializes_fits_and_destroys_in_order(self, monkeypatch):
        calls = self._install_fake_verl_trainer(monkeypatch)
        cfg = SimpleNamespace(marker="cfg", model=SimpleNamespace(lora_rank=0))

        VerlSftMethodHandler._run_verl_sft(cfg, 0.05)

        assert calls["initialized"] is True
        assert calls["fit_called"] is True
        assert calls["destroyed"] is True
        assert calls["config"] is cfg

    def test_destroys_process_group_even_when_fit_raises(self, monkeypatch):
        calls = self._install_fake_verl_trainer(
            monkeypatch, fit_side_effect=RuntimeError("training exploded")
        )
        cfg = SimpleNamespace(model=SimpleNamespace(lora_rank=0))

        with pytest.raises(RuntimeError, match="training exploded"):
            VerlSftMethodHandler._run_verl_sft(cfg, 0.05)

        assert calls["destroyed"] is True

    def test_patches_lora_dropout_only_when_lora_enabled(self, monkeypatch):
        import foresightrlcore.methods.verl_compat as compat_mod

        self._install_fake_verl_trainer(monkeypatch)
        patch_calls = []
        # _run_verl_sft imports patch_verl_lora_dropout function-locally
        # (see foresightrlcore.methods.verl_compat) — patch the name on
        # that module, not on sft_verl itself, so the fresh import at
        # call time picks up the fake.
        monkeypatch.setattr(
            compat_mod,
            "patch_verl_lora_dropout",
            lambda dropout: patch_calls.append(dropout),
        )

        cfg_no_lora = SimpleNamespace(model=SimpleNamespace(lora_rank=0))
        VerlSftMethodHandler._run_verl_sft(cfg_no_lora, 0.05)
        assert patch_calls == []

        cfg_with_lora = SimpleNamespace(model=SimpleNamespace(lora_rank=32))
        VerlSftMethodHandler._run_verl_sft(cfg_with_lora, 0.05)
        assert patch_calls == [0.05]


class TestRunMergeLoraOnly:
    """_run_merge_lora_only's rank gating and checkpoint/output path

    wiring, with foresightrlcore.methods.verl_checkpoint_merge's three
    functions replaced (already separately, thoroughly tested on their
    own in test_verl_checkpoint_merge.py).
    """

    def _install_fakes(self, monkeypatch):
        import foresightrlcore.methods.verl_checkpoint_merge as merge_mod

        captured: dict = {}

        def fake_find_latest(checkpoint_root, step):
            captured["checkpoint_root"] = checkpoint_root
            captured["step"] = step
            return os.path.join(checkpoint_root, "global_step_10")

        def fake_reconstruct(checkpoint_dir, trust_remote_code=True):
            captured["checkpoint_dir"] = checkpoint_dir
            captured["reconstruct_trust_remote_code"] = trust_remote_code
            return {"state": "dict"}

        def fake_merge_and_save(
            state_dict, base_model_path, lora, output_dir, trust_remote_code=True
        ):
            captured["state_dict"] = state_dict
            captured["base_model_path"] = base_model_path
            captured["lora"] = lora
            captured["output_dir"] = output_dir
            captured["merge_trust_remote_code"] = trust_remote_code

        def fake_check_meta(checkpoint_dir, lora):
            captured["meta_checked_dir"] = checkpoint_dir
            captured["meta_checked_lora"] = lora

        monkeypatch.setattr(merge_mod, "find_latest_checkpoint_step", fake_find_latest)
        monkeypatch.setattr(merge_mod, "check_lora_train_meta_matches", fake_check_meta)
        monkeypatch.setattr(merge_mod, "reconstruct_full_state_dict", fake_reconstruct)
        monkeypatch.setattr(merge_mod, "merge_lora_and_save", fake_merge_and_save)
        return captured

    def test_non_rank0_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        captured = self._install_fakes(monkeypatch)

        VerlSftMethodHandler._run_merge_lora_only(
            SimpleNamespace(
                output=SimpleNamespace(checkpoint_s3_path="s3://x", s3_path="s3://y")
            ),
            SimpleNamespace(merge_checkpoint_step=None),
            SimpleNamespace(),
        )

        assert captured == {}

    def test_rank0_wires_checkpoint_and_output_paths_from_s3(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        captured = self._install_fakes(monkeypatch)

        config = SimpleNamespace(
            output=SimpleNamespace(
                checkpoint_s3_path="s3://bucket/checkpoints/job",
                s3_path="s3://bucket/output/job",
            ),
            model=SimpleNamespace(
                name="s3://bucket/models/gemma4/", trust_remote_code=None
            ),
        )
        sft_cfg = SimpleNamespace(merge_checkpoint_step=10)
        lora = SimpleNamespace(marker="lora")

        VerlSftMethodHandler._run_merge_lora_only(config, sft_cfg, lora)

        assert captured["checkpoint_root"] == "s3://bucket/checkpoints/job"
        assert captured["step"] == 10
        assert (
            captured["checkpoint_dir"] == "s3://bucket/checkpoints/job/global_step_10"
        )
        assert captured["output_dir"] == "s3://bucket/output/job"
        # The ORIGINAL pretrained model, not the checkpoint's own
        # (adapter-only) directory — see merge_lora_and_save's docstring.
        assert captured["base_model_path"] == "s3://bucket/models/gemma4/"
        assert captured["state_dict"] == {"state": "dict"}
        assert (
            captured["meta_checked_dir"] == "s3://bucket/checkpoints/job/global_step_10"
        )
        assert captured["meta_checked_lora"] is lora
        assert captured["lora"] is lora
        # trust_remote_code=None (unset) falls back to True, matching the
        # training path's own fallback semantics.
        assert captured["reconstruct_trust_remote_code"] is True
        assert captured["merge_trust_remote_code"] is True

    def test_merge_forwards_explicit_trust_remote_code_false(self, monkeypatch):
        """model.trust_remote_code: false must reach both

        reconstruct_full_state_dict and merge_lora_and_save — previously
        both call sites hardcoded True regardless, silently ignoring an
        explicit False.
        """
        monkeypatch.setenv("RANK", "0")
        captured = self._install_fakes(monkeypatch)

        config = SimpleNamespace(
            output=SimpleNamespace(
                checkpoint_s3_path="s3://bucket/checkpoints/job",
                s3_path="s3://bucket/output/job",
            ),
            model=SimpleNamespace(
                name="s3://bucket/models/gemma4/", trust_remote_code=False
            ),
        )
        sft_cfg = SimpleNamespace(merge_checkpoint_step=None)

        VerlSftMethodHandler._run_merge_lora_only(config, sft_cfg, SimpleNamespace())

        assert captured["reconstruct_trust_remote_code"] is False
        assert captured["merge_trust_remote_code"] is False

    def test_rank0_prefers_local_dirs_when_set_by_sagemaker_overrides(
        self, monkeypatch
    ):
        monkeypatch.setenv("RANK", "0")
        captured = self._install_fakes(monkeypatch)

        config = SimpleNamespace(
            output=SimpleNamespace(
                checkpoint_s3_path="s3://bucket/checkpoints/job",
                s3_path="s3://bucket/output/job",
            ),
            model=SimpleNamespace(
                name="/opt/ml/input/data/model", trust_remote_code=None
            ),
            _local_output_dir="/opt/ml/checkpoints",
            _final_model_dir="/opt/ml/model",
        )
        sft_cfg = SimpleNamespace(merge_checkpoint_step=None)

        VerlSftMethodHandler._run_merge_lora_only(config, sft_cfg, SimpleNamespace())

        assert captured["checkpoint_root"] == "/opt/ml/checkpoints"
        assert captured["output_dir"] == "/opt/ml/model"
        assert captured["base_model_path"] == "/opt/ml/input/data/model"


class TestCheckPackedDataMatchesConfig:
    """Cross-checks model.enable_thinking/thinking_kwarg and

    training.sft_verl.max_length against what rl-train tokenize-pack
    actually baked into the packed data's metadata sidecar — packed data
    is never re-tokenized at train time, so a mismatch here would
    otherwise silently train on the wrong chat-template rendering, or
    silently truncate every packed row.
    """

    def _sft_cfg(self, max_length=100000, truncation="error", data_format="packed"):
        return SimpleNamespace(
            max_length=max_length, truncation=truncation, data_format=data_format
        )

    def _write_metadata(self, data_dir, entries, pack_size=8192):
        import json

        path = data_dir / f"{pack_size}_metadata.jsonl"
        path.write_text(json.dumps(entries) + "\n")
        return path

    def test_no_metadata_file_is_a_noop_for_non_packed_format(self, tmp_path):
        # flat/messages data_format never has a sidecar at all — its
        # absence is the normal case, not a sign anything is wrong.
        _check_packed_data_matches_config(
            str(tmp_path),
            True,
            "enable_thinking",
            self._sft_cfg(data_format="messages"),
        )  # must not raise

    def test_no_metadata_file_raises_for_packed_format(self, tmp_path):
        # data_format: packed implies the data came from rl-train
        # tokenize-pack, which always writes this sidecar — its absence
        # here is anomalous and must not silently skip the check.
        with pytest.raises(ValueError, match="no \\*_metadata.jsonl sidecar"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )

    def test_legacy_metadata_without_new_fields_is_a_noop(self, tmp_path):
        self._write_metadata(tmp_path, [{"split": "training"}])
        _check_packed_data_matches_config(
            str(tmp_path), True, "enable_thinking", self._sft_cfg()
        )  # must not raise

    def test_matching_thinking_config_does_not_raise(self, tmp_path):
        self._write_metadata(
            tmp_path,
            [
                {
                    "split": "training",
                    "enable_thinking": True,
                    "thinking_kwarg": "enable_thinking",
                }
            ],
        )
        _check_packed_data_matches_config(
            str(tmp_path), True, "enable_thinking", self._sft_cfg()
        )  # must not raise

    def test_mismatched_enable_thinking_raises(self, tmp_path):
        self._write_metadata(
            tmp_path,
            [
                {
                    "split": "training",
                    "enable_thinking": False,
                    "thinking_kwarg": "enable_thinking",
                }
            ],
        )
        with pytest.raises(ValueError, match="enable_thinking=False"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )

    def test_mismatched_thinking_kwarg_raises_even_when_enable_thinking_matches(
        self, tmp_path
    ):
        # A typo'd/mismatched --thinking-kwarg is silently ignored by
        # apply_chat_template at pack time, so enable_thinking alone
        # matching is not enough to prove the right kwarg was rendered.
        self._write_metadata(
            tmp_path,
            [
                {
                    "split": "training",
                    "enable_thinking": True,
                    "thinking_kwarg": "enable_think",  # typo
                }
            ],
        )
        with pytest.raises(ValueError, match="thinking_kwarg='enable_think'"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )

    def test_mismatch_in_second_entry_is_still_caught(self, tmp_path):
        self._write_metadata(
            tmp_path,
            [
                {
                    "split": "training",
                    "enable_thinking": True,
                    "thinking_kwarg": "enable_thinking",
                },
                {
                    "split": "validation",
                    "enable_thinking": False,
                    "thinking_kwarg": "enable_thinking",
                },
            ],
        )
        with pytest.raises(ValueError, match="split='validation'"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )

    def test_max_length_below_pack_size_raises_when_truncation_not_error(
        self, tmp_path
    ):
        self._write_metadata(
            tmp_path, [{"split": "training", "pack_size": 8192}], pack_size=8192
        )
        with pytest.raises(ValueError, match="max_length=4096"):
            _check_packed_data_matches_config(
                str(tmp_path),
                True,
                "enable_thinking",
                self._sft_cfg(max_length=4096, truncation="right"),
            )

    def test_max_length_below_pack_size_is_fine_when_truncation_is_error(
        self, tmp_path
    ):
        # truncation: error (the default) already fails loudly on its own
        # the moment a real oversized row is encountered — no need for
        # this check to duplicate that.
        self._write_metadata(
            tmp_path, [{"split": "training", "pack_size": 8192}], pack_size=8192
        )
        _check_packed_data_matches_config(
            str(tmp_path),
            True,
            "enable_thinking",
            self._sft_cfg(max_length=4096, truncation="error"),
        )  # must not raise

    def test_max_length_at_or_above_pack_size_does_not_raise(self, tmp_path):
        self._write_metadata(
            tmp_path, [{"split": "training", "pack_size": 8192}], pack_size=8192
        )
        _check_packed_data_matches_config(
            str(tmp_path),
            True,
            "enable_thinking",
            self._sft_cfg(max_length=8192, truncation="right"),
        )  # must not raise

    def test_non_directory_data_path_does_not_crash_for_non_packed_format(
        self, tmp_path
    ):
        # data.train_path is usually a directory (or S3 prefix ending in
        # "/"), but must not crash if it's some other string with no
        # matching metadata sidecar alongside it (for a non-packed format,
        # where a missing sidecar is expected, not anomalous).
        _check_packed_data_matches_config(
            str(tmp_path / "no-such-prefix"),
            True,
            "enable_thinking",
            self._sft_cfg(data_format="messages"),
        )

    def test_foreign_line_delimited_jsonl_sidecar_does_not_crash_for_non_packed(
        self, tmp_path
    ):
        # A genuine JSONL sidecar (one object per line, as a *different*
        # packing tool might write) must not crash this best-effort check
        # for a non-packed format — it should just be treated as
        # unreadable and skipped.
        path = tmp_path / "8192_metadata.jsonl"
        path.write_text(
            '{"split": "training", "enable_thinking": true}\n'
            '{"split": "validation", "enable_thinking": true}\n'
        )
        _check_packed_data_matches_config(
            str(tmp_path),
            True,
            "enable_thinking",
            self._sft_cfg(data_format="messages"),
        )  # must not raise

    def test_foreign_line_delimited_jsonl_sidecar_raises_for_packed_format(
        self, tmp_path
    ):
        path = tmp_path / "8192_metadata.jsonl"
        path.write_text(
            '{"split": "training", "enable_thinking": true}\n'
            '{"split": "validation", "enable_thinking": true}\n'
        )
        with pytest.raises(ValueError, match="not in the shape"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )

    def test_non_dict_entries_in_the_list_are_skipped(self, tmp_path):
        self._write_metadata(
            tmp_path, ["not-a-dict", {"split": "training", "pack_size": 8192}]
        )
        _check_packed_data_matches_config(
            str(tmp_path), True, "enable_thinking", self._sft_cfg()
        )  # must not raise

    def test_empty_sidecar_does_not_crash_for_non_packed_format(self, tmp_path):
        (tmp_path / "8192_metadata.jsonl").write_text("")
        _check_packed_data_matches_config(
            str(tmp_path),
            True,
            "enable_thinking",
            self._sft_cfg(data_format="messages"),
        )  # must not raise

    def test_empty_sidecar_raises_for_packed_format(self, tmp_path):
        (tmp_path / "8192_metadata.jsonl").write_text("")
        with pytest.raises(ValueError, match="could not be read/parsed"):
            _check_packed_data_matches_config(
                str(tmp_path), True, "enable_thinking", self._sft_cfg()
            )


class TestRun:
    """VerlSftMethodHandler.run()'s real Hydra-cfg-building logic, with

    ``hydra`` faked in sys.modules (omegaconf itself is a real dependency
    already, so OmegaConf.update(...) inside run() executes for real
    against the fake-but-non-struct-mode DictConfig hydra.compose()
    returns), and _ensure_flash_attn_compatible/_run_verl_sft stubbed out
    (both are separately, fully tested on their own above).
    """

    def _install_fake_hydra(self, monkeypatch):
        from omegaconf import OmegaConf

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fresh_cfg():
            return OmegaConf.create(
                {
                    "model": {},
                    "engine": {},
                    "optim": {},
                    "data": {"custom_cls": {}},
                    "checkpoint": {},
                    "trainer": {},
                }
            )

        fake_hydra: Any = ModuleType("hydra")
        fake_hydra.initialize_config_module = (
            lambda config_module, version_base=None: _Ctx()
        )
        fake_hydra.compose = lambda config_name: _fresh_cfg()
        monkeypatch.setitem(sys.modules, "hydra", fake_hydra)

    def _build_config(self, tmp_yaml, sft_verl_overrides=None, **top_overrides):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = False
        if sft_verl_overrides:
            d["training"]["sft_verl"].update(sft_verl_overrides)
        for k, v in top_overrides.items():
            d[k] = v
        content = yaml.dump(d)
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def _run(self, monkeypatch, config, local_world_size="8"):
        import foresightrlcore.methods.sft_verl as mod

        self._install_fake_hydra(monkeypatch)
        # A real torchrun launch always sets this (see
        # cfg.trainer.n_gpus_per_node's assignment in run(), which now
        # raises rather than silently defaulting) — set it here once so
        # every TestRun test doesn't need to. Pass local_world_size=None
        # to instead test the "unset" case.
        if local_world_size is None:
            monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
        monkeypatch.setattr(
            mod, "_ensure_flash_attn_compatible", lambda wheel_url: None
        )
        monkeypatch.setattr(
            mod, "_glob_data_files", lambda *a, **k: ["/fake/train.file"]
        )
        monkeypatch.setattr(mod, "_cache_jsonl_as_parquet", lambda files: files)
        # Reads real files via glob — data.train_path here is a fake S3
        # URL with no real sidecar; that consistency check is separately,
        # fully tested on its own in TestCheckPackedDataMatchesConfig.
        monkeypatch.setattr(
            mod, "_check_packed_data_matches_config", lambda *a, **k: None
        )

        captured: dict = {
            "transformers_version_calls": [],
            "tie_word_embeddings_patch_calls": 0,
        }

        def fake_ensure_transformers_version(min_version):
            captured["transformers_version_calls"].append(min_version)

        monkeypatch.setattr(
            mod,
            "_ensure_transformers_version_compatible",
            fake_ensure_transformers_version,
        )

        def fake_patch_tie_word_embeddings():
            captured["tie_word_embeddings_patch_calls"] += 1

        # A local import inside run() (`from
        # foresightrlcore.methods.verl_compat import
        # patch_muse_glimmer_tie_word_embeddings`), not a module-level
        # attribute of sft_verl — same reason TestRunVerlSft patches
        # verl_compat directly for patch_verl_lora_dropout below.
        import foresightrlcore.methods.verl_compat as compat_mod

        monkeypatch.setattr(
            compat_mod,
            "patch_muse_glimmer_tie_word_embeddings",
            fake_patch_tie_word_embeddings,
        )

        def fake_run_verl_sft(cfg, lora_dropout):
            captured["cfg"] = cfg
            captured["lora_dropout"] = lora_dropout

        monkeypatch.setattr(
            mod.VerlSftMethodHandler, "_run_verl_sft", staticmethod(fake_run_verl_sft)
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)
        self._last_captured = captured
        return captured["cfg"]

    def test_messages_format_maps_core_fields_correctly(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)

        assert cfg.model.path == config.model.name
        assert cfg.model.use_remove_padding is False
        assert (
            cfg.model.enable_gradient_checkpointing
            == config.training.gradient_checkpointing
        )
        assert cfg.model.lora_rank == config.model.lora.r
        assert cfg.model.lora_alpha == config.model.lora.alpha
        assert self._last_captured["lora_dropout"] == config.model.lora.dropout
        assert list(cfg.model.target_modules) == list(config.model.lora.target_modules)
        assert cfg.engine.ulysses_sequence_parallel_size == 1
        assert cfg.engine.strategy == config.training.sft_verl.fsdp_strategy
        assert cfg.optim.lr == config.training.learning_rate
        assert cfg.data.train_files == ["/fake/train.file"]
        assert cfg.data.val_files is None
        assert cfg.data.train_batch_size == config.training.sft_verl.train_batch_size
        assert cfg.data.custom_cls.get("path") is None
        assert cfg.trainer.project_name == config.job.name
        assert cfg.trainer.experiment_name == config.job.name
        assert cfg.trainer.total_epochs == config.training.epochs
        assert cfg.trainer.save_freq == config.training.sft_verl.save_freq
        assert cfg.trainer.resume_mode == config.training.sft_verl.resume_mode
        assert cfg.checkpoint.save_lora_only == config.training.sft_verl.save_lora_only

    def test_eval_path_set_when_configured(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            data={
                "train_path": "s3://bucket/data/train",
                "eval_path": "s3://bucket/data/eval",
            },
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.data.val_files == ["/fake/train.file"]

    def test_flat_format_sets_custom_cls_and_prompt_keys(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={
                "data_format": "flat",
                "prompt_key": "q",
                "response_key": "a",
                "system_prompt": "be nice",
            },
        )
        cfg = self._run(monkeypatch, config)

        assert (
            cfg.data.custom_cls.path == "pkg://foresightrlcore.methods.verl_sft_dataset"
        )
        assert cfg.data.custom_cls.name == "FlatPromptResponseSFTDataset"
        assert cfg.data.prompt_key == "q"
        assert cfg.data.response_key == "a"
        assert cfg.data.system_prompt == "be nice"

    def test_packed_format_sets_custom_cls_to_packed_dataset(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "packed"}
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.data.custom_cls.name == "PackedSFTDataset"
        # "packed"/"flat" both need custom_cls, but only "flat" needs the
        # prompt/response/system_prompt OmegaConf.update calls.
        assert cfg.data.get("prompt_key") is None

    def test_target_parameters_set_via_omegaconf_force_add(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={
                "name": "org/fake-model",
                "lora": {
                    "r": 64,
                    "alpha": 128,
                    "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
                },
            },
        )
        cfg = self._run(monkeypatch, config)
        assert list(cfg.model.target_parameters) == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]

    def test_empty_target_modules_falls_back_to_all_linear(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={
                "name": "org/fake-model",
                "lora": {"r": 64, "alpha": 128, "target_modules": []},
            },
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.model.target_modules == "all-linear"

    def test_exclude_modules_set_when_configured(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={
                "name": "org/fake-model",
                "lora": {"r": 64, "alpha": 128, "exclude_modules": ["router.proj"]},
            },
        )
        cfg = self._run(monkeypatch, config)
        assert list(cfg.model.exclude_modules) == ["router.proj"]

    def test_exclude_modules_regex_string_passed_through_unwrapped(
        self, tmp_yaml, monkeypatch
    ):
        """A regex string must reach verl's config unchanged, not shredded

        into individual characters by an errant list(...) wrap — needed to
        exclude an entire subtree (e.g. a nested vision tower) that a
        plain list's exact-trailing-segment matching can't express.
        """
        regex = ".*(vision_tower|vision_adapter).*"
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={
                "name": "meta-models/Muse-Glimmer-30B",
                "lora": {"r": 64, "alpha": 128, "exclude_modules": regex},
            },
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.model.exclude_modules == regex

    def test_muse_glimmer_profile_triggers_transformers_version_swap(
        self, tmp_yaml, monkeypatch
    ):
        """model.profile=muse_glimmer must trigger the transformers version

        swap (verl's own <5.11 cap doesn't cover this model's
        transformers>=5.15 requirement) — see model_profiles.py's
        min_transformers_version and this module's
        _ensure_transformers_version_compatible.
        """
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={"name": "meta-models/Muse-Glimmer-30B", "profile": "muse_glimmer"},
        )
        self._run(monkeypatch, config)
        assert self._last_captured["transformers_version_calls"] == ["5.15.0"]
        assert self._last_captured["tie_word_embeddings_patch_calls"] == 1

    def test_default_profile_does_not_trigger_transformers_version_swap(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        self._run(monkeypatch, config)
        assert self._last_captured["transformers_version_calls"] == []
        assert self._last_captured["tie_word_embeddings_patch_calls"] == 0

    def test_attn_implementation_sets_override_config(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={"name": "org/fake-model", "attn_implementation": "sdpa"},
        )
        cfg = self._run(monkeypatch, config)
        assert dict(cfg.model.override_config) == {"attn_implementation": "sdpa"}

    def test_gemma4_moe_profile_resolves_to_verl_cfg_end_to_end(
        self, tmp_yaml, monkeypatch
    ):
        """The one dedicated real-model test in this file: model.profile:

        gemma4_moe alone (no explicit lora/attn_implementation fields at
        all) must resolve target_modules/target_parameters/
        attn_implementation through to verl's cfg exactly the same way
        the explicit-field tests above (test_target_parameters_set_via_
        omegaconf_force_add, test_empty_target_modules_falls_back_to_all_
        linear, test_attn_implementation_sets_override_config) verify the
        underlying mechanics with — this test is what actually exercises
        the real model this recipe was built for, end to end.
        """
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={"name": "google/gemma-4-26B-A4B-it", "profile": "gemma4_moe"},
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.model.target_modules == "all-linear"
        assert list(cfg.model.target_parameters) == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]
        assert list(cfg.model.exclude_modules) == ["router.proj"]
        assert dict(cfg.model.override_config) == {"attn_implementation": "sdpa"}

    def test_flash_attn_fix_runs_even_for_sdpa_profile(self, tmp_yaml, monkeypatch):
        """Regression guard: verl.trainer.sft_trainer's own import chain

        (sft_trainer.py -> multiturn_sft_dataset.py ->
        verl.models.transformers.qwen2_vl -> `from flash_attn import
        flash_attn_func, ...`) unconditionally imports flash_attn at
        module load, for every sft_verl job — including a model.profile
        like gemma4_moe that forces attn_implementation="sdpa" and never
        calls into flash_attn's kernels at runtime. A prior version of
        run() gated the _ensure_flash_attn_compatible call on
        attn_implementation == "flash_attention_2", reasoning that
        sdpa-only jobs "never touch flash-attn at all" — true for the
        attention computation, false for this import. That left the
        container's generically pip-resolved flash-attn (linked against a
        CUDA runtime this DLC image doesn't ship) in place, and a real
        SageMaker job (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-
        20260929150913, gemma4_moe/sdpa) crashed on both nodes with
        `ImportError: libcudart.so.12: cannot open shared object file`
        before a single training step ran. This test does NOT use the
        shared _run() helper (which stubs _ensure_flash_attn_compatible
        unconditionally) — it spies on the real function directly to
        confirm it is actually invoked for an sdpa profile.
        """
        import foresightrlcore.methods.sft_verl as mod

        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            model={"name": "google/gemma-4-26B-A4B-it", "profile": "gemma4_moe"},
        )
        self._install_fake_hydra(monkeypatch)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        monkeypatch.setattr(
            mod, "_glob_data_files", lambda *a, **k: ["/fake/train.file"]
        )
        monkeypatch.setattr(mod, "_cache_jsonl_as_parquet", lambda files: files)
        monkeypatch.setattr(
            mod, "_check_packed_data_matches_config", lambda *a, **k: None
        )
        monkeypatch.setattr(
            mod.VerlSftMethodHandler,
            "_run_verl_sft",
            staticmethod(lambda cfg, lora_dropout: None),
        )

        captured: dict = {"calls": []}

        def fake_ensure_flash_attn(wheel_url):
            captured["calls"].append(wheel_url)

        monkeypatch.setattr(
            mod, "_ensure_flash_attn_compatible", fake_ensure_flash_attn
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)

        assert len(captured["calls"]) == 1

    def test_packed_data_check_covers_eval_path_when_different_from_train(
        self, tmp_yaml, monkeypatch
    ):
        """Regression guard: _check_packed_data_matches_config was only

        ever called with data.train_path — a thinking/max_length mismatch
        baked into an eval-only pack (e.g. a validation set packed via its
        own `rl-train tokenize-pack --split-name validation` run, living
        at a genuinely different prefix from training data) was never
        caught. Does NOT use the shared _run() helper (which stubs this
        function unconditionally) — spies on it directly to confirm it's
        called for both paths when they differ.
        """
        import foresightrlcore.methods.sft_verl as mod

        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "packed"},
            data={
                "train_path": "s3://bucket/data/train",
                "eval_path": "s3://bucket/data/eval",
            },
        )
        self._install_fake_hydra(monkeypatch)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        monkeypatch.setattr(
            mod, "_glob_data_files", lambda *a, **k: ["/fake/train.file"]
        )
        monkeypatch.setattr(mod, "_ensure_flash_attn_compatible", lambda *a: None)
        monkeypatch.setattr(
            mod.VerlSftMethodHandler,
            "_run_verl_sft",
            staticmethod(lambda cfg, lora_dropout: None),
        )

        captured: list = []
        monkeypatch.setattr(
            mod,
            "_check_packed_data_matches_config",
            lambda data_path, *a, **k: captured.append(data_path),
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)

        assert captured == ["s3://bucket/data/train", "s3://bucket/data/eval"]

    def test_packed_data_check_not_duplicated_when_eval_equals_train(
        self, tmp_yaml, monkeypatch
    ):
        """No-op case: eval_path == train_path means the exact same

        *_metadata.jsonl sidecar was already checked once — no need (and
        no correctness gap) to check it again.
        """
        import foresightrlcore.methods.sft_verl as mod

        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "packed"},
            data={
                "train_path": "s3://bucket/data/shared",
                "eval_path": "s3://bucket/data/shared",
            },
        )
        self._install_fake_hydra(monkeypatch)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        monkeypatch.setattr(
            mod, "_glob_data_files", lambda *a, **k: ["/fake/train.file"]
        )
        monkeypatch.setattr(mod, "_ensure_flash_attn_compatible", lambda *a: None)
        monkeypatch.setattr(
            mod.VerlSftMethodHandler,
            "_run_verl_sft",
            staticmethod(lambda cfg, lora_dropout: None),
        )

        captured: list = []
        monkeypatch.setattr(
            mod,
            "_check_packed_data_matches_config",
            lambda data_path, *a, **k: captured.append(data_path),
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)

        assert captured == ["s3://bucket/data/shared"]

    def test_weight_decay_always_reaches_verl(self, tmp_yaml, monkeypatch):
        """weight_decay is a plain float (not Optional), so there's no

        "unset" sentinel to skip on — it must always be forwarded, or
        verl's own FSDPOptimizerConfig default (0.01) silently applies
        instead of ours (0.0) or whatever the user actually configured.
        """
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = False
        d["training"]["weight_decay"] = 0.05
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        cfg = self._run(monkeypatch, config)
        assert cfg.optim.weight_decay == 0.05

    def test_weight_decay_default_of_zero_also_reaches_verl(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.optim.weight_decay == 0.0

    def test_max_grad_norm_set_reaches_verl_as_clip_grad(self, tmp_yaml, monkeypatch):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = False
        d["training"]["max_grad_norm"] = 0.5
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        cfg = self._run(monkeypatch, config)
        assert cfg.optim.clip_grad == 0.5

    def test_max_grad_norm_unset_leaves_clip_grad_untouched(
        self, tmp_yaml, monkeypatch
    ):
        # None means "no override" — verl's own clip_grad default applies,
        # same convention as model.attn_implementation elsewhere.
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)
        assert "clip_grad" not in cfg.optim

    def test_dataloader_num_workers_unset_leaves_verl_default_untouched(
        self, tmp_yaml, monkeypatch
    ):
        # None means "no override" — verl's own hydra default (8) applies,
        # same convention as max_grad_norm/model.attn_implementation
        # elsewhere.
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)
        assert "num_workers" not in cfg.data

    def test_dataloader_num_workers_set_reaches_verl(self, tmp_yaml, monkeypatch):
        """Regression guard for a real SageMaker crash: verl's

        val_dataloader shares the exact same data.num_workers field as
        train_dataloader (verl/trainer/sft_trainer.py's _build_dataloader),
        and every GPU-local-rank process forks that many worker
        subprocesses independently. On a real job
        (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-20260929183107,
        Gemma 4 26B-A4B with data.eval_path set), the first-ever
        val_dataloader construction crashed on every rank with `OSError:
        [Errno 12] Cannot allocate memory` at os.fork() — training itself
        had already completed and checkpointed fine, since
        train_dataloader's workers fork early, before the model/optimizer
        grow the parent process's memory footprint. This field exists so a
        config can lower that worker count.
        """
        config = self._build_config(
            tmp_yaml,
            sft_verl_overrides={
                "data_format": "messages",
                "dataloader_num_workers": 2,
            },
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.data.num_workers == 2

    def test_local_world_size_sets_n_gpus_per_node(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config, local_world_size="4")
        assert cfg.trainer.n_gpus_per_node == 4

    def test_missing_local_world_size_raises(self, tmp_yaml, monkeypatch):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE is not set"):
            self._run(monkeypatch, config, local_world_size=None)

    def test_max_steps_sets_total_training_steps_when_positive(
        self, tmp_yaml, monkeypatch
    ):
        d = _sft_verl_config_dict()
        d["training"]["sft_verl"]["ulysses_sequence_parallel_size"] = 1
        d["training"]["sft_verl"]["use_remove_padding"] = False
        d["training"]["max_steps"] = 20
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        cfg = self._run(monkeypatch, config)
        assert cfg.trainer.total_training_steps == 20

    def test_max_steps_unset_leaves_total_training_steps_untouched(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)
        assert "total_training_steps" not in cfg.trainer

    def test_local_output_dir_and_final_model_dir_override_s3_defaults(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        config._local_output_dir = "/opt/ml/checkpoints"
        config._final_model_dir = "/opt/ml/model"
        cfg = self._run(monkeypatch, config)
        assert cfg.trainer.default_local_dir == "/opt/ml/checkpoints"

    def test_output_s3_path_used_when_local_output_dir_unset(
        self, tmp_yaml, monkeypatch
    ):
        config = self._build_config(
            tmp_yaml, sft_verl_overrides={"data_format": "messages"}
        )
        cfg = self._run(monkeypatch, config)
        assert cfg.trainer.default_local_dir == config.output.s3_path

    def test_merge_lora_only_dispatches_and_returns_before_building_cfg(
        self, tmp_yaml, monkeypatch
    ):
        """merge_lora_only must be checked, and dispatched, before ANY of

        hydra/flash-attn/GPU-family setup — deliberately does NOT fake
        hydra or stub _ensure_flash_attn_compatible/resolve_flash_attn_wheel
        here, and deliberately uses an instance_type OUTSIDE GPU_FAMILIES
        (merge_lora_only never touches the GPU/attention path, so
        examples/smoke_tests/sft_verl_merge_smoke.yaml's own header
        explicitly allows a cheaper CPU-only instance for this mode).
        Before the reorder fix,
        this would have failed with "Unrecognized GPU family" — and with
        an ImportError for the real (unfaked, uninstalled) hydra module —
        before ever reaching the merge dispatch.
        """
        import foresightrlcore.methods.sft_verl as mod

        d = _sft_verl_config_dict()
        d["infra"]["instance_type"] = "ml.m5.24xlarge"
        d["training"]["sft_verl"] = {"merge_lora_only": True}
        d["output"] = {
            "checkpoint_s3_path": "s3://bucket/checkpoints/step_10",
            "s3_path": "s3://bucket/output",
        }
        del d["data"]
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        captured: dict = {}

        def fake_merge(config_, sft_cfg, lora):
            captured["called_with_config"] = config_ is config

        monkeypatch.setattr(
            mod.VerlSftMethodHandler, "_run_merge_lora_only", staticmethod(fake_merge)
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)

        assert captured["called_with_config"] is True

    def test_merge_lora_only_still_runs_transformers_version_swap(
        self, tmp_yaml, monkeypatch
    ):
        """Regression guard: merge_lora_only loads the base model via plain

        transformers AutoModel classes (see
        foresightrlcore.methods.verl_checkpoint_merge.merge_lora_and_save),
        which for model.profile: muse_glimmer only exist in
        transformers>=5.15.0 — the container's verl-capped install is too
        old. A real SageMaker merge job failed with exactly this because
        the swap used to run only in the training branch, after the
        merge_lora_only early return. Deliberately does NOT fake hydra
        here either (same reasoning as the sibling test above) — the swap
        must happen before hydra/flash-attn setup too.
        """
        import foresightrlcore.methods.sft_verl as mod

        d = _sft_verl_config_dict()
        d["model"] = {"name": "meta-models/Muse-Glimmer-30B", "profile": "muse_glimmer"}
        d["infra"]["instance_type"] = "ml.m5.24xlarge"
        d["training"]["sft_verl"] = {"merge_lora_only": True}
        d["output"] = {
            "checkpoint_s3_path": "s3://bucket/checkpoints/step_10",
            "s3_path": "s3://bucket/output",
        }
        del d["data"]
        content = yaml.dump(d)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        captured: dict = {"transformers_version_calls": [], "merge_called": False}

        def fake_ensure_transformers_version(min_version):
            captured["transformers_version_calls"].append(min_version)

        monkeypatch.setattr(
            mod,
            "_ensure_transformers_version_compatible",
            fake_ensure_transformers_version,
        )

        def fake_merge(config_, sft_cfg, lora):
            captured["merge_called"] = True

        monkeypatch.setattr(
            mod.VerlSftMethodHandler, "_run_merge_lora_only", staticmethod(fake_merge)
        )

        handler = mod.VerlSftMethodHandler()
        handler.run(config)

        assert captured["transformers_version_calls"] == ["5.15.0"]
        assert captured["merge_called"] is True
