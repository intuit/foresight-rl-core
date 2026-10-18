"""Training-rank S3 checkpoint sync used when SageMaker CheckpointConfig is omitted."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.methods.base import (
    _restore_mixed_checkpoints,
    _self_manages_checkpoint_s3,
    _upload_mixed_checkpoints,
)


def test_self_manages_false_without_checkpoint_path() -> None:
    infra = SimpleNamespace(
        uses_mixed_instance_types=lambda: True,
    )
    config = SimpleNamespace(
        output=SimpleNamespace(checkpoint_s3_path=None),
        infra=infra,
    )
    assert _self_manages_checkpoint_s3(config) is False


def test_self_manages_false_when_homogeneous() -> None:
    infra = SimpleNamespace(
        uses_mixed_instance_types=lambda: False,
    )
    config = SimpleNamespace(
        output=SimpleNamespace(checkpoint_s3_path="s3://b/ckpt"),
        infra=infra,
    )
    assert _self_manages_checkpoint_s3(config) is False


def test_self_manages_true_for_mixed_with_checkpoint_uri(tmp_yaml) -> None:
    content = yaml.dump(
        {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "infra": {
                "topology": "dedicated_vllm",
                "vllm_mode": "server",
                "instance_type": "ml.g6.12xlarge",
                "instance_count": 2,
                "vllm_instance_type": "ml.g6.16xlarge",
            },
            "data": {"train_path": "s3://b/d"},
            "output": {
                "s3_path": "s3://b/o",
                "checkpoint_s3_path": "s3://b/ckpt",
            },
            "task": {"type": "advice"},
        }
    )
    config = TrainingJobConfig.from_yaml(tmp_yaml(content))
    assert config.infra.uses_mixed_instance_types() is True
    assert _self_manages_checkpoint_s3(config) is True


def test_restore_rank0_syncs_s3_to_local(tmp_path) -> None:
    local = str(tmp_path / "ckpt")
    with (
        patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
        patch("foresightrlcore.methods.base._aws_s3_sync") as sync,
    ):
        _restore_mixed_checkpoints("s3://b/ckpt", local)
    sync.assert_called_once_with("s3://b/ckpt", local)
    assert os.path.exists(os.path.join(local, ".foresight_ckpt_restore.done"))


def test_restore_non_rank0_waits_for_sentinel(tmp_path) -> None:
    local = str(tmp_path / "ckpt")
    os.makedirs(local)
    with open(os.path.join(local, ".foresight_ckpt_restore.done"), "w") as handle:
        handle.write("ok")
    with (
        patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=False),
        patch("foresightrlcore.methods.base._aws_s3_sync") as sync,
    ):
        _restore_mixed_checkpoints("s3://b/ckpt", local)
    sync.assert_not_called()


def test_restore_non_rank0_raises_on_rank0_failure(tmp_path) -> None:
    local = str(tmp_path / "ckpt")
    os.makedirs(local)
    with open(os.path.join(local, ".foresight_ckpt_restore.failed"), "w") as handle:
        handle.write("boom")
    with patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=False):
        with pytest.raises(RuntimeError, match="rank0 failed"):
            _restore_mixed_checkpoints("s3://b/ckpt", local)


def test_upload_only_local_rank0(tmp_path) -> None:
    local = str(tmp_path / "ckpt")
    os.makedirs(local)
    with (
        patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=False),
        patch("foresightrlcore.methods.base._aws_s3_sync") as sync,
    ):
        _upload_mixed_checkpoints(local, "s3://b/ckpt")
    sync.assert_not_called()

    with (
        patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
        patch("foresightrlcore.methods.base._aws_s3_sync") as sync,
    ):
        _upload_mixed_checkpoints(local, "s3://b/ckpt")
    sync.assert_called_once_with(local, "s3://b/ckpt")


def test_aws_s3_sync_invokes_cli_without_delete() -> None:
    from foresightrlcore.methods.base import _aws_s3_sync

    completed = MagicMock(returncode=0, stdout="", stderr="")
    with (
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run", return_value=completed) as run,
    ):
        _aws_s3_sync("/tmp/ckpt", "s3://b/ckpt")
    run.assert_called_once()
    assert run.call_args[0][0] == ["aws", "s3", "sync", "/tmp/ckpt", "s3://b/ckpt"]
    assert "--delete" not in run.call_args[0][0]
