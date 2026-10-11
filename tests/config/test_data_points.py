"""Tests for training.data_points → max_steps derivation.

Covers:
- compute_training_processes for every topology / vllm_mode combination
- _resolve_data_points correctly derives and mutates max_steps
- Hard validation error when both max_steps and data_points are set
- Local backend uses GPU list length
- aws_serverless skips computation and emits a warning
"""

from __future__ import annotations

import math

import pytest
import yaml

from foresightrlcore.config.profiles import compute_training_processes
from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.config.validation import validate_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(overrides: dict) -> TrainingJobConfig:
    """Build a TrainingJobConfig from a minimal base merged with overrides."""
    base: dict = {
        "job": {"name": "test"},
        "model": {"name": "m"},
        "method": {"type": "grpo"},
        "training": {
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "grpo": {"num_generations": 8},
        },
        "data": {"train_path": "s3://b/train"},
        "output": {"s3_path": "s3://b/out"},
        "execution": {"type": "aws_training_job"},
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
    }
    _deep_merge(base, overrides)
    return TrainingJobConfig.model_validate(base)


def _deep_merge(base: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# compute_training_processes — topology coverage
# ---------------------------------------------------------------------------


class TestComputeTrainingProcesses:
    """Unit tests for the process-count formula per topology."""

    def test_single_node_server_uses_training_gpu_count(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",
                    "instance_count": 1,
                    "vllm_gpu_count": 2,
                    "training_gpu_count": 6,
                }
            }
        )
        assert compute_training_processes(config) == 6

    def test_single_node_colocate_divides_by_tp(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 1,
                    "vllm": {"tensor_parallel_size": 2},
                }
            }
        )
        # (8 × 1) / 2 = 4
        assert compute_training_processes(config) == 4

    def test_single_node_none_all_gpus(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "none",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 1,
                }
            }
        )
        assert compute_training_processes(config) == 8

    def test_multi_node_server_2_nodes(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 2,
                    "vllm_gpu_count": 2,
                }
            }
        )
        # (8 − 2) × 2 = 12
        assert compute_training_processes(config) == 12

    def test_multi_node_server_4_nodes(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 4,
                    "vllm_gpu_count": 4,
                }
            }
        )
        # (8 − 4) × 4 = 16
        assert compute_training_processes(config) == 16

    def test_dedicated_vllm_3_nodes(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 3,
                }
            }
        )
        # node 0 → vLLM only; 2 training nodes × 8 GPUs = 16
        assert compute_training_processes(config) == 16

    def test_dedicated_vllm_mixed_uses_training_instance_gpus(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "vllm_mode": "server",
                    "instance_type": "ml.g6e.48xlarge",  # 8 GPUs — training
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",  # 4 GPUs — ignored
                    "vllm_instance_count": 1,
                }
            }
        )
        # 1 training node × 8 GPUs
        assert compute_training_processes(config) == 8

    def test_dedicated_vllm_mixed_three_nodes(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "vllm_mode": "server",
                    "instance_type": "ml.g6e.48xlarge",  # 8 GPUs
                    "instance_count": 3,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "vllm_instance_count": 1,
                }
            }
        )
        # 2 training nodes × 8 GPUs
        assert compute_training_processes(config) == 16

    def test_none_mode_multi_node(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "none",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs
                    "instance_count": 3,
                }
            }
        )
        # all GPUs on all nodes
        assert compute_training_processes(config) == 24

    def test_gpus_per_node_override_takes_precedence(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",  # 8 GPUs — ignored
                    "instance_count": 2,
                    "vllm_gpu_count": 2,
                    "gpus_per_node": 4,  # explicit override
                }
            }
        )
        # (4 − 2) × 2 = 4
        assert compute_training_processes(config) == 4

    def test_unknown_instance_type_defaults_to_8(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "none",
                    "instance_type": "ml.p99.huge",  # not in GPU_COUNTS
                    "instance_count": 1,
                }
            }
        )
        assert compute_training_processes(config) == 8

    def test_local_backend_uses_gpu_list_length(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "server",
                    "vllm_gpu_count": 2,
                    "training_gpu_count": 6,
                },
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/out"},
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
                        "output_dir": "/tmp/out",
                    },
                },
            }
        )
        # server mode: 8 gpus total − 2 vllm = 6 training
        assert compute_training_processes(config) == 6

    def test_local_backend_none_mode_all_gpus(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "none",
                },
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/out"},
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": [0, 1, 2, 3],
                        "output_dir": "/tmp/out",
                    },
                },
            }
        )
        assert compute_training_processes(config) == 4


# ---------------------------------------------------------------------------
# max_steps derivation via validate_config
# ---------------------------------------------------------------------------


class TestDataPointsResolution:
    """Integration tests: data_points → max_steps via validate_config."""

    def test_data_points_derives_max_steps(self):
        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",
                    "instance_count": 2,
                    "vllm_gpu_count": 2,
                },
                "training": {"data_points": 10000},
            }
        )
        assert config.training.max_steps is None  # not yet resolved

        errors = validate_config(config)
        assert errors == []

        # training_procs = (8−2)×2 = 12
        # effective_batch = 4 × 12 × 2 = 96
        # max_steps = ceil(10000 / 96) = 105
        expected = math.ceil(10000 / (4 * 12 * 2))
        assert config.training.max_steps == expected

    def test_data_points_scales_with_gpu_count(self):
        """Same data_points on 4-node produces fewer steps than 2-node."""
        config_2node = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",
                    "instance_count": 2,
                    "vllm_gpu_count": 2,
                },
                "training": {"data_points": 10000},
            }
        )
        config_4node = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "server",
                    "instance_type": "ml.p4de.24xlarge",
                    "instance_count": 4,
                    "vllm_gpu_count": 4,
                },
                "training": {"data_points": 10000},
            }
        )
        validate_config(config_2node)
        validate_config(config_4node)

        # 4-node sees 10000 examples in fewer optimizer steps
        assert config_4node.training.max_steps is not None
        assert config_2node.training.max_steps is not None
        assert config_4node.training.max_steps < config_2node.training.max_steps

    def test_no_data_points_leaves_max_steps_unchanged(self):
        config = _make_config({"training": {"max_steps": 50}})
        errors = validate_config(config)
        assert errors == []
        assert config.training.max_steps == 50

    def test_neither_field_set_leaves_max_steps_none(self):
        config = _make_config({})
        errors = validate_config(config)
        assert errors == []
        assert config.training.max_steps is None


# ---------------------------------------------------------------------------
# Conflict: both max_steps and data_points set → hard error
# ---------------------------------------------------------------------------


class TestDataPointsMaxStepsConflict:
    def test_both_set_returns_error(self):
        config = _make_config({"training": {"max_steps": 100, "data_points": 10000}})
        errors = validate_config(config)
        assert len(errors) == 1
        assert "mutually exclusive" in errors[0]

    def test_both_set_max_steps_not_mutated(self):
        config = _make_config({"training": {"max_steps": 42, "data_points": 9999}})
        validate_config(config)
        # max_steps must not have been overwritten despite the error
        assert config.training.max_steps == 42


# ---------------------------------------------------------------------------
# aws_serverless: data_points skipped with a warning
# ---------------------------------------------------------------------------


class TestDataPointsServerless:
    def test_serverless_data_points_skipped(self, recwarn):
        config = _make_config(
            {
                "training": {"data_points": 5000},
                "execution": {"type": "aws_serverless"},
            }
        )
        errors = validate_config(config)
        # No hard error — serverless config issues come from _validate_serverless
        # which will fail on missing serverless section; filter those out
        dp_errors = [e for e in errors if "mutually exclusive" in e]
        assert dp_errors == []
        # max_steps must not have been derived
        assert config.training.max_steps is None
