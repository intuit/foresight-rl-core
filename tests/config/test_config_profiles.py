"""Tests for foresightrlcore.config.profiles — infra profiles and GPU lookup."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.profiles import (
    INFRA_PROFILES,
    gpu_family,
    gpus_for_instance_type,
    instance_has_efa,
    preferred_image_instance_type,
)
from foresightrlcore.config.schema import TrainingJobConfig


class TestGpuLookup:
    """GPU count for known instance types."""

    def test_p4d(self):
        assert gpus_for_instance_type("ml.p4d.24xlarge") == 8

    def test_g5_12xlarge(self):
        assert gpus_for_instance_type("ml.g5.12xlarge") == 4

    def test_g6e_xlarge(self):
        assert gpus_for_instance_type("ml.g6e.xlarge") == 1

    def test_g6_2xlarge(self):
        assert gpus_for_instance_type("ml.g6.2xlarge") == 1
        assert gpus_for_instance_type("ml.g6.8xlarge") == 1
        assert gpus_for_instance_type("ml.g6.12xlarge") == 4

    def test_unknown_defaults_to_8(self):
        assert gpus_for_instance_type("ml.unknown.mega") == 8


class TestGpuFamilyAndEfa:
    def test_p4d_and_p4de_same_family(self):
        assert gpu_family("ml.p4d.24xlarge") == "p4d"
        assert gpu_family("ml.p4de.24xlarge") == "p4d"

    def test_p5_variants(self):
        assert gpu_family("ml.p5.48xlarge") == "p5"
        assert gpu_family("ml.p5e.48xlarge") == "p5"

    def test_g5_g6e(self):
        assert gpu_family("ml.g5.12xlarge") == "g5"
        assert gpu_family("ml.g6e.12xlarge") == "g6e"

    def test_efa_p4d_yes_g5_small_no(self):
        assert instance_has_efa("ml.p4d.24xlarge") is True
        assert instance_has_efa("ml.g5.4xlarge") is False
        assert instance_has_efa("ml.g5.8xlarge") is True
        assert instance_has_efa("ml.g5.12xlarge") is True

    def test_efa_g6e_size_threshold(self):
        assert instance_has_efa("ml.g6e.xlarge") is False
        assert instance_has_efa("ml.g6e.12xlarge") is True
        assert instance_has_efa("ml.g6e.48xlarge") is True

    def test_efa_g6_size_threshold(self):
        assert instance_has_efa("ml.g6.8xlarge") is False
        assert instance_has_efa("ml.g6.12xlarge") is True

    def test_preferred_image_picks_higher_capability(self):
        assert (
            preferred_image_instance_type("ml.g6e.12xlarge", "ml.p5.48xlarge")
            == "ml.p5.48xlarge"
        )
        assert (
            preferred_image_instance_type("ml.p4d.24xlarge", "ml.g6e.12xlarge")
            == "ml.p4d.24xlarge"
        )
        # Tie keeps training type.
        assert (
            preferred_image_instance_type("ml.g6e.48xlarge", "ml.g6e.12xlarge")
            == "ml.g6e.48xlarge"
        )


class TestInfraProfiles:
    """Verify profile definitions."""

    def test_known_profiles_exist(self):
        expected = {
            "single_split_vllm",
            "torchrun",
            "colocate",
            "dedicated_vllm",
            "multi_node_server",
        }
        assert set(INFRA_PROFILES.keys()) == expected

    def test_invalid_profile_raises(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {"infra_profile": "nonexistent"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        with pytest.raises(Exception, match="Unknown infra_profile"):
            TrainingJobConfig.from_yaml(tmp_yaml(content))
