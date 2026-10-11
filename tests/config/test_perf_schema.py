"""Tests for foresightrlcore.config.perf_schema -- PerfTestJobConfig and helpers."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from foresightrlcore.config.perf_schema import (
    GatlingConfig,
    PerfTestJobConfig,
    PerfTestRunConfig,
    VllmServingConfig,
)
from foresightrlcore.config.schema import SpeculativeConfig


@pytest.fixture
def minimal_perf_cfg_dict() -> dict:
    return {
        "job": {"name": "perf-test"},
        "model": {"name": "s3://bucket/model/"},
        "perf_test": {
            "name": "run1",
            "instance_type": "ml.g5.xlarge",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }


@pytest.fixture
def minimal_perf_cfg_yaml(tmp_path, minimal_perf_cfg_dict):
    p = tmp_path / "perf.yaml"
    p.write_text(yaml.dump(minimal_perf_cfg_dict))
    return p


class TestVllmServingConfig:
    def test_defaults(self):
        c = VllmServingConfig()
        assert c.port == 8080
        assert c.tensor_parallel_size == 1
        assert c.trust_remote_code is True
        assert c.max_num_seqs == 64
        assert c.vllm_env == {}
        assert c.speculative is None

    def test_speculative_config_reuses_existing_model(self):
        c = VllmServingConfig(
            speculative=SpeculativeConfig(
                method="eagle3",
                num_speculative_tokens=5,
                model="s3://bucket/draft/",
            )
        )
        assert c.speculative is not None
        assert c.speculative.method == "eagle3"
        assert c.speculative.num_speculative_tokens == 5

    def test_invalid_dtype_rejected(self):
        with pytest.raises(ValueError, match="Invalid dtype"):
            VllmServingConfig(dtype="bogus")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            VllmServingConfig(unknown_field=True)  # type: ignore[call-arg]


class TestGatlingConfig:
    def test_defaults_require_data_s3_path(self):
        with pytest.raises(ValueError):
            GatlingConfig()  # type: ignore[call-arg]

    def test_data_s3_path_must_be_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            GatlingConfig(data_s3_path="/local/path")

    def test_data_s3_path_trailing_slash_added(self):
        g = GatlingConfig(data_s3_path="s3://bucket/data")
        assert g.data_s3_path.endswith("/")

    def test_default_simulation_class(self):
        """Defaults to an OSS simulation, so a perf run needs no extension."""
        g = GatlingConfig(data_s3_path="s3://bucket/data/")
        assert g.simulation_class == "org.foresightrl.perf.ThrottledTpsCsvSimulation"
        assert g.tps_levels == [1, 2, 3, 4]

    def test_empty_tps_levels_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            GatlingConfig(data_s3_path="s3://bucket/data/", tps_levels=[])

    def test_negative_tps_rejected(self):
        with pytest.raises(ValueError):
            GatlingConfig(data_s3_path="s3://bucket/data/", tps_levels=[0])

    def test_injection_model_defaults_open(self):
        g = GatlingConfig(data_s3_path="s3://bucket/data/")
        assert g.injection_model == "open"

    def test_injection_model_closed_allowed(self):
        g = GatlingConfig(data_s3_path="s3://bucket/data/", injection_model="closed")
        assert g.injection_model == "closed"

    def test_injection_model_invalid_rejected(self):
        with pytest.raises(ValueError, match="injection_model"):
            GatlingConfig(data_s3_path="s3://bucket/data/", injection_model="hybrid")


class TestPerfTestRunConfig:
    def test_invalid_name_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            PerfTestRunConfig(
                name="bad name!",
                gatling=GatlingConfig(data_s3_path="s3://b/d/"),
            )

    def test_smoke_test_mode(self):
        r = PerfTestRunConfig(
            name="smoke",
            gatling=GatlingConfig(data_s3_path="s3://b/d/"),
            run_gatling_test=False,
        )
        assert r.run_gatling_test is False


class TestPerfTestJobConfig:
    def test_from_yaml(self, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        assert c.job.name == "perf-test"
        assert c.perf_test.name == "run1"

    def test_recipe_resolves_perf_test_image(self, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        assert c.container.perf_test_image_uri is not None
        assert "djl-inference" in c.container.perf_test_image_uri

    def test_recipe_resolves_perf_test_requirements(self, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        assert c.container.resolved_requirements_perf_test is not None
        assert len(c.container.resolved_requirements_perf_test) > 0
        assert any(
            "py-markdown-table" in r
            for r in c.container.resolved_requirements_perf_test
        )

    def test_unsupported_backend_rejected(self, tmp_path, minimal_perf_cfg_dict):
        minimal_perf_cfg_dict["execution"]["type"] = "aws_serverless"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(minimal_perf_cfg_dict))
        with pytest.raises(ValueError, match="aws_serverless"):
            PerfTestJobConfig.from_yaml(p)

    def test_merge_overrides_dotted(self, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        c2 = c.merge_overrides(**{"job.name": "renamed"})
        assert c2.job.name == "renamed"
        # Original unchanged
        assert c.job.name == "perf-test"

    def test_merge_overrides_model_name_shortcut(self, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        c2 = c.merge_overrides(model_name="s3://other/model/")
        assert c2.model.name == "s3://other/model/"

    def test_to_yaml_roundtrip(self, tmp_path, minimal_perf_cfg_yaml):
        c = PerfTestJobConfig.from_yaml(minimal_perf_cfg_yaml)
        out = tmp_path / "out.yaml"
        c.to_yaml(out)
        c2 = PerfTestJobConfig.from_yaml(out)
        assert c2.job.name == c.job.name
        assert c2.perf_test.name == c.perf_test.name

    def test_speculative_config_in_yaml(self, tmp_path):
        content = textwrap.dedent("""
            job: {name: spec}
            model: {name: "s3://b/m/"}
            perf_test:
              name: r1
              serving:
                speculative:
                  method: eagle3
                  num_speculative_tokens: 5
                  model: "s3://b/draft/"
                  disable_by_batch_size: 16
              gatling:
                data_s3_path: "s3://b/data/"
            execution: {type: aws_training_job}
            output: {s3_path: "s3://b/out/"}
            container: {recipe: qwen3_5}
            """)
        p = tmp_path / "spec.yaml"
        p.write_text(content)
        c = PerfTestJobConfig.from_yaml(p)
        spec_cfg = c.perf_test.serving.speculative
        assert spec_cfg is not None
        assert spec_cfg.method == "eagle3"
        # `disable_by_batch_size` rides on extra='allow' from SpeculativeConfig.
        assert getattr(spec_cfg, "disable_by_batch_size", None) == 16
