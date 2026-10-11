"""Tests for foresightrlcore.execution.aws_hyperpod_eks — AwsHyperPodEksBackend.

Focus is on the pure-Python surface that builds Kubernetes manifests and
bash command lists; the actual `subprocess.run("kubectl ...")` /
`boto3.client("s3").upload_file(...)` calls are mocked so we never touch a
real cluster or AWS account. The handful of SSM-tunnel tests use
``socket.create_connection`` and ``subprocess.Popen`` mocks for the same
reason.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.schema import (
    Topology,
    TrainingJobConfig,
    VllmMode,
)
from foresightrlcore.execution.aws_hyperpod_eks import (
    AwsHyperPodEksBackend,
    HyperPodEksSettings,
)

# ─── Fixtures ────────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _make_config(overrides: dict | None = None) -> TrainingJobConfig:
    """Build a minimally-valid TrainingJobConfig targeting hyperpod_eks."""
    base = {
        "job": {"name": "hp-test"},
        "model": {"name": "Qwen/Qwen3-8B"},
        "method": {"type": "grpo"},
        "training": {"grpo": {"num_generations": 4}},
        "infra": {
            "instance_type": "ml.p4d.24xlarge",
            "instance_count": 2,
            "topology": "dedicated_vllm",
            "vllm_mode": "server",
            "vllm": {
                "port": 8000,
                "tensor_parallel_size": 8,
                "data_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": False,
            },
        },
        "data": {"train_path": "s3://bucket/path/train.jsonl"},
        "output": {"s3_path": "s3://bucket/output"},
        "execution": {
            "type": "aws_hyperpod_eks",
            "hyperpod_eks": {
                "cluster_name": "test-cluster",
                "eks_cluster_name": "test-eks",
                "namespace": "default",
                "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/test:latest",
                "service_account": "default",
                "fsx_filesystem_id": "",
                "fsx_mount_path": "/fsx",
                "pvc_name": "fsx-claim",
                "restart_policy": "OnFailure",
                "max_restarts": 3,
            },
        },
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
        "environment": {"HF_HOME": "/fsx/hf_cache"},
    }
    if overrides:
        _deep_merge(base, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base, f)
        path = f.name
    try:
        return TrainingJobConfig.from_yaml(path)
    finally:
        os.unlink(path)


@pytest.fixture
def backend():
    return AwsHyperPodEksBackend()


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def settings(backend, config):
    return backend._parse_settings(config)


# ─── _get_gpus_for_instance ──────────────────────────────────────────────


class TestGetGpusForInstance:
    @pytest.mark.parametrize(
        "instance_type,expected",
        [
            ("ml.p4d.24xlarge", 8),
            ("ml.p5.48xlarge", 8),
            ("ml.g5.12xlarge", 4),
            ("ml.g5.24xlarge", 4),
            ("ml.g6e.48xlarge", 8),
            ("p4d.24xlarge", 8),
            ("p5.48xlarge", 8),
        ],
    )
    def test_known_instance(self, backend, instance_type, expected):
        assert backend._get_gpus_for_instance(instance_type) == expected

    def test_strips_ml_prefix(self, backend):
        # "ml.unknown" isn't in the table, but "unknown" might map after stripping
        # — in practice unknown falls through to the default 8.
        assert backend._get_gpus_for_instance("ml.does_not_exist") == 8

    def test_unknown_defaults_to_eight(self, backend):
        assert backend._get_gpus_for_instance("c5.large") == 8


# ─── _get_efa_for_instance ───────────────────────────────────────────────


class TestGetEfaForInstance:
    @pytest.mark.parametrize(
        "instance_type,expected",
        [
            ("ml.p4d.24xlarge", 4),
            ("ml.p5.48xlarge", 32),
            ("ml.p5en.48xlarge", 32),
            ("p4d.24xlarge", 4),
            ("p5.48xlarge", 32),
        ],
    )
    def test_known_instance(self, backend, instance_type, expected):
        assert backend._get_efa_for_instance(instance_type) == expected

    def test_unknown_defaults_to_four(self, backend):
        assert backend._get_efa_for_instance("ml.does_not_exist") == 4


# ─── _get_source_s3_uri ──────────────────────────────────────────────────


class TestGetSourceS3Uri:
    def test_with_s3_prefix(self, backend, config):
        uri = backend._get_source_s3_uri(config)
        assert uri == "s3://bucket/output/hyperpod-code/sourcedir.tar.gz"

    def test_with_trailing_slash_normalized(self, backend):
        cfg = _make_config({"output": {"s3_path": "s3://bucket/output/"}})
        uri = backend._get_source_s3_uri(cfg)
        # Helper should not produce a double-slash in the key portion
        assert uri == "s3://bucket/output/hyperpod-code/sourcedir.tar.gz"

    def test_bucket_only_no_prefix(self, backend):
        cfg = _make_config({"output": {"s3_path": "s3://bucket"}})
        uri = backend._get_source_s3_uri(cfg)
        assert uri == "s3://bucket/hyperpod-code/sourcedir.tar.gz"

    def test_fsx_path_raises(self, backend):
        """Regression: /fsx/... paths used to silently parse as bucket=''.

        See the FILM-5272 PR review (#4). The advice example YAML had this
        shape — `_upload_source_to_s3` would have crashed inside boto3 with
        an opaque error. We now raise early at parse time.
        """
        cfg = _make_config({"output": {"s3_path": "/fsx/output/grpo-advice-hyperpod"}})
        with pytest.raises(ValueError, match="output.s3_path"):
            backend._get_source_s3_uri(cfg)

    def test_bare_bucket_path_raises(self, backend):
        cfg = _make_config({"output": {"s3_path": "bucket/path"}})
        with pytest.raises(ValueError, match="s3://"):
            backend._get_source_s3_uri(cfg)


# ─── _write_resolved_requirements ───────────────────────────────────────


class TestWriteResolvedRequirements:
    """Mirrors the SageMaker launcher's recipe-resolution priority order.

    Priority: inline ``container.requirements`` > resolved recipe deps >
    HyperPod EKS default (qwen3). The HyperPod-specific default avoids a
    foot-gun: without it the backend would silently fall through to
    ``runtime/scripts/requirements.txt``, which is synced to qwen3_5
    (transformers>=5.6) and crashes the trl-0.28.x stack at import time.
    """

    def test_inline_requirements_win(self, backend, tmp_path):
        cfg = _make_config({"container": {"requirements": ["pkg-a==1.0", "pkg-b>=2"]}})
        backend._write_resolved_requirements(cfg, str(tmp_path))
        content = (tmp_path / "requirements.txt").read_text().splitlines()
        assert content == ["pkg-a==1.0", "pkg-b>=2"]

    def test_recipe_resolution(self, backend, tmp_path):
        cfg = _make_config({"container": {"recipe": "qwen3"}})
        backend._write_resolved_requirements(cfg, str(tmp_path))
        lines = (tmp_path / "requirements.txt").read_text().splitlines()
        # Trl is exact-pinned; transformers carries the 4.x lower bound.
        # Don't lock the test against the exact transformers spec — qwen3.txt
        # is meant to track master, so the constraint may evolve.
        assert any("trl[vllm]==0.28.0" in ln for ln in lines)
        assert any(ln.startswith("transformers") for ln in lines)

    def test_default_qwen3_when_no_recipe(self, backend, tmp_path, caplog):
        cfg = _make_config({})
        assert cfg.container.recipe is None
        assert cfg.container.requirements is None
        with caplog.at_level("WARNING"):
            backend._write_resolved_requirements(cfg, str(tmp_path))
        lines = (tmp_path / "requirements.txt").read_text().splitlines()
        # The qwen3 recipe pins are present
        assert any("trl[vllm]==0.28.0" in ln for ln in lines)
        assert any(ln.startswith("transformers") for ln in lines)
        # A clear warning is emitted
        assert any("defaulting to 'qwen3'" in rec.message for rec in caplog.records)

    def test_no_container_attribute_is_noop(self, backend, tmp_path):
        # Defensive: if a caller passes a stripped-down object without a
        # ``container`` attribute, the helper just returns without writing.
        class _Stub:
            pass

        backend._write_resolved_requirements(_Stub(), str(tmp_path))
        assert not (tmp_path / "requirements.txt").exists()


# ─── _parse_settings ─────────────────────────────────────────────────────


class TestParseSettings:
    def test_full_config(self, backend, config):
        s = backend._parse_settings(config)
        assert s.cluster_name == "test-cluster"
        assert s.namespace == "default"
        assert s.image_uri.startswith("123.")
        assert s.pvc_name == "fsx-claim"
        assert s.max_restarts == 3

    def test_ssm_relay_no_implicit_kubeconfig(self, backend):
        # Setting ssm_relay_instance_id alone no longer injects a default
        # ~/.kube/hyperpod-ssm path. The launcher refuses to mutate the
        # user's home directory; callers must point kubeconfig_path at a
        # real file (typically intuit/examples/kubeconfigs/<cluster>.yaml).
        cfg = _make_config(
            {"execution": {"hyperpod_eks": {"ssm_relay_instance_id": "i-abc"}}}
        )
        s = backend._parse_settings(cfg)
        assert s.kubeconfig_path is None
        assert s.ssm_relay_instance_id == "i-abc"

    def test_explicit_kubeconfig_expanded(self, backend):
        cfg = _make_config(
            {
                "execution": {
                    "hyperpod_eks": {
                        "ssm_relay_instance_id": "i-abc",
                        "kubeconfig_path": "~/my-kube",
                    }
                }
            }
        )
        s = backend._parse_settings(cfg)
        # ~ should be expanded to an absolute path
        assert s.kubeconfig_path is not None
        assert not s.kubeconfig_path.startswith("~")
        # And the result is absolute (resolved against cwd if relative).
        assert Path(s.kubeconfig_path).is_absolute()

    def test_relative_kubeconfig_resolved_to_absolute(self, backend, tmp_path):
        # Relative paths should be resolved to absolute against cwd so
        # every kubectl invocation uses the same file regardless of where
        # the launcher recurses to.
        kc = tmp_path / "kc.yaml"
        kc.write_text("apiVersion: v1\nkind: Config\n")
        rel = os.path.relpath(str(kc))
        cfg = _make_config(
            {
                "execution": {
                    "hyperpod_eks": {
                        "ssm_relay_instance_id": "i-abc",
                        "kubeconfig_path": rel,
                    }
                }
            }
        )
        s = backend._parse_settings(cfg)
        assert s.kubeconfig_path is not None
        assert Path(s.kubeconfig_path).is_absolute()
        assert Path(s.kubeconfig_path).resolve() == kc.resolve()

    def test_no_hyperpod_eks_section_returns_defaults(self, backend):
        cfg_mock = MagicMock()
        cfg_mock.execution.hyperpod_eks = None
        s = backend._parse_settings(cfg_mock)
        assert s == HyperPodEksSettings()


# ─── validate ────────────────────────────────────────────────────────────


class TestValidate:
    def test_happy_path(self, backend, config):
        with patch.object(backend, "_kubectl_available", return_value=True):
            errors = backend.validate(config)
        # boto3 may or may not be installed — accept either, but topology/image
        # should be clean
        assert not any("HyperPod EKS backend requires" in e for e in errors)
        assert not any("kubectl is not available" in e for e in errors)

    def test_missing_kubectl(self, backend, config):
        with patch.object(backend, "_kubectl_available", return_value=False):
            errors = backend.validate(config)
        assert any("kubectl is not available" in e for e in errors)

    def test_dedicated_vllm_with_single_instance_fails(self, backend):
        cfg = _make_config({"infra": {"instance_count": 1}})
        with patch.object(backend, "_kubectl_available", return_value=True):
            errors = backend.validate(cfg)
        assert any("dedicated_vllm topology requires" in e for e in errors)

    def test_vllm_mode_server_with_invalid_topology_fails(self, backend):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "multi_node",
                    "vllm_mode": "server",
                    "instance_count": 2,
                }
            }
        )
        with patch.object(backend, "_kubectl_available", return_value=True):
            errors = backend.validate(cfg)
        assert any("vllm_mode=server with topology=multi_node" in e for e in errors)

    def test_boto3_missing_returns_early(self, backend, config):
        # Force the `import boto3` inside validate() to raise.
        with patch.dict("sys.modules", {"boto3": None}):
            errors = backend.validate(config)
        # First error mentions boto3, AND we return after that — so the
        # kubectl/topology error messages should NOT show.
        assert errors == [
            "boto3 is not installed. Install with: uv pip install -e '.[aws]'"
        ]

    def test_missing_image_uri_caught(self, backend, config):
        # Mutate the parsed config to drop image_uri
        config.execution.hyperpod_eks.image_uri = ""
        with patch.object(backend, "_kubectl_available", return_value=True):
            errors = backend.validate(config)
        assert any("image_uri" in e for e in errors)


# ─── _kubectl_available / _is_tunnel_up ──────────────────────────────────


class TestKubectlAvailability:
    def test_kubectl_present(self, backend):
        mock = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock):
            assert backend._kubectl_available() is True

    def test_kubectl_missing(self, backend):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert backend._kubectl_available() is False

    def test_kubectl_command_fails(self, backend):
        mock = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock):
            assert backend._kubectl_available() is False


class TestTunnelUp:
    def test_port_reachable(self, backend):
        sock = MagicMock()
        sock.__enter__ = MagicMock(return_value=sock)
        sock.__exit__ = MagicMock(return_value=False)
        with patch("socket.create_connection", return_value=sock):
            assert backend._is_tunnel_up() is True

    def test_port_not_reachable(self, backend):
        with patch("socket.create_connection", side_effect=OSError):
            assert backend._is_tunnel_up() is False


# ─── _run_kubectl ─────────────────────────────────────────────────────────


class TestRunKubectl:
    def test_no_settings_uses_default_env(self, backend):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            backend._run_kubectl(["get", "pods"])
            args, kwargs = mock_run.call_args
            assert args[0][:2] == ["kubectl", "get"]
            # Env was passed through (we copy os.environ)
            assert "env" in kwargs

    def test_settings_with_kubeconfig_injects_env(self, backend):
        s = HyperPodEksSettings(kubeconfig_path="/tmp/kube")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            backend._run_kubectl(["get", "pods"], settings=s)
            kwargs = mock_run.call_args.kwargs
            assert kwargs["env"]["KUBECONFIG"] == "/tmp/kube"


# ─── _get_eks_endpoint ───────────────────────────────────────────────────


class TestGetEksEndpoint:
    def test_from_kubeconfig_tls_name(self, backend, tmp_path):
        kc = tmp_path / "kubeconfig"
        kc.write_text(
            yaml.dump(
                {
                    "clusters": [
                        {
                            "cluster": {
                                "server": "https://localhost:6443",
                                "tls-server-name": "real-eks.us-east-1.eks.amazonaws.com",
                            },
                        }
                    ]
                }
            )
        )
        s = HyperPodEksSettings(kubeconfig_path=str(kc))
        assert backend._get_eks_endpoint(s) == "real-eks.us-east-1.eks.amazonaws.com"

    def test_from_kubeconfig_server_url(self, backend, tmp_path):
        kc = tmp_path / "kubeconfig"
        kc.write_text(
            yaml.dump(
                {
                    "clusters": [
                        {
                            "cluster": {
                                "server": "https://real-eks.us-east-1.eks.amazonaws.com:443",
                            },
                        }
                    ]
                }
            )
        )
        s = HyperPodEksSettings(kubeconfig_path=str(kc))
        assert backend._get_eks_endpoint(s) == "real-eks.us-east-1.eks.amazonaws.com"

    def test_skips_localhost_server(self, backend, tmp_path):
        # When the kubeconfig points at the SSM tunnel localhost,
        # we should fall through to the next strategy.
        kc = tmp_path / "kubeconfig"
        kc.write_text(
            yaml.dump(
                {
                    "clusters": [
                        {
                            "cluster": {
                                "server": "https://localhost:6443",
                                # No tls-server-name either
                            },
                        }
                    ]
                }
            )
        )
        s = HyperPodEksSettings(kubeconfig_path=str(kc))
        # No describe-cluster fallback either → raises
        with pytest.raises(RuntimeError, match="Cannot determine EKS API endpoint"):
            backend._get_eks_endpoint(s)

    def test_raises_when_no_source_works(self, backend):
        s = HyperPodEksSettings()
        with pytest.raises(RuntimeError, match="Cannot determine EKS API endpoint"):
            backend._get_eks_endpoint(s)


# ─── _build_container_spec / _build_pod_spec ─────────────────────────────


class TestBuildContainerSpec:
    def test_basic_structure(self, backend, config, settings):
        spec = backend._build_container_spec(
            config, settings, role="vllm", gpus=8, command=["sleep", "1"]
        )
        assert spec["name"] == "pytorch"  # required by Kubeflow operator
        assert spec["image"] == settings.image_uri
        assert spec["imagePullPolicy"] == "Always"
        assert spec["command"] == ["sleep", "1"]
        assert spec["resources"]["limits"]["nvidia.com/gpu"] == 8
        # EFA count looked up by instance type — ml.p4d.24xlarge → 4
        assert spec["resources"]["limits"]["vpc.amazonaws.com/efa"] == 4

    def test_role_env_var_propagated(self, backend, config, settings):
        spec = backend._build_container_spec(
            config, settings, role="vllm", gpus=8, command=[]
        )
        env_map = {e["name"]: e["value"] for e in spec["env"]}
        assert env_map["ROLE"] == "vllm"
        assert env_map["TOPOLOGY"] == "dedicated_vllm"
        assert env_map["VLLM_MODE"] == "server"

    def test_user_environment_propagated(self, backend, config, settings):
        spec = backend._build_container_spec(
            config, settings, role="training", gpus=4, command=[]
        )
        env_map = {e["name"]: e["value"] for e in spec["env"]}
        assert env_map["HF_HOME"] == "/fsx/hf_cache"

    def test_baseline_nccl_env_present(self, backend, config, settings):
        spec = backend._build_container_spec(
            config, settings, role="training", gpus=4, command=[]
        )
        env_map = {e["name"]: e["value"] for e in spec["env"]}
        assert env_map["NCCL_DEBUG"] == "INFO"
        assert env_map["FI_PROVIDER"] == "efa"
        assert env_map["NCCL_SOCKET_IFNAME"] == "eth0"

    def test_volume_mounts_includes_shm_and_fsx_when_pvc(
        self, backend, config, settings
    ):
        spec = backend._build_container_spec(
            config, settings, role="training", gpus=4, command=[]
        )
        names = [v["name"] for v in spec["volumeMounts"]]
        assert "shm" in names
        # PVC is configured → fsx-volume present
        assert "fsx-volume" in names

    def test_no_fsx_volume_mount_when_neither_pvc_nor_filesystem_id(
        self, backend, config
    ):
        bare = HyperPodEksSettings(image_uri="img")
        spec = backend._build_container_spec(
            config, bare, role="training", gpus=4, command=[]
        )
        names = [v["name"] for v in spec["volumeMounts"]]
        assert "shm" in names
        assert "fsx-volume" not in names


class TestBuildPodSpec:
    def test_with_pvc_name(self, backend, settings):
        container = {"name": "pytorch"}
        spec = backend._build_pod_spec(container, settings, "ml.p4d.24xlarge", 8)
        vol_names = [v["name"] for v in spec["volumes"]]
        assert "shm" in vol_names
        assert "fsx-volume" in vol_names
        fsx = next(v for v in spec["volumes"] if v["name"] == "fsx-volume")
        assert fsx["persistentVolumeClaim"]["claimName"] == "fsx-claim"

    def test_with_fsx_filesystem_id(self, backend):
        s = HyperPodEksSettings(image_uri="img", fsx_filesystem_id="fs-abc")
        spec = backend._build_pod_spec({"name": "pytorch"}, s, "ml.p4d.24xlarge", 8)
        fsx = next(v for v in spec["volumes"] if v["name"] == "fsx-volume")
        assert fsx["persistentVolumeClaim"]["claimName"] == "fsx-fs-abc"

    def test_no_fsx_volume_when_neither_configured(self, backend):
        s = HyperPodEksSettings(image_uri="img")
        spec = backend._build_pod_spec({"name": "pytorch"}, s, "ml.p4d.24xlarge", 8)
        vol_names = [v["name"] for v in spec["volumes"]]
        assert "fsx-volume" not in vol_names

    def test_node_selector_has_instance_type(self, backend, settings):
        spec = backend._build_pod_spec(
            {"name": "pytorch"}, settings, "ml.p5.48xlarge", 8
        )
        assert (
            spec["nodeSelector"]["node.kubernetes.io/instance-type"] == "ml.p5.48xlarge"
        )

    def test_extra_node_selector_merged(self, backend):
        s = HyperPodEksSettings(image_uri="img", node_selector={"my-pool": "training"})
        spec = backend._build_pod_spec({"name": "pytorch"}, s, "ml.p4d.24xlarge", 8)
        assert spec["nodeSelector"]["my-pool"] == "training"

    def test_gpu_toleration_always_added(self, backend, settings):
        spec = backend._build_pod_spec(
            {"name": "pytorch"}, settings, "ml.p4d.24xlarge", 8
        )
        assert any(
            t.get("key") == "nvidia.com/gpu" and t.get("effect") == "NoSchedule"
            for t in spec["tolerations"]
        )

    def test_service_account_set(self, backend, settings):
        spec = backend._build_pod_spec(
            {"name": "pytorch"}, settings, "ml.p4d.24xlarge", 8
        )
        assert spec["serviceAccountName"] == "default"


# ─── Command builders ────────────────────────────────────────────────────


class TestBuildVllmCommand:
    def test_basic(self, backend, config):
        cmd = backend._build_vllm_command(config)
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        body = cmd[2]
        # vLLM is launched via vllm_launcher.py (single entry point shared
        # with SageMaker and local backends) — not `trl vllm-serve` directly.
        assert "python /tmp/code/vllm_launcher.py" in body
        assert "trl vllm-serve" not in body
        assert "--model Qwen/Qwen3-8B" in body
        assert "--port 8000" in body
        assert "--tensor-parallel-size 8" in body
        assert "--data-parallel-size 1" in body
        # Source S3 download
        assert "s3://bucket/output/hyperpod-code/sourcedir.tar.gz" in body
        # GPUSTATS sidecar present
        assert "GPUSTATS" in body

    def test_enforce_eager(self, backend):
        cfg = _make_config({"infra": {"vllm": {"enforce_eager": True}}})
        body = backend._build_vllm_command(cfg)[2]
        assert "--enforce-eager" in body

    def test_enforce_eager_off(self, backend, config):
        body = backend._build_vllm_command(config)[2]
        # The literal "--enforce-eager" flag must not appear when disabled
        assert "--enforce-eager" not in body

    def test_patch_env_vars_exported_when_yaml_opts_in(self, backend):
        """Reasoning/compilation/disable opt-ins must reach the launcher
        via bash `export VAR=...` lines in the heredoc."""
        cfg = _make_config(
            {
                "infra": {
                    "vllm": {
                        "reasoning_parser": "qwen3",
                        "reasoning_config": {
                            "reasoning_start_str": "<think>",
                            "reasoning_end_str": "</think>",
                        },
                        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
                        "disable_custom_all_reduce": True,
                    }
                }
            }
        )
        body = backend._build_vllm_command(cfg)[2]
        # Each opt-in becomes an `export` line before the launcher invocation
        assert "export VLLM_REASONING_CONFIG_JSON=" in body
        assert "export VLLM_COMPILATION_CONFIG_JSON=" in body
        assert "export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1" in body
        # Verify the exports come BEFORE the launcher line — order matters
        launcher_idx = body.index("python /tmp/code/vllm_launcher.py")
        assert body.index("export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1") < launcher_idx
        assert body.index("export VLLM_REASONING_CONFIG_JSON=") < launcher_idx
        assert body.index("export VLLM_COMPILATION_CONFIG_JSON=") < launcher_idx

    def test_no_patch_exports_when_no_opt_in(self, backend, config):
        """Default config has no opt-ins → no exports clutter the heredoc."""
        body = backend._build_vllm_command(config)[2]
        assert "VLLM_REASONING_CONFIG_JSON" not in body
        assert "VLLM_COMPILATION_CONFIG_JSON" not in body
        assert "VLLM_DISABLE_CUSTOM_ALL_REDUCE" not in body


class TestBuildTorchrunCommand:
    def test_uses_torchrun_path_and_instance_count(self, backend, settings):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "none",
                    "instance_count": 4,
                }
            }
        )
        body = backend._build_torchrun_command(cfg, settings)[2]
        assert "/usr/local/bin/torchrun" in body
        assert "--nnodes=1:4" in body
        # nproc_per_node should equal GPUs for the instance (p4d → 8)
        assert "--nproc_per_node=8" in body
        # Requirements path branches on `requirements_torchrun.txt` existence
        assert "requirements_torchrun.txt" in body


class TestBuildTrainingCommandSeparateVllm:
    def test_basic(self, backend, config, settings):
        cmd = backend._build_training_command_separate_vllm(
            config, settings, vllm_host="my-vllm", vllm_port=8000
        )
        body = cmd[2]
        assert "my-vllm:8000" in body
        assert "torchrun" in body
        # Post-training: model sync + vLLM cleanup
        assert "aws s3 sync /opt/ml/model" in body
        assert "GPUSTATS" in body

    def test_post_training_uses_correct_k8s_api_urls(self, backend, config, settings):
        """Regression: services must hit /api/v1, deployments /apis/apps/v1.

        Earlier versions hit /apis/v1 for services, which is a non-existent
        route — Kubernetes 404'd silently and the orphan Service remained.
        """
        body = backend._build_training_command_separate_vllm(
            config, settings, vllm_host="my-vllm", vllm_port=8000
        )[2]
        # The snippet computes URL_BASE from KIND — both literals must appear
        assert "/api/v1" in body  # core resources (services)
        assert "/apis/apps/v1" in body  # deployments
        # The wrong route must not appear
        assert "/apis/v1/namespaces" not in body

    def test_worker_0_gating_for_cleanup(self, backend, config, settings):
        body = backend._build_training_command_separate_vllm(
            config, settings, vllm_host="vllm-x", vllm_port=8000
        )[2]
        assert '"$HOSTNAME" == *-worker-0' in body

    def test_nnodes_is_count_minus_one(self, backend, settings):
        cfg = _make_config({"infra": {"instance_count": 9}})
        body = backend._build_training_command_separate_vllm(
            cfg, settings, vllm_host="vllm-x", vllm_port=8000
        )[2]
        # 9 nodes minus 1 vLLM dedicated node → 8 training workers
        assert "--nnodes=1:8" in body


# ─── _build_*_manifests / _build_manifest ────────────────────────────────


class TestBuildDedicatedVllmManifests:
    def test_returns_three_manifests(self, backend, config, settings):
        out = backend._build_dedicated_vllm_manifests(config, settings, "job-1")
        assert len(out) == 3
        kinds = [m["kind"] for m in out]
        assert kinds == ["Deployment", "Service", "PyTorchJob"]

    def test_vllm_service_is_headless(self, backend, config, settings):
        out = backend._build_dedicated_vllm_manifests(config, settings, "job-1")
        svc = next(m for m in out if m["kind"] == "Service")
        # Headless service required for TRL's NCCL TCPStore on group_port=51216
        assert svc["spec"]["clusterIP"] == "None"

    def test_pytorchjob_has_elastic_policy(self, backend, config, settings):
        out = backend._build_dedicated_vllm_manifests(config, settings, "job-1")
        job = next(m for m in out if m["kind"] == "PyTorchJob")
        assert job["spec"]["elasticPolicy"]["rdzvBackend"] == "c10d"
        # Worker replicas = instance_count - 1 (one node hosts vLLM)
        assert (
            job["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"]
            == config.infra.instance_count - 1
        )

    def test_vllm_service_port_matches_config(self, backend, config, settings):
        out = backend._build_dedicated_vllm_manifests(config, settings, "job-1")
        svc = next(m for m in out if m["kind"] == "Service")
        assert svc["spec"]["ports"][0]["port"] == config.infra.vllm.port


class TestBuildPyTorchJob:
    def test_single_node(self, backend, settings):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_count": 1,
                }
            }
        )
        m = backend._build_pytorchjob(cfg, settings, "j-1")
        assert m["kind"] == "PyTorchJob"
        assert m["metadata"]["labels"]["topology"] == "single_node"
        assert m["spec"]["elasticPolicy"]["rdzvBackend"] == "c10d"

    def test_multi_node_none_mode(self, backend, settings):
        # vllm_mode=server on multi_node_server is now rejected up-front
        # (see test_vllm_mode_server_raises_for_non_dedicated_topology in
        # TestBuildStandardWorkerSpec). Use vllm_mode=none here to keep
        # coverage of the multi-node manifest shape.
        cfg = _make_config(
            {
                "infra": {
                    "topology": "multi_node_server",
                    "vllm_mode": "none",
                    "instance_count": 4,
                }
            }
        )
        m = backend._build_pytorchjob(cfg, settings, "j-1")
        assert m["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] == 4


class TestBuildManifestDispatch:
    def test_dedicated_vllm_returns_list(self, backend, config, settings):
        out = backend._build_manifest(config, settings)
        assert isinstance(out, list)
        assert len(out) == 3

    def test_single_node_returns_pytorchjob(self, backend):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_count": 1,
                }
            }
        )
        s = HyperPodEksSettings(image_uri="img")
        out = backend._build_manifest(cfg, s)
        assert isinstance(out, dict)
        assert out["kind"] == "PyTorchJob"

    def test_job_name_includes_timestamp_and_grpo_prefix(
        self, backend, config, settings
    ):
        # First Deployment manifest carries the job_name (vllm suffix appended)
        out = backend._build_manifest(config, settings)
        deployment = out[0]
        name = deployment["metadata"]["name"]
        assert name.startswith("grpo-hp-test-")
        assert name.endswith("-vllm")


class TestBuildStandardWorkerSpec:
    def test_vllm_mode_none_uses_torchrun(self, backend):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "none",
                    "instance_count": 2,
                }
            }
        )
        s = HyperPodEksSettings(image_uri="img")
        spec = backend._build_standard_worker_spec(cfg, s)
        cmd = spec["Worker"]["template"]["spec"]["containers"][0]["command"]
        assert "torchrun" in cmd[2]

    def test_vllm_mode_colocate_uses_torchrun(self, backend):
        # vllm_mode=colocate runs torchrun (vLLM is in-process; no separate
        # endpoint to wait for). This is the canonical single_node path.
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_count": 1,
                }
            }
        )
        s = HyperPodEksSettings(image_uri="img")
        spec = backend._build_standard_worker_spec(cfg, s)
        cmd = spec["Worker"]["template"]["spec"]["containers"][0]["command"]
        assert "torchrun" in cmd[2]

    def test_vllm_mode_server_raises_for_non_dedicated_topology(self, backend):
        # vllm_mode=server is only supported via the DEDICATED_VLLM split.
        # The PR review (#6) flagged this as the inherently-misconfigured
        # combo; we now raise loudly at manifest-build time instead of
        # silently emitting a pod that waits forever on a vLLM URL no one
        # serves.
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "server",
                    "instance_count": 1,
                }
            }
        )
        s = HyperPodEksSettings(image_uri="img")
        with pytest.raises(NotImplementedError, match="topology='dedicated_vllm'"):
            backend._build_standard_worker_spec(cfg, s)


# ─── _apply_manifest / _cleanup_vllm_deployment ──────────────────────────


class TestApplyManifest:
    def test_happy_path(self, backend, settings):
        manifest = {"kind": "Service", "metadata": {"name": "x"}}
        with patch.object(
            backend,
            "_run_kubectl",
            return_value=MagicMock(returncode=0, stdout="service/x created", stderr=""),
        ) as run:
            backend._apply_manifest(manifest, settings=settings)
            run.assert_called_once()

    def test_failure_raises(self, backend, settings):
        manifest = {"kind": "Service", "metadata": {"name": "x"}}
        with patch.object(
            backend,
            "_run_kubectl",
            return_value=MagicMock(returncode=1, stdout="", stderr="forbidden"),
        ):
            with pytest.raises(RuntimeError, match="kubectl apply failed"):
                backend._apply_manifest(manifest, settings=settings)


class TestCleanupVllmDeployment:
    def test_deletes_both_kinds(self, backend, settings):
        with patch.object(
            backend, "_run_kubectl", return_value=MagicMock(returncode=0, stderr="")
        ) as run:
            backend._cleanup_vllm_deployment("job-1", "default", settings)
            # One call per kind (deployment + service)
            assert run.call_count == 2

    def test_warns_on_failure_but_does_not_raise(self, backend, settings):
        with patch.object(
            backend,
            "_run_kubectl",
            return_value=MagicMock(returncode=1, stderr="not found"),
        ):
            # Should not raise — we only log
            backend._cleanup_vllm_deployment("job-1", "default", settings)


# ─── get_status ──────────────────────────────────────────────────────────


class TestGetStatus:
    def test_happy_path(self, backend):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="Succeeded\n")
            out = backend.get_status("my-job")
            assert out == {"job_id": "my-job", "status": "Succeeded"}

    def test_kubectl_failure(self, backend):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stdout="")
            out = backend.get_status("my-job")
            assert out["status"] == "Unknown"

    def test_subprocess_raises_returns_unknown_with_error(self, backend):
        with patch("subprocess.run", side_effect=Exception("boom")):
            out = backend.get_status("my-job")
            assert out["status"] == "Unknown"
            assert "boom" in out["error"]


# ─── SSM tunnel ──────────────────────────────────────────────────────────


class TestSsmTunnel:
    def test_no_op_when_no_relay_configured(self, backend):
        s = HyperPodEksSettings()
        # No relay → method returns without touching subprocess
        with patch("subprocess.Popen") as popen:
            backend._ensure_ssm_tunnel(s)
            popen.assert_not_called()

    def test_no_op_when_tunnel_already_up(self, backend):
        s = HyperPodEksSettings(
            ssm_relay_instance_id="i-abc",
            kubeconfig_path="/tmp/no-such",
        )
        with (
            patch.object(backend, "_is_tunnel_up", return_value=True),
            patch("subprocess.Popen") as popen,
        ):
            backend._ensure_ssm_tunnel(s)
            popen.assert_not_called()

    def test_starts_tunnel_when_not_up(self, backend, tmp_path):
        # Provide a kubeconfig with the EKS endpoint so _get_eks_endpoint works.
        kc = tmp_path / "kubeconfig"
        kc.write_text(
            yaml.dump(
                {
                    "clusters": [
                        {
                            "cluster": {
                                "server": "https://eks.example.com:443",
                            },
                        }
                    ]
                }
            )
        )
        s = HyperPodEksSettings(
            ssm_relay_instance_id="i-abc",
            kubeconfig_path=str(kc),
        )

        states = [False, True]  # Tunnel comes up after the second probe
        fake_proc = MagicMock()
        with (
            patch.object(
                backend, "_is_tunnel_up", side_effect=lambda *_a, **_kw: states.pop(0)
            ),
            patch("subprocess.Popen", return_value=fake_proc) as popen,
            patch("time.sleep"),
        ):
            backend._ensure_ssm_tunnel(s)
            popen.assert_called_once()
            # Check the SSM start-session args
            args = popen.call_args.args[0]
            assert args[:3] == ["aws", "ssm", "start-session"]
            assert "--target" in args
            assert "i-abc" in args
        assert backend._ssm_proc is fake_proc
        assert backend._ssm_tunnel_started_by_us is True

    def test_teardown_terminates_if_started_by_us(self, backend):
        proc = MagicMock()
        backend._ssm_proc = proc
        backend._ssm_tunnel_started_by_us = True
        backend._teardown_ssm_tunnel()
        proc.terminate.assert_called_once()
        assert backend._ssm_proc is None

    def test_teardown_no_op_if_not_started_by_us(self, backend):
        proc = MagicMock()
        backend._ssm_proc = proc
        backend._ssm_tunnel_started_by_us = False
        backend._teardown_ssm_tunnel()
        proc.terminate.assert_not_called()


# ─── launch() smoke ──────────────────────────────────────────────────────


class TestLaunchSmoke:
    def test_launch_dedicated_vllm_submits_three_manifests(self, backend, config):
        """End-to-end launch with all IO mocked.

        Verifies: settings parsed → tunnel managed → manifest built → all
        three resources applied → job_name returned.
        """
        with (
            patch.object(backend, "_ensure_ssm_tunnel"),
            patch.object(backend, "_teardown_ssm_tunnel"),
            patch.object(
                backend,
                "_upload_source_to_s3",
                return_value="s3://b/sourcedir.tar.gz",
            ),
            patch.object(backend, "_apply_manifest") as apply_m,
        ):
            job_id = backend.launch(config, wait=False)
        # Three manifests for dedicated_vllm
        assert apply_m.call_count == 3
        assert job_id.startswith("grpo-hp-test-")

    def test_launch_single_node_submits_one_manifest(self, backend):
        cfg = _make_config(
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_count": 1,
                }
            }
        )
        with (
            patch.object(backend, "_ensure_ssm_tunnel"),
            patch.object(backend, "_teardown_ssm_tunnel"),
            patch.object(
                backend, "_upload_source_to_s3", return_value="s3://b/x.tar.gz"
            ),
            patch.object(backend, "_apply_manifest") as apply_m,
        ):
            job_id = backend.launch(cfg, wait=False)
        assert apply_m.call_count == 1
        assert job_id.startswith("grpo-hp-test-")


# ─── Module-level registration ──────────────────────────────────────────


class TestBackendRegistration:
    def test_register_backend_invoked_at_import(self):
        from foresightrlcore.execution import get_backend

        # get_backend returns the class itself (consistent with the rest of
        # the registry — see execution/__init__.py); construct it to verify.
        cls = get_backend("aws_hyperpod_eks")
        assert cls is AwsHyperPodEksBackend
        assert isinstance(cls(), AwsHyperPodEksBackend)

