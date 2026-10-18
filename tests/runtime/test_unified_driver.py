"""Tests for the unified training driver and supporting infrastructure.

Verifies config loading, method resolution, and structural correctness
of entrypoint.sh, launcher, and train.py — all now under foresightrlcore/.
"""

from __future__ import annotations

import os
import subprocess

import yaml

from foresightrlcore.runtime.scripts.train import _resolve_method_type, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RUNTIME_SCRIPTS = os.path.join(ROOT, "foresightrlcore", "runtime", "scripts")


class TestLoadConfig:
    """Config loading from YAML."""

    def test_loads_from_path(self, tmp_path):
        config = {"job": {"name": "test"}, "model": {"name": "m"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config))
        result = load_config(str(path))
        assert result["job"]["name"] == "test"

    def test_missing_config_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")


class TestResolveMethodType:
    """Method type extraction from config dict."""

    def test_explicit_grpo(self):
        config = {"method": {"type": "grpo"}}
        assert _resolve_method_type(config) == "grpo"

    def test_explicit_sft(self):
        config = {"method": {"type": "sft"}}
        assert _resolve_method_type(config) == "sft"

    def test_default_to_grpo(self):
        config: dict[str, object] = {}
        assert _resolve_method_type(config) == "grpo"

    def test_missing_method_key(self):
        config = {"job": {"name": "test"}}
        assert _resolve_method_type(config) == "grpo"


class TestEntrypointReferences:
    """Verify entrypoint.sh uses TRAIN_SCRIPT variable pointing to train.py."""

    def test_entrypoint_uses_train_script_var(self):
        entrypoint_path = os.path.join(RUNTIME_SCRIPTS, "entrypoint.sh")
        with open(entrypoint_path) as f:
            content = f.read()

        assert "TRAIN_SCRIPT=" in content
        assert "${TRAIN_SCRIPT}" in content

    def test_entrypoint_points_to_train_py(self):
        entrypoint_path = os.path.join(RUNTIME_SCRIPTS, "entrypoint.sh")
        with open(entrypoint_path) as f:
            content = f.read()

        assert "train.py" in content
        assert "train_grpo.py" not in content


class TestSingleNodePcieNcclTransport:
    """Single-node PCIe (P2P-disabled) hosts must not fall through
    to the DLC's Libfabric/EFA plugin, which selects NET/OFI even with no EFA
    device present and kills vLLM weight sync ("NET/OFI ... RC: 103. Error:
    4126 (Unresponsive receiver)"). The guard must pin NCCL onto the socket
    transport, but only when the operator hasn't already set an override.
    """

    @staticmethod
    def _no_p2p_block():
        entrypoint_path = os.path.join(RUNTIME_SCRIPTS, "entrypoint.sh")
        with open(entrypoint_path) as f:
            content = f.read()

        marker = "        export NCCL_SHM_DISABLE=1"
        start = content.index(marker)
        end = content.index('echo "NCCL net transport:', start)
        end = content.index("\n", end) + 1
        return content[start:end]

    @classmethod
    def _run_block(cls, env_overrides):
        """Execute the actual guard block under sh, with the given env
        preset, and return the resulting NCCL_NET_PLUGIN/NCCL_NET values.
        This exercises real runtime behavior instead of just checking that
        guard and assignment strings both appear somewhere nearby, so an
        assignment that escaped its own guard would be caught.
        """
        script = (
            "set -eu\n"
            "gpu_p2p_available() { return 1; }\n"
            + cls._no_p2p_block()
            + '\nprintf "NCCL_NET_PLUGIN=%s\\nNCCL_NET=%s\\n" '
            '"${NCCL_NET_PLUGIN:-}" "${NCCL_NET:-}"\n'
        )
        env = {**os.environ, **env_overrides}
        result = subprocess.run(
            ["sh", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_pins_socket_transport_when_unset(self):
        values = self._run_block({})
        assert values["NCCL_NET_PLUGIN"] == "none"
        assert values["NCCL_NET"] == "Socket"

    def test_respects_full_operator_override(self):
        values = self._run_block({"NCCL_NET_PLUGIN": "ofi", "NCCL_NET": "efa"})
        assert values["NCCL_NET_PLUGIN"] == "ofi"
        assert values["NCCL_NET"] == "efa"

    def test_respects_net_plugin_override_independently(self):
        values = self._run_block({"NCCL_NET_PLUGIN": "ofi"})
        assert values["NCCL_NET_PLUGIN"] == "ofi"
        assert values["NCCL_NET"] == "Socket"

    def test_respects_net_override_independently(self):
        values = self._run_block({"NCCL_NET": "efa"})
        assert values["NCCL_NET_PLUGIN"] == "none"
        assert values["NCCL_NET"] == "efa"


class TestRuntimeScriptsComplete:
    """Verify all required runtime scripts are present."""

    def test_train_py_exists(self):
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "train.py"))

    def test_entrypoint_exists(self):
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "entrypoint.sh"))

    def test_nccl_env_exists(self):
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "nccl_env.py"))

    def test_requirements_exist(self):
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "requirements.txt"))
        assert os.path.isfile(
            os.path.join(RUNTIME_SCRIPTS, "requirements_torchrun.txt")
        )

    def test_deepspeed_configs_exist(self):
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "ds_zero2_config.json"))
        assert os.path.isfile(os.path.join(RUNTIME_SCRIPTS, "ds_zero3_config.json"))

    def test_no_python_plugin_dirs_in_runtime_scripts(self):
        """Tasks and clients live in foresightrlcore.tasks / .clients, not here."""
        assert not os.path.isdir(os.path.join(RUNTIME_SCRIPTS, "tasks"))
        assert not os.path.isdir(os.path.join(RUNTIME_SCRIPTS, "clients"))


class TestLauncherReferences:
    """Verify the canonical launcher uses train.py."""

    def test_launcher_uses_train_py(self):
        launcher_path = os.path.join(
            ROOT, "foresightrlcore", "execution", "aws_sagemaker_launcher.py"
        )
        with open(launcher_path) as f:
            content = f.read()

        assert '"train.py"' in content

    def test_launcher_scripts_dir_points_to_runtime(self):
        launcher_path = os.path.join(
            ROOT, "foresightrlcore", "execution", "aws_sagemaker_launcher.py"
        )
        with open(launcher_path) as f:
            content = f.read()

        assert "runtime" in content
        assert "scripts" in content
