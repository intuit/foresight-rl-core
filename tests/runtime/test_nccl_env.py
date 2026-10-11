"""Tests for foresightrlcore.runtime.scripts.nccl_env — NCCL env configuration."""

from __future__ import annotations

from foresightrlcore.runtime.scripts.nccl_env import apply_nccl_env, get_nccl_env


class TestGetNcclEnv:
    """NCCL environment variable generation."""

    def test_single_node_returns_empty(self):
        env = get_nccl_env(multi_node=False)
        assert env == {}

    def test_single_node_server_no_p2p_disables_p2p_and_shm(self):
        env = get_nccl_env(multi_node=False, vllm_mode="server", p2p_available=False)
        assert env["NCCL_P2P_DISABLE"] == "1"
        assert env["NCCL_SHM_DISABLE"] == "1"

    def test_single_node_server_with_p2p_returns_empty(self):
        env = get_nccl_env(multi_node=False, vllm_mode="server", p2p_available=True)
        assert env == {}

    def test_single_node_no_vllm_no_p2p_returns_empty(self):
        env = get_nccl_env(multi_node=False, vllm_mode="none", p2p_available=False)
        assert env == {}

    def test_multi_node_efa_transport(self, monkeypatch):
        monkeypatch.delenv("NCCL_PROTO", raising=False)
        env = get_nccl_env(nccl_transport="efa", multi_node=True)
        assert env["FI_PROVIDER"] == "efa"
        assert env["FI_EFA_USE_DEVICE_RDMA"] == "1"
        assert env["FI_EFA_FORK_SAFE"] == "1"
        assert "NCCL_TIMEOUT" in env
        assert "NCCL_NET" not in env
        assert "NCCL_PROTO" not in env

    def test_multi_node_efa_nccl_proto_propagated_when_set(self, monkeypatch):
        monkeypatch.setenv("NCCL_PROTO", "Simple")
        env = get_nccl_env(nccl_transport="efa", multi_node=True)
        assert env["NCCL_PROTO"] == "Simple"

    def test_multi_node_sockets_transport(self):
        env = get_nccl_env(nccl_transport="sockets", multi_node=True)
        assert env["NCCL_NET"] == "Socket"
        assert env["NCCL_NET_PLUGIN"] == ""
        assert "FI_PROVIDER" not in env

    def test_torchrun_mode_longer_timeout(self):
        env = get_nccl_env(nccl_transport="efa", vllm_mode="none", multi_node=True)
        assert env["NCCL_TIMEOUT"] == "7200"
        assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "7200"
        assert env["NCCL_DEBUG"] == "INFO"

    def test_server_mode_timeout(self):
        env = get_nccl_env(nccl_transport="efa", vllm_mode="server", multi_node=True)
        assert env["NCCL_TIMEOUT"] == "1800000"
        assert env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "1800"
        assert env["NCCL_DEBUG"] == "WARN"

    def test_colocate_mode_timeout(self):
        env = get_nccl_env(nccl_transport="efa", vllm_mode="colocate", multi_node=True)
        assert env["NCCL_TIMEOUT"] == "1800000"
        assert env["NCCL_DEBUG"] == "WARN"

    def test_blocking_wait_disabled(self):
        env = get_nccl_env(nccl_transport="efa", multi_node=True)
        assert env["TORCH_NCCL_BLOCKING_WAIT"] == "0"

    def test_torchrun_has_async_error_handling(self):
        env = get_nccl_env(vllm_mode="none", multi_node=True)
        assert env.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1"

    def test_server_mode_no_async_error_handling(self):
        env = get_nccl_env(vllm_mode="server", multi_node=True)
        assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in env


class TestApplyNcclEnv:
    """Verify apply_nccl_env sets os.environ."""

    def test_applies_to_environ(self, monkeypatch):
        import os

        monkeypatch.delenv("FI_PROVIDER", raising=False)
        result = apply_nccl_env(nccl_transport="efa", multi_node=True)
        assert os.environ["FI_PROVIDER"] == "efa"
        assert result["FI_PROVIDER"] == "efa"
