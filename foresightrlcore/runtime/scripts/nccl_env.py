"""Centralized NCCL environment configuration.

Provides a single function to build the correct set of environment variables
for NCCL transport, timeouts, and debugging across all topology modes.
Called from entrypoint.sh and usable from Python-side setup.

Transport modes:
  - "efa": RDMA over AWS EFA (fast, requires proper SG rules)
  - "sockets": TCP fallback (slower but avoids EFA handshake failures)

Timeout presets vary by topology:
  - colocate/server modes: 30-min NCCL timeout (vLLM init can block)
  - torchrun (no vLLM): 2-hour timeout (generation runs on training ranks)
"""

from __future__ import annotations

import os


def get_nccl_env(
    nccl_transport: str = "efa",
    topology: str = "single_node",
    vllm_mode: str = "server",
    multi_node: bool = False,
    p2p_available: bool = True,
) -> dict[str, str]:
    """Build NCCL environment variables for the given configuration.

    Args:
        nccl_transport: "efa" or "sockets"
        topology: Topology string from config
        vllm_mode: "server", "colocate", or "none"
        multi_node: Whether this is a multi-node setup
        p2p_available: Whether CUDA P2P is supported between GPUs (e.g. NVLink).
            On PCIe-only instances (g5 A10G) this should be False.

    Returns:
        Dict of env var name -> value to set in the process environment.
    """
    env: dict[str, str] = {}

    if not multi_node:
        if vllm_mode == "server" and not p2p_available:
            env["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "1")
            env["NCCL_SHM_DISABLE"] = os.environ.get("NCCL_SHM_DISABLE", "1")
        return env

    if nccl_transport == "sockets":
        env["NCCL_NET"] = "Socket"
        env["NCCL_NET_PLUGIN"] = ""
        env["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "^lo,docker0")
    else:
        env["FI_PROVIDER"] = os.environ.get("FI_PROVIDER", "efa")
        env["FI_EFA_USE_DEVICE_RDMA"] = os.environ.get("FI_EFA_USE_DEVICE_RDMA", "1")
        env["FI_EFA_FORK_SAFE"] = os.environ.get("FI_EFA_FORK_SAFE", "1")
        env["RDMAV_FORK_SAFE"] = os.environ.get("RDMAV_FORK_SAFE", "1")
        nccl_proto = os.environ.get("NCCL_PROTO")
        if nccl_proto:
            env["NCCL_PROTO"] = nccl_proto
        env["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "^lo,docker0")

    if vllm_mode == "none":
        timeout_ms = os.environ.get("NCCL_TIMEOUT", "7200")
        timeout_sec = os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        nccl_debug = os.environ.get("NCCL_DEBUG", "INFO")
        env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
        )
    else:
        timeout_ms = os.environ.get("NCCL_TIMEOUT", "1800000")
        timeout_sec = os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
        nccl_debug = os.environ.get("NCCL_DEBUG", "WARN")

    env["NCCL_TIMEOUT"] = timeout_ms
    env["NCCL_DEBUG"] = nccl_debug
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = timeout_sec
    env["TORCH_NCCL_BLOCKING_WAIT"] = os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "0")

    return env


def apply_nccl_env(
    nccl_transport: str = "efa",
    topology: str = "single_node",
    vllm_mode: str = "server",
    multi_node: bool = False,
    p2p_available: bool = True,
) -> dict[str, str]:
    """Build and apply NCCL env vars to os.environ. Returns the applied dict."""
    env = get_nccl_env(nccl_transport, topology, vllm_mode, multi_node, p2p_available)
    os.environ.update(env)
    return env
