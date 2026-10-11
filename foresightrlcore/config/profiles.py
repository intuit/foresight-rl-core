"""Infrastructure profile presets.

Named presets that expand to topology + vllm_mode defaults.
Explicit field values in the config always override profile defaults.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresightrlcore.config.schema import InfraConfig, TrainingJobConfig

INFRA_PROFILES: dict[str, dict[str, str]] = {
    "single_split_vllm": {"topology": "single_node", "vllm_mode": "server"},
    "torchrun": {"topology": "single_node", "vllm_mode": "none"},
    "colocate": {"topology": "single_node", "vllm_mode": "colocate"},
    "dedicated_vllm": {"topology": "dedicated_vllm", "vllm_mode": "server"},
    "multi_node_server": {"topology": "multi_node_server", "vllm_mode": "server"},
}

GPU_COUNTS: dict[str, int] = {
    "ml.p4d.24xlarge": 8,
    "ml.p4de.24xlarge": 8,
    "ml.p5.48xlarge": 8,
    "ml.g5.xlarge": 1,
    "ml.g5.2xlarge": 1,
    "ml.g5.4xlarge": 1,
    "ml.g5.8xlarge": 1,
    "ml.g5.12xlarge": 4,
    "ml.g5.24xlarge": 4,
    "ml.g5.48xlarge": 8,
    "ml.g6.xlarge": 1,
    "ml.g6.2xlarge": 1,
    "ml.g6.4xlarge": 1,
    "ml.g6.8xlarge": 1,
    "ml.g6.12xlarge": 4,
    "ml.g6.16xlarge": 1,
    "ml.g6.24xlarge": 4,
    "ml.g6.48xlarge": 8,
    "ml.g6e.xlarge": 1,
    "ml.g6e.2xlarge": 1,
    "ml.g6e.4xlarge": 1,
    "ml.g6e.8xlarge": 1,
    "ml.g6e.12xlarge": 4,
    "ml.g6e.24xlarge": 4,
    "ml.g6e.48xlarge": 8,
}


def resolve_infra_profile(infra: "InfraConfig") -> None:
    """Apply profile defaults to InfraConfig in-place.

    Only sets fields that weren't explicitly provided by the user.
    Called from InfraConfig's model_validator.
    """
    if infra.infra_profile is None:
        return

    profile_name = infra.infra_profile
    if profile_name not in INFRA_PROFILES:
        available = list(INFRA_PROFILES.keys())
        raise ValueError(
            f"Unknown infra_profile: '{profile_name}'. Available: {available}"
        )


def gpus_for_instance_type(instance_type: str) -> int:
    """Return GPU count for a known instance type, default 8 for unknown."""
    return GPU_COUNTS.get(instance_type, 8)


# Higher rank → more capable DLC image (CUDA / GPU arch). Used when mixed
# dedicated_vllm types need a single container that runs on both nodes.
_IMAGE_CAPABILITY_RANK: dict[str, int] = {
    "p6": 50,
    "p5": 40,
    "p4d": 30,
    "g6e": 20,
    "g6": 15,
    "g5": 10,
}

_SIZE_LEADING_INT = re.compile(r"^(\d+)")


def gpu_family(instance_type: str) -> str:
    """Return a coarse GPU-architecture family for an instance type.

    ``p4d`` / ``p4de`` collapse to ``p4d`` (A100). ``p5`` / ``p5e`` / ``p5en``
    collapse to ``p5`` (H100/H200). Used to reject accidental mixed-family
    dedicated_vllm clusters that would break TRL NCCL weight sync.
    """
    name = instance_type.removeprefix("ml.").split(".")[0].lower()
    if name.startswith("p4d"):
        return "p4d"
    if name.startswith("p5"):
        return "p5"
    if name.startswith("p6"):
        return "p6"
    return name


def instance_has_efa(instance_type: str) -> bool:
    """Whether this SageMaker instance type exposes EFA.

    p4d/p5/p6 always do. g5 exposes EFA on 8xlarge and larger (SageMaker
    InstanceGroups treat those as EFA even without RDMA). g6/g6e expose EFA
    on 12xlarge and larger. Unknown types are treated as no-EFA so mixed-EFA
    checks fail closed.
    """
    family = gpu_family(instance_type)
    if family in {"p4d", "p5", "p6"}:
        return True
    if family in {"g5", "g6", "g6e"}:
        size = instance_type.rsplit(".", 1)[-1]
        match = _SIZE_LEADING_INT.match(size)
        if match is None:
            return False
        threshold = 8 if family == "g5" else 12
        return int(match.group(1)) >= threshold
    return False


def preferred_image_instance_type(training_type: str, vllm_type: str) -> str:
    """Pick the instance type whose DLC image covers both GPU families.

    Rank is p6 > p5 > p4d > g6e > g6 > g5. Ties keep the training type.
    """
    train_rank = _IMAGE_CAPABILITY_RANK.get(gpu_family(training_type), 0)
    vllm_rank = _IMAGE_CAPABILITY_RANK.get(gpu_family(vllm_type), 0)
    return training_type if train_rank >= vllm_rank else vllm_type


def compute_training_processes(config: "TrainingJobConfig") -> int:
    """Return total training process count (DDP world_size) for this config.

    Mirrors the GPU split logic in entrypoint.sh for every topology so that
    data_points → max_steps derivation matches what the container actually runs.

    Resolution order for GPUs-per-node:
      1. infra.gpus_per_node (explicit user override)
      2. gpus_for_instance_type(infra.instance_type)  (training type; also
         used for mixed dedicated_vllm, where vLLM GPUs are a different type)
      3. len(execution.local.gpus)  (local backend only)
    """
    infra = config.infra
    topology = infra.topology.value
    vllm_mode = infra.vllm_mode.value

    # ── GPU count per node ────────────────────────────────────────────
    exec_type = config.execution.type
    if exec_type == "local":
        local_cfg = config.execution.local or {}
        raw_gpus = local_cfg.get("gpus", [])
        if isinstance(raw_gpus, list):
            gpus = len(raw_gpus)
        else:
            gpus = len(str(raw_gpus).split(","))
    elif infra.gpus_per_node is not None:
        gpus = infra.gpus_per_node
    else:
        gpus = gpus_for_instance_type(infra.instance_type)

    nodes = infra.instance_count

    # ── vllm_mode: none — all GPUs train, across all nodes ───────────
    if vllm_mode == "none":
        return gpus * nodes

    # ── Topology-specific formulas (server / colocate) ────────────────
    if topology == "single_node":
        if vllm_mode == "server":
            # YAML training_gpu_count is the explicit split on this node
            return infra.training_gpu_count
        # colocate: each TP group is one training process
        tp = infra.vllm.tensor_parallel_size
        return (gpus * nodes) // max(tp, 1)

    if topology == "multi_node_server":
        # Every node: (total_gpus − vllm_gpu_count) training procs
        # vLLM runs only on node 0 but the split is symmetric for DDP
        return (gpus - infra.vllm_gpu_count) * nodes

    if topology == "dedicated_vllm":
        if infra.vllm_instance_type is not None:
            # Mixed InstanceGroups: remaining nodes after the vLLM group train.
            # ``gpus`` is already resolved from instance_type (training) or
            # gpus_per_node.
            train_nodes = max(infra.mixed_training_instance_count(), 1)
            return gpus * train_nodes
        # Homogeneous: node 0 all vLLM; remaining nodes all training.
        train_nodes = max(nodes - 1, 1)
        return gpus * train_nodes

    if topology == "multi_node":
        # Legacy heterogeneous SDK v2: separate vLLM + training instance groups
        mn = infra.multi_node
        if mn is not None:
            train_gpus = gpus_for_instance_type(mn.training_instance_type)
            return train_gpus * mn.training_instance_count
        # Fallback: treat like dedicated_vllm
        return gpus * max(nodes - 1, 1)

    # Unknown topology — best-effort: all GPUs on all nodes
    return gpus * nodes
