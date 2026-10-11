"""HyperPod EKS — parsed settings dataclass.

Split out from the main package module so the manifest builder mixin
can import ``HyperPodEksSettings`` without triggering a circular import
(``aws_hyperpod_eks/__init__.py`` imports the mixin, the mixin needs
the settings type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HyperPodEksSettings:
    """Parsed settings from execution.hyperpod_eks config block."""

    cluster_name: str = ""
    eks_cluster_name: str = ""
    namespace: str = "kubeflow"
    image_uri: str = ""
    service_account: str = "default"
    fsx_filesystem_id: str = ""
    fsx_mount_path: str = "/fsx"
    storage_class: str = "fsx-sc"
    pvc_name: str = ""
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    restart_policy: str = "OnFailure"
    max_restarts: int = 3
    ssm_relay_instance_id: Optional[str] = None
    kubeconfig_path: Optional[str] = None
