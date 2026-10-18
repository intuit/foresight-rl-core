"""HyperPod EKS — Kubernetes manifest construction.

This module owns the pure-function side of the backend: turning a
validated ``TrainingJobConfig`` + parsed ``HyperPodEksSettings`` into
the Kubernetes resource dictionaries that ``kubectl apply`` consumes.

Everything here is exposed as a ``ManifestsBackendMixin`` rather than
as module-level functions so the existing test surface keeps working
unchanged — ``tests/execution/test_aws_hyperpod_eks_backend.py`` calls
``backend._build_X(...)`` on instances of ``AwsHyperPodEksBackend`` in
dozens of places, and inheriting the methods preserves that contract.

The mixin assumes the concrete subclass provides:

  * ``self._get_source_s3_uri(config) -> str``  — used by the bash
    commands that ``aws s3 cp`` the source tarball at pod startup.

That method lives on the main backend class because it shares the
``_parse_output_s3_path`` parsing logic with the source-upload path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from foresightrlcore.config.schema import Topology, VllmMode
from foresightrlcore.execution.sagemaker_common import extra_packages_env

from .settings import HyperPodEksSettings

logger = logging.getLogger(__name__)


class ManifestsBackendMixin:
    """Manifest-building methods for ``AwsHyperPodEksBackend``.

    Split out from the main backend class purely for file-size reasons
    (the bulk of the backend is YAML construction). All methods are
    pure functions of ``(config, settings, ...)`` with no instance
    state — but kept as bound methods so existing tests that call
    ``backend._build_X(...)`` continue to work unchanged.
    """

    def _build_manifest(
        self, config: Any, settings: HyperPodEksSettings
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Build Kubernetes manifest(s) for the training job.

        For dedicated_vllm topology, returns a list of manifests:
        - Deployment for vLLM server
        - Service for vLLM (so training can discover it)
        - PyTorchJob for training nodes

        For other topologies, returns a single PyTorchJob manifest.
        """
        import time

        timestamp = int(time.time())
        job_name = f"grpo-{config.job.name}-{timestamp}"

        if config.infra.topology == Topology.DEDICATED_VLLM:
            return self._build_dedicated_vllm_manifests(config, settings, job_name)

        return self._build_pytorchjob(config, settings, job_name)

    def _build_dedicated_vllm_manifests(
        self, config: Any, settings: HyperPodEksSettings, job_name: str
    ) -> List[Dict[str, Any]]:
        """Build separate Kubernetes resources for dedicated_vllm topology.

        Following AWS best practice from awsome-distributed-training repo:
        vLLM runs as a separate Deployment, training runs as PyTorchJob.
        This avoids the distributed env var injection issues of PyTorchJob
        when master/worker have different roles.

        Returns list of manifests:
        1. Deployment for vLLM server
        2. Service for vLLM (ClusterIP for training to discover)
        3. PyTorchJob for training (workers only, no master)
        """
        instance_type = config.infra.instance_type
        gpus_per_node = self._get_gpus_for_instance(instance_type)
        training_nodes = config.infra.instance_count - 1
        vllm_port = config.infra.vllm.port

        base_labels = {
            "app": "foresight-rl",
            "job-name": config.job.name,
            "job-id": job_name,
            "topology": "dedicated_vllm",
        }

        vllm_name = f"{job_name}-vllm"
        vllm_container = self._build_container_spec(
            config,
            settings,
            role="vllm",
            gpus=gpus_per_node,
            command=self._build_vllm_command(config),
        )
        vllm_deployment: Dict[str, Any] = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": vllm_name,
                "namespace": settings.namespace,
                "labels": {**base_labels, "component": "vllm"},
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "foresight-rl",
                        "job-id": job_name,
                        "component": "vllm",
                    },
                },
                "template": {
                    "metadata": {
                        "labels": {**base_labels, "component": "vllm"},
                    },
                    "spec": self._build_pod_spec(
                        vllm_container, settings, instance_type, gpus_per_node
                    ),
                },
            },
        }

        vllm_service: Dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": vllm_name,
                "namespace": settings.namespace,
                "labels": {**base_labels, "component": "vllm"},
            },
            "spec": {
                # Headless service (clusterIP: None) — DNS resolves directly to
                # the vLLM pod's IP. This is required for TRL's vLLM client:
                # after the HTTP handshake on `vllm_port`, TRL opens a NCCL
                # process group via TCPStore on `group_port` (default 51216),
                # which a regular ClusterIP would not forward. With a headless
                # service, training reaches the vLLM pod directly on any port.
                "clusterIP": "None",
                "selector": {
                    "app": "foresight-rl",
                    "job-id": job_name,
                    "component": "vllm",
                },
                "ports": [
                    {"name": "vllm-http", "port": vllm_port, "targetPort": vllm_port},
                ],
            },
        }

        training_container = self._build_container_spec(
            config,
            settings,
            role="training",
            gpus=gpus_per_node,
            command=self._build_training_command_separate_vllm(
                config, settings, vllm_host=vllm_name, vllm_port=vllm_port
            ),
        )

        # Single Worker replica group (no Master) + elasticPolicy with c10d
        # rendezvous. The Kubeflow training operator only injects torch
        # distributed env vars (PET_NNODES, PET_NODE_RANK, PET_RDZV_BACKEND,
        # PET_RDZV_ENDPOINT, etc.) when elasticPolicy is set; without it, only
        # PET_NNODES is set and torchrun has no way to discover peers.
        # rdzvBackend: c10d uses PyTorch's built-in TCPStore (no external etcd).
        # See intuit/docs/hyperpod-eks-setup-notes.md "Distributed Training Pattern" for
        # the A2 vs B decision and links to the AWS FSDP2 reference manifest.
        training_job: Dict[str, Any] = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": job_name,
                "namespace": settings.namespace,
                "labels": {**base_labels, "component": "training"},
                "annotations": {
                    "sagemaker.amazonaws.com/enable-job-auto-resume": "true",
                    "sagemaker.amazonaws.com/max-retry": str(settings.max_restarts),
                },
            },
            "spec": {
                "elasticPolicy": {
                    "rdzvBackend": "c10d",
                    "minReplicas": 1,
                    "maxReplicas": training_nodes,
                    "maxRestarts": settings.max_restarts,
                },
                "pytorchReplicaSpecs": {
                    "Worker": {
                        "replicas": training_nodes,
                        "restartPolicy": settings.restart_policy,
                        "template": {
                            "metadata": {
                                "labels": {**base_labels, "component": "training"},
                            },
                            "spec": self._build_pod_spec(
                                training_container,
                                settings,
                                instance_type,
                                gpus_per_node,
                            ),
                        },
                    },
                },
                "runPolicy": {
                    "cleanPodPolicy": "Running",
                    "backoffLimit": settings.max_restarts,
                },
            },
        }

        return [vllm_deployment, vllm_service, training_job]

    def _build_training_command_separate_vllm(
        self, config: Any, settings: HyperPodEksSettings, vllm_host: str, vllm_port: int
    ) -> List[str]:
        """Build training command when vLLM runs as separate Deployment.

        Similar to AWS awsome-distributed-training pattern:
        - Training knows vLLM host via service name
        - Single-node training when only 1 training node
        - Multi-node distributed when >1 training nodes

        Post-training steps (run on rank 0 / single node):
        - Sync /opt/ml/model to output.s3_path via aws s3 sync
        - Delete the vLLM Deployment + Service via the k8s API (using the
          pod's service account token — requires the foresight-rl-worker RBAC
          Role to be bound to the default service account)
        """
        gpus = self._get_gpus_for_instance(config.infra.instance_type)
        source_s3 = self._get_source_s3_uri(config)
        training_nodes = config.infra.instance_count - 1
        train_data_s3 = str(config.data.train_path)
        output_s3 = str(config.output.s3_path).rstrip("/")

        # Post-training: S3 sync + vLLM Deployment cleanup.
        #
        # NOTE on rank detection: under torchrun, RANK / LOCAL_RANK / WORLD_SIZE
        # are set inside each Python child process, NOT in the parent bash.
        # After torchrun exits, those vars are unset in this shell. So we use
        # HOSTNAME-based gating instead:
        #   - S3 model sync: every pod attempts it, but only the global-rank-0
        #     pod actually has model files in /opt/ml/model (TRL only saves on
        #     main_process). The empty-dir guard makes non-rank-0 pods no-op.
        #     `aws s3 sync` is idempotent, so a no-op is harmless.
        #   - vLLM Deployment delete: only worker-0 attempts it. Worker-0 is
        #     the c10d rendezvous host by Kubeflow convention, so it's the
        #     stable "leader" for cleanup. If it crashes before this point,
        #     the orphan vLLM Deployment must be cleaned up manually.
        # K8s API URL roots:
        #   - core resources (services, pods, configmaps): /api/v1/...
        #   - extension resources (deployments under apps/v1): /apis/apps/v1/...
        # Earlier versions of this snippet used /apis/v1/... for services, which
        # is a non-existent route — silent 404 on cleanup, leaving an orphan
        # Service after every dedicated_vllm run.
        post_training_snippet = f"""
# --- Post-training ---
echo "Training command exited with code $?"

# Diagnostic: list TensorBoard event files before sync so we can verify they
# exist (or get a clear breadcrumb if they don't). Only meaningful on the
# rank-0 pod; other pods will just see an empty dir and that's fine.
if [ -d /opt/ml/model/runs ]; then
    echo "TensorBoard runs/ directory contents:"
    find /opt/ml/model/runs -type f | head -20
    echo "TensorBoard event file count: $(find /opt/ml/model/runs -name 'events.out.tfevents.*' | wc -l)"
fi

if [ -d /opt/ml/model ] && [ -n "$(ls -A /opt/ml/model 2>/dev/null)" ]; then
    echo "Model directory non-empty (this pod hosted global rank 0). Syncing to S3..."
    aws s3 sync /opt/ml/model {output_s3}/model/ --quiet \\
        && echo "Model synced to {output_s3}/model/"
fi

if [[ "$HOSTNAME" == *-worker-0 ]]; then
    echo "worker-0: cleaning up vLLM Deployment..."
    KUBE_TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
    KUBE_NS=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
    KUBE_API="https://kubernetes.default.svc"
    VLLM_NAME="{vllm_host}"
    for KIND in deployments services; do
        if [ "$KIND" = "deployments" ]; then
            URL_BASE="$KUBE_API/apis/apps/v1"
        else
            URL_BASE="$KUBE_API/api/v1"
        fi
        RESP=$(curl -s -o /dev/null -w "%{{http_code}}" -X DELETE \\
            --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \\
            -H "Authorization: Bearer $KUBE_TOKEN" \\
            "$URL_BASE/namespaces/$KUBE_NS/$KIND/$VLLM_NAME")
        echo "DELETE $KIND/$VLLM_NAME → HTTP $RESP"
    done
fi
"""

        # torchrun + c10d rendezvous handles all distributed env var wiring.
        # The Kubeflow operator with elasticPolicy.rdzvBackend=c10d injects
        # PET_NNODES, PET_NODE_RANK, PET_RDZV_BACKEND, PET_RDZV_ENDPOINT,
        # PET_NPROC_PER_NODE — torchrun reads them natively. No need to plumb
        # WORLD_SIZE/RANK/MASTER_ADDR ourselves.
        # `--nnodes=1:N` is the elastic range syntax: tolerates 1..N nodes.
        return [
            "/bin/bash",
            "-c",
            f"""
set -e
echo "Downloading source code from S3..."
aws s3 cp {source_s3} /tmp/sourcedir.tar.gz
mkdir -p /tmp/code && tar -xzf /tmp/sourcedir.tar.gz -C /tmp/code
cd /tmp/code

echo "Installing dependencies on $(hostname)..."
pip install --quiet -r requirements.txt
echo "Downloading training data from S3..."
mkdir -p /opt/ml/input/data/train
aws s3 cp {train_data_s3} /opt/ml/input/data/train/

echo "Waiting for vLLM server at {vllm_host}:{vllm_port}..."
for i in $(seq 1 90); do
    if curl -s "http://{vllm_host}:{vllm_port}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server... ($i/90)"
    sleep 10
done

export VLLM_SERVER_URL="http://{vllm_host}:{vllm_port}"

# Background GPU monitor: prints one [GPUSTATS] line per GPU every 30s to
# stdout. Lets us reconstruct util / power / mem timeline from `kubectl logs
# ... | grep GPUSTATS`. Self-contained, untrapped, & -detached: failures here
# never affect training. Killed automatically when the container exits.
(
    while true; do
        TS=$(date -u +%H:%M:%S)
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \\
            --format=csv,noheader,nounits 2>/dev/null \\
            | awk -F'[, ]+' -v ts="$TS" -v host="$(hostname)" \\
                '{{print "[GPUSTATS] " ts " " host " gpu="$1" util="$2"% mem="$3"MiB pow="$4"W"}}' \\
            || true
        sleep 30
    done
) &
GPUSTATS_PID=$!

echo "Starting distributed training via torchrun: nnodes=1:{training_nodes}, nproc_per_node={gpus}"
echo "  Distributed env (operator-injected): PET_NNODES=${{PET_NNODES:-?}} PET_NODE_RANK=${{PET_NODE_RANK:-?}} PET_RDZV_ENDPOINT=${{PET_RDZV_ENDPOINT:-?}}"
/usr/local/bin/torchrun \\
    --nproc_per_node={gpus} \\
    --nnodes=1:{training_nodes} \\
    /tmp/code/train.py --config args.yaml --vllm-host {vllm_host}
{post_training_snippet}""",
        ]

    def _build_pytorchjob(
        self, config: Any, settings: HyperPodEksSettings, job_name: str
    ) -> Dict[str, Any]:
        """Build standard Kubeflow PyTorchJob manifest.

        Exercised for ``single_node`` and ``multi_node_server`` topologies —
        ``DEDICATED_VLLM`` is routed to ``_build_dedicated_vllm_manifests`` in
        ``_build_manifest``.

        ``elasticPolicy.rdzvBackend=c10d`` makes the Kubeflow operator inject
        the PET_RDZV_* env vars that torchrun needs for peer discovery. See
        ``intuit/docs/hyperpod-eks-setup-notes.md`` "Distributed Training Pattern"
        for the rationale (A2 vs B).
        """
        topology = config.infra.topology
        worker_spec = self._build_standard_worker_spec(config, settings)

        worker_replicas = worker_spec.get("Worker", {}).get(
            "replicas", config.infra.instance_count
        )

        manifest: Dict[str, Any] = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": job_name,
                "namespace": settings.namespace,
                "labels": {
                    "app": "foresight-rl",
                    "job-name": config.job.name,
                    "topology": topology.value,
                    "vllm-mode": config.infra.vllm_mode.value,
                },
                "annotations": {
                    "sagemaker.amazonaws.com/enable-job-auto-resume": "true",
                    "sagemaker.amazonaws.com/max-retry": str(settings.max_restarts),
                },
            },
            "spec": {
                "elasticPolicy": {
                    "rdzvBackend": "c10d",
                    "minReplicas": 1,
                    "maxReplicas": worker_replicas,
                    "maxRestarts": settings.max_restarts,
                },
                "pytorchReplicaSpecs": worker_spec,
                "runPolicy": {
                    "cleanPodPolicy": "Running",
                    "backoffLimit": settings.max_restarts,
                },
            },
        }

        return manifest

    def _build_standard_worker_spec(
        self, config: Any, settings: HyperPodEksSettings
    ) -> Dict[str, Any]:
        """Build replica specs for standard multi-node training.

        Single homogeneous Worker replica group (no Master). Together with the
        elasticPolicy.rdzvBackend=c10d on the parent PyTorchJob, this matches
        the AWS FSDP2 EKS reference pattern (see
        intuit/docs/hyperpod-eks-setup-notes.md "Distributed Training Pattern").
        """
        instance_type = config.infra.instance_type
        gpus_per_node = self._get_gpus_for_instance(instance_type)
        vllm_mode = config.infra.vllm_mode
        topology = config.infra.topology

        # vllm_mode=server is only meaningful when the manifest itself stands
        # up a separate vLLM endpoint — that's the DEDICATED_VLLM path
        # (routed away in ``_build_manifest``). For ``single_node`` /
        # ``multi_node_server`` we'd need a sidecar/role-based split that
        # this backend has never implemented. Fail loudly rather than emit a
        # manifest that waits forever on a nonexistent vLLM URL.
        if vllm_mode == VllmMode.SERVER:
            raise NotImplementedError(
                f"vllm_mode='server' on HyperPod EKS is only supported with "
                f"topology='dedicated_vllm' (which uses a separate vLLM "
                f"Deployment + Service). Got topology='{topology.value}'. "
                f"Use 'dedicated_vllm' for split vLLM, 'colocate' for "
                f"in-process vLLM, or 'none' to disable vLLM."
            )

        # vllm_mode in {NONE, COLOCATE} — both run torchrun; for COLOCATE the
        # trainer process starts vLLM internally (no separate endpoint needed).
        command = self._build_torchrun_command(config, settings)

        container = self._build_container_spec(
            config, settings, role="worker", gpus=gpus_per_node, command=command
        )

        return {
            "Worker": {
                "replicas": config.infra.instance_count,
                "restartPolicy": settings.restart_policy,
                "template": {
                    "spec": self._build_pod_spec(
                        container, settings, instance_type, gpus_per_node
                    ),
                },
            },
        }

    def _build_container_spec(
        self,
        config: Any,
        settings: HyperPodEksSettings,
        role: str,
        gpus: int,
        command: List[str],
    ) -> Dict[str, Any]:
        """Build container specification."""
        env_vars: List[Dict[str, str]] = [
            {"name": "NCCL_DEBUG", "value": "INFO"},
            {"name": "FI_PROVIDER", "value": "efa"},
            {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "1"},
            {"name": "NCCL_PROTO", "value": "simple"},
            {"name": "RDMAV_FORK_SAFE", "value": "1"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
        ]

        for key, value in config.environment.items():
            env_vars.append({"name": key, "value": str(value)})

        for key, value in extra_packages_env(config.container.extra_packages).items():
            env_vars.append({"name": key, "value": value})

        env_vars.append({"name": "ROLE", "value": role})
        env_vars.append({"name": "TOPOLOGY", "value": config.infra.topology.value})
        env_vars.append({"name": "VLLM_MODE", "value": config.infra.vllm_mode.value})

        volume_mounts: List[Dict[str, str]] = []
        if settings.pvc_name or settings.fsx_filesystem_id:
            volume_mounts.append(
                {
                    "name": "fsx-volume",
                    "mountPath": settings.fsx_mount_path,
                }
            )
        volume_mounts.append({"name": "shm", "mountPath": "/dev/shm"})

        container: Dict[str, Any] = {
            "name": "pytorch",  # Must be named 'pytorch' for Kubeflow operator
            "image": settings.image_uri,
            "imagePullPolicy": "Always",
            "command": command,
            "env": env_vars,
            "resources": {
                "limits": {
                    "nvidia.com/gpu": gpus,
                    "vpc.amazonaws.com/efa": self._get_efa_for_instance(
                        config.infra.instance_type
                    ),
                },
                "requests": {
                    "nvidia.com/gpu": gpus,
                    "vpc.amazonaws.com/efa": self._get_efa_for_instance(
                        config.infra.instance_type
                    ),
                },
            },
            "volumeMounts": volume_mounts,
        }

        return container

    def _build_pod_spec(
        self,
        container: Dict[str, Any],
        settings: HyperPodEksSettings,
        instance_type: str,
        gpus: int,
    ) -> Dict[str, Any]:
        """Build pod specification with volumes, node selector, etc."""
        volumes: List[Dict[str, Any]] = [
            {
                "name": "shm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
            }
        ]

        if settings.pvc_name:
            volumes.append(
                {
                    "name": "fsx-volume",
                    "persistentVolumeClaim": {"claimName": settings.pvc_name},
                }
            )
        elif settings.fsx_filesystem_id:
            volumes.append(
                {
                    "name": "fsx-volume",
                    "persistentVolumeClaim": {
                        "claimName": f"fsx-{settings.fsx_filesystem_id}"
                    },
                }
            )

        node_selector: Dict[str, str] = dict(settings.node_selector)
        # HyperPod nodes have ml. prefix in instance type label
        node_selector["node.kubernetes.io/instance-type"] = instance_type

        tolerations: List[Dict[str, str]] = [dict(t) for t in settings.tolerations]
        tolerations.append(
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        )

        pod_spec: Dict[str, Any] = {
            "serviceAccountName": settings.service_account,
            "containers": [container],
            "volumes": volumes,
            "nodeSelector": node_selector,
            "tolerations": tolerations,
            # hostNetwork disabled for now - may need to enable for EFA
            # "hostNetwork": True,
            # "dnsPolicy": "ClusterFirstWithHostNet",
        }

        return pod_spec

    def _build_vllm_command(self, config: Any) -> List[str]:
        """Build command to start vLLM server.

        vLLM is launched via ``vllm_launcher.py`` (not ``trl vllm-serve``
        directly) so spawn-safety env vars and optional TRL source patches
        are applied consistently with the SageMaker / local backends. See
        ``foresightrlcore/runtime/scripts/vllm_launcher.py`` for the
        contract. The launcher file is copied into the source tarball
        (``runtime/scripts/`` contents land at ``/tmp/code/`` root), so the
        path below is stable.
        """
        vllm_cfg = config.infra.vllm
        source_s3 = self._get_source_s3_uri(config)
        # Patch opt-ins are forwarded via env vars the launcher reads at
        # startup. Empty values are filtered out so the launcher's no-op
        # fast path stays free of malformed input.
        patch_env_lines = self._build_vllm_patch_env_lines(vllm_cfg)
        patch_env_block = "\n".join(patch_env_lines)
        return [
            "/bin/bash",
            "-c",
            f"""
set -e
echo "Downloading source code from S3..."
aws s3 cp {source_s3} /tmp/sourcedir.tar.gz
mkdir -p /tmp/code && tar -xzf /tmp/sourcedir.tar.gz -C /tmp/code
cd /tmp/code

echo "Installing dependencies on $(hostname)..."
pip install --quiet -r requirements.txt

# Background GPU monitor: see comment in _build_training_command_separate_vllm.
# Same loop on the vLLM pod gives us a parallel timeline of generation-side
# GPU saturation, which combined with the training-side timeline lets us
# determine which phase is the actual bottleneck.
(
    while true; do
        TS=$(date -u +%H:%M:%S)
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F'[, ]+' -v ts="$TS" -v host="$(hostname)" \
                '{{print "[GPUSTATS] " ts " " host " gpu="$1" util="$2"% mem="$3"MiB pow="$4"W"}}' \
            || true
        sleep 30
    done
) &

echo "Starting vLLM server on $(hostname)"
echo "  vLLM parallelism: TP={vllm_cfg.tensor_parallel_size}, DP={vllm_cfg.data_parallel_size}, total GPUs used = {vllm_cfg.tensor_parallel_size * vllm_cfg.data_parallel_size}"
{patch_env_block}
python /tmp/code/vllm_launcher.py \
    --model {config.model.name} \
    --host 0.0.0.0 \
    --port {vllm_cfg.port} \
    --tensor-parallel-size {vllm_cfg.tensor_parallel_size} \
    --data-parallel-size {vllm_cfg.data_parallel_size} \
    --gpu-memory-utilization {vllm_cfg.gpu_memory_utilization} \
    {"--enforce-eager" if vllm_cfg.enforce_eager else ""}
""",
        ]

    @staticmethod
    def _build_vllm_patch_env_lines(vllm_cfg: Any) -> List[str]:
        """Render ``infra.vllm.*`` patch opt-ins as ``export VAR=...`` lines.

        Mirrors ``LocalBackend._build_vllm_patch_env`` but emits bash
        ``export`` statements suitable for inlining into the heredoc.
        Returns an empty list when no opt-ins are active so the heredoc
        stays clean.
        """
        import json as _json
        import shlex

        lines: List[str] = []
        reasoning_config = getattr(vllm_cfg, "reasoning_config", None)
        if reasoning_config:
            val = _json.dumps(reasoning_config, separators=(",", ":"))
            lines.append(f"export VLLM_REASONING_CONFIG_JSON={shlex.quote(val)}")
        compilation_config = getattr(vllm_cfg, "compilation_config", None)
        if compilation_config:
            val = _json.dumps(compilation_config, separators=(",", ":"))
            lines.append(f"export VLLM_COMPILATION_CONFIG_JSON={shlex.quote(val)}")
        if getattr(vllm_cfg, "disable_custom_all_reduce", False):
            lines.append("export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1")
        return lines

    def _build_torchrun_command(
        self, config: Any, settings: HyperPodEksSettings
    ) -> List[str]:
        """Build torchrun command for vllm_mode=none.

        Uses requirements_torchrun.txt (lighter deps, no vLLM) like sibling repo.

        No explicit --nnodes/--node_rank/--master_addr/--master_port flags:
        these are auto-populated by torchrun from the PET_* env vars that the
        Kubeflow operator injects when the parent PyTorchJob has
        elasticPolicy.rdzvBackend=c10d configured. `--nnodes=1:N` is the
        elastic range syntax — accepts 1 to instance_count nodes.
        """
        gpus = self._get_gpus_for_instance(config.infra.instance_type)
        source_s3 = self._get_source_s3_uri(config)
        instance_count = config.infra.instance_count
        return [
            "/bin/bash",
            "-c",
            f"""
set -e
echo "Downloading source code from S3..."
aws s3 cp {source_s3} /tmp/sourcedir.tar.gz
mkdir -p /tmp/code && tar -xzf /tmp/sourcedir.tar.gz -C /tmp/code
cd /tmp/code

echo "Installing dependencies on $(hostname)..."
if [ -f requirements_torchrun.txt ]; then
    pip install --quiet -r requirements_torchrun.txt
else
    pip install --quiet -r requirements.txt
fi

echo "Starting torchrun: nnodes=1:{instance_count}, nproc_per_node={gpus}"
echo "  Distributed env (operator-injected): PET_NNODES=${{PET_NNODES:-?}} PET_NODE_RANK=${{PET_NODE_RANK:-?}} PET_RDZV_ENDPOINT=${{PET_RDZV_ENDPOINT:-?}}"
/usr/local/bin/torchrun \\
    --nproc_per_node={gpus} \\
    --nnodes=1:{instance_count} \\
    /tmp/code/train.py --config args.yaml
""",
        ]

    def _get_gpus_for_instance(self, instance_type: str) -> int:
        """Get GPU count for an instance type."""
        gpu_map = {
            "ml.p4d.24xlarge": 8,
            "ml.p4de.24xlarge": 8,
            "ml.p5.48xlarge": 8,
            "ml.p5e.48xlarge": 8,
            "ml.p5en.48xlarge": 8,
            "ml.g5.12xlarge": 4,
            "ml.g5.24xlarge": 4,
            "ml.g5.48xlarge": 8,
            "ml.g6e.12xlarge": 4,
            "ml.g6e.24xlarge": 4,
            "ml.g6e.48xlarge": 8,
            "p4d.24xlarge": 8,
            "p4de.24xlarge": 8,
            "p5.48xlarge": 8,
        }
        clean_type = instance_type.replace("ml.", "")
        return gpu_map.get(instance_type, gpu_map.get(clean_type, 8))

    def _get_efa_for_instance(self, instance_type: str) -> int:
        """Get EFA adapter count for an instance type."""
        efa_map = {
            "ml.p4d.24xlarge": 4,
            "ml.p4de.24xlarge": 4,
            "ml.p5.48xlarge": 32,
            "ml.p5e.48xlarge": 32,
            "ml.p5en.48xlarge": 32,
            "p4d.24xlarge": 4,
            "p5.48xlarge": 32,
        }
        clean_type = instance_type.replace("ml.", "")
        return efa_map.get(instance_type, efa_map.get(clean_type, 4))

    # Type stub for the method that lives on the concrete backend class.
    # Declared here so mypy doesn't complain about the `self._get_source_s3_uri`
    # references in the bash command builders above. The real implementation
    # is on AwsHyperPodEksBackend.
    def _get_source_s3_uri(self, config: Any) -> str:  # pragma: no cover
        raise NotImplementedError(
            "Mixin _get_source_s3_uri stub — overridden by AwsHyperPodEksBackend"
        )
