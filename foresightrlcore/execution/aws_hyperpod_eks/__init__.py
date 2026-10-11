"""AWS SageMaker HyperPod EKS execution backend.

Submits training jobs to a HyperPod EKS cluster using the Kubeflow
``PyTorchJob`` CRD (``kubeflow.org/v1``). For the ``dedicated_vllm``
topology, the backend additionally emits a standalone vLLM ``Deployment``
+ ``Service`` so the trainer can hit a stable endpoint. The HyperPod
``HyperPodPyTorchJob`` CRD path (Pattern B) is *not* used today — see
``intuit/docs/hyperpod-eks-setup-notes.md`` "Distributed Training Pattern".

It supports multi-node GRPO training with dedicated_vllm topology where
one node runs vLLM and the other runs the training workload.

Prerequisites:
  - kubectl configured with access to the HyperPod EKS cluster
  - Training container image pushed to ECR
  - Kubeflow training operator installed (``pytorchjobs.kubeflow.org`` CRD)
  - FSx for Lustre or other storage configured for data/checkpoints

Module layout
-------------

This package is split into three submodules for size reasons:

  * ``settings`` — the ``HyperPodEksSettings`` dataclass.
  * ``manifests`` — ``ManifestsBackendMixin`` with all the
    ``_build_*`` / ``_get_*`` methods that turn a config into
    Kubernetes resources.
  * ``__init__`` (this file) — ``AwsHyperPodEksBackend`` itself plus
    its dependencies (validate, launch, SSM tunnel, kubectl wrapper,
    settings parsing, source upload, apply/wait/cleanup) and the
    ``register_backend`` call.

Public surface is preserved: ``from foresightrlcore.execution.aws_hyperpod_eks
import AwsHyperPodEksBackend, HyperPodEksSettings`` still works because
both names are re-exported below.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from foresightrlcore.config.schema import Topology, VllmMode
from foresightrlcore.execution import register_backend
from foresightrlcore.execution.sagemaker_common import (
    STAGE_IGNORE,
    STAGE_IGNORE_PATTERNS,
    stage_extra_packages,
)

from .manifests import ManifestsBackendMixin
from .settings import HyperPodEksSettings

PACKAGE_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PACKAGE_ROOT / "runtime" / "scripts"

logger = logging.getLogger(__name__)


__all__ = ["AwsHyperPodEksBackend", "HyperPodEksSettings"]


class AwsHyperPodEksBackend(ManifestsBackendMixin):
    """Execution backend for AWS SageMaker HyperPod with EKS orchestration."""

    backend_type = "aws_hyperpod_eks"

    def __init__(self) -> None:
        # SSM port-forward state. Initialized here (not via class-level
        # defaults) so ``_teardown_ssm_tunnel`` can be safely called on a
        # fresh instance before ``_ensure_ssm_tunnel`` has run — e.g. from
        # a test that exercises teardown in isolation, or from the
        # ``launch`` ``finally`` block if ``_launch_inner`` raises before
        # the tunnel is opened.
        self._ssm_proc: Optional[subprocess.Popen] = None
        self._ssm_tunnel_started_by_us: bool = False

    def validate(self, config: Any) -> list[str]:
        """Validate HyperPod EKS-specific config."""
        errors: list[str] = []

        try:
            import boto3  # noqa: F401
        except ImportError:
            errors.append(
                "boto3 is not installed. Install with: uv pip install -e '.[aws]'"
            )
            return errors

        eks_cfg = config.execution.hyperpod_eks

        if not eks_cfg or not getattr(eks_cfg, "image_uri", None):
            errors.append(
                "HyperPod EKS backend requires execution.hyperpod_eks.image_uri "
                "(ECR image with training code)"
            )

        if not self._kubectl_available():
            errors.append(
                "kubectl is not available or not configured. "
                "Ensure kubectl is installed and kubeconfig is set."
            )

        topology = config.infra.topology
        vllm_mode = config.infra.vllm_mode
        instance_count = config.infra.instance_count

        if topology == Topology.DEDICATED_VLLM and instance_count < 2:
            errors.append(
                f"dedicated_vllm topology requires instance_count >= 2, "
                f"got {instance_count}"
            )

        if vllm_mode == VllmMode.SERVER and topology not in (
            Topology.SINGLE_NODE,
            Topology.DEDICATED_VLLM,
            Topology.MULTI_NODE_SERVER,
        ):
            errors.append(
                f"vllm_mode=server with topology={topology.value} is not supported. "
                f"Use dedicated_vllm or single_node topology."
            )

        return errors

    def launch(self, config: Any, wait: bool = False) -> str:
        """Submit a training job to the HyperPod EKS cluster."""
        settings = self._parse_settings(config)

        self._ensure_ssm_tunnel(settings)
        try:
            return self._launch_inner(config, settings, wait=wait)
        finally:
            if not wait:
                # Tear down the tunnel once the job is submitted.
                # When wait=True we keep it open for status polling, then tear down.
                self._teardown_ssm_tunnel()

    def _launch_inner(
        self, config: Any, settings: HyperPodEksSettings, wait: bool = False
    ) -> str:
        """Internal launch logic (called after tunnel is established)."""
        # Upload source code and config to S3
        source_s3_uri = self._upload_source_to_s3(config)
        logger.info(f"Source code uploaded: {source_s3_uri}")

        manifests = self._build_manifest(config, settings)

        # Handle both single manifest and list of manifests
        if isinstance(manifests, list):
            # dedicated_vllm topology: multiple resources
            job_name = None
            for m in manifests:
                if m["kind"] == "PyTorchJob":
                    job_name = m["metadata"]["name"]
                logger.info(f"Applying {m['kind']}: {m['metadata']['name']}")
                self._apply_manifest(m, settings=settings)
            if job_name is None:
                job_name = manifests[0]["metadata"]["name"]
        else:
            job_name = manifests["metadata"]["name"]
            logger.info(f"Submitting job to HyperPod EKS: {job_name}")
            logger.info(f"Namespace: {settings.namespace}")
            logger.info("CRD type: PyTorchJob")
            self._apply_manifest(manifests, settings=settings)

        logger.info(f"Topology: {config.infra.topology.value}")
        logger.info(f"vLLM mode: {config.infra.vllm_mode.value}")

        is_dedicated_vllm = config.infra.topology == Topology.DEDICATED_VLLM
        if wait:
            try:
                self._wait_for_job(
                    job_name,
                    settings.namespace,
                    settings,
                    cleanup_vllm=is_dedicated_vllm,
                )
            finally:
                self._teardown_ssm_tunnel()
        elif is_dedicated_vllm:
            logger.warning(
                f"Job {job_name} submitted without wait=True. "
                f"The vLLM Deployment '{job_name}-vllm' will keep running after training "
                f"completes and must be deleted manually:\n"
                f"  kubectl delete deployment,service {job_name}-vllm -n {settings.namespace}"
            )

        return job_name

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a HyperPod EKS training job."""
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pytorchjob",
                    job_id,
                    "-o",
                    "jsonpath={.status.conditions[-1].type}",
                ],
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip() if result.returncode == 0 else "Unknown"
            return {"job_id": job_id, "status": status}
        except Exception as e:
            return {"job_id": job_id, "status": "Unknown", "error": str(e)}

    # ------------------------------------------------------------------
    # SSM tunnel management
    #
    # State (``_ssm_proc``, ``_ssm_tunnel_started_by_us``) is initialised in
    # ``__init__`` above — see the comment there for why.
    # ------------------------------------------------------------------

    def _is_tunnel_up(self, port: int = 6443, timeout: float = 1.0) -> bool:
        """Return True if localhost:<port> is reachable (tunnel already open)."""
        try:
            with socket.create_connection(("localhost", port), timeout=timeout):
                return True
        except OSError:
            return False

    def _ensure_ssm_tunnel(self, settings: HyperPodEksSettings) -> None:
        """Open SSM port-forward tunnel if needed and not already running.

        When ssm_relay_instance_id is configured the backend uses AWS SSM
        Session Manager to forward localhost:6443 → EKS API endpoint:443
        through a VPC relay instance.          The relay must be an EC2 instance inside the EKS VPC with the SSM agent
        running and AmazonSSMManagedInstanceCore permissions — typically the
        dedicated bastion host.

        The tunnel is a no-op when:
        - ssm_relay_instance_id is not set in the config, or
        - localhost:6443 is already reachable (someone else opened it).
        """
        if not settings.ssm_relay_instance_id:
            return

        if self._is_tunnel_up():
            logger.info("SSM tunnel already up on localhost:6443, reusing.")
            return

        # Derive EKS API hostname from the kubeconfig or the cluster name
        eks_endpoint = self._get_eks_endpoint(settings)

        cmd = [
            "aws",
            "ssm",
            "start-session",
            "--target",
            settings.ssm_relay_instance_id,
            "--region",
            "us-east-1",
            "--document-name",
            "AWS-StartPortForwardingSessionToRemoteHost",
            "--parameters",
            (
                f'{{"host":["{eks_endpoint}"],'
                f'"portNumber":["443"],'
                f'"localPortNumber":["6443"]}}'
            ),
        ]

        env = os.environ.copy()
        # Inherit AWS_PROFILE from kubeconfig exec block if set
        if settings.kubeconfig_path:
            env["KUBECONFIG"] = settings.kubeconfig_path

        logger.info(
            f"Starting SSM tunnel: localhost:6443 → {eks_endpoint}:443 "
            f"via {settings.ssm_relay_instance_id}"
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._ssm_proc = proc
        self._ssm_tunnel_started_by_us = True

        # Wait up to 15 s for the port to open
        deadline = time.time() + 15
        while time.time() < deadline:
            if self._is_tunnel_up():
                logger.info("SSM tunnel is up.")
                return
            time.sleep(1)

        proc.terminate()
        raise RuntimeError(
            "SSM port-forward tunnel did not come up within 15 seconds. "
            "Check that the relay instance is running and the SSM agent is active."
        )

    def _teardown_ssm_tunnel(self) -> None:
        """Terminate the SSM tunnel process if we started it."""
        if self._ssm_proc is not None and self._ssm_tunnel_started_by_us:
            logger.info("Tearing down SSM tunnel.")
            self._ssm_proc.terminate()
            try:
                self._ssm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ssm_proc.kill()
            self._ssm_proc = None
            self._ssm_tunnel_started_by_us = False

    def _get_eks_endpoint(self, settings: HyperPodEksSettings) -> str:
        """Derive the EKS API server hostname for the SSM tunnel target.

        Tries (in order):
        1. Parse the server URL from the kubeconfig (most reliable)
        2. Call aws eks describe-cluster
        3. Fall back to a deterministic hostname pattern from the OIDC ID
        """
        # 1. Parse kubeconfig
        if settings.kubeconfig_path:
            kube_path = Path(settings.kubeconfig_path).expanduser()
            if kube_path.exists():
                try:
                    with open(kube_path) as f:
                        kube_cfg = yaml.safe_load(f)
                    for cluster in kube_cfg.get("clusters", []):
                        server = cluster.get("cluster", {}).get("server", "")
                        # May be https://localhost:6443 (SSM) — skip those
                        tls_name = cluster.get("cluster", {}).get("tls-server-name", "")
                        if tls_name and "localhost" not in tls_name:
                            return tls_name
                        if server and "localhost" not in server:
                            from urllib.parse import urlparse

                            return urlparse(server).hostname or server
                except Exception:
                    pass

        # 2. aws eks describe-cluster
        if settings.eks_cluster_name:
            try:
                import boto3

                eks = boto3.client("eks", region_name="us-east-1")
                resp = eks.describe_cluster(name=settings.eks_cluster_name)
                endpoint = resp["cluster"]["endpoint"]
                from urllib.parse import urlparse

                return urlparse(endpoint).hostname or endpoint
            except Exception as e:
                logger.warning(f"Could not describe EKS cluster: {e}")

        raise RuntimeError(
            "Cannot determine EKS API endpoint for SSM tunnel. "
            "Set kubeconfig_path with a valid kubeconfig or ensure "
            "eks_cluster_name is set and AWS creds can describe the cluster."
        )

    # ------------------------------------------------------------------
    # kubectl wrapper
    # ------------------------------------------------------------------

    def _run_kubectl(
        self,
        args: List[str],
        settings: Optional[HyperPodEksSettings] = None,
        check: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run kubectl with the correct KUBECONFIG injected."""
        env = os.environ.copy()
        if settings and settings.kubeconfig_path:
            env["KUBECONFIG"] = str(Path(settings.kubeconfig_path).expanduser())
        return subprocess.run(
            ["kubectl"] + args,
            capture_output=capture_output,
            text=True,
            check=check,
            env=env,
        )

    def _kubectl_available(self) -> bool:
        """Check if kubectl is available and configured."""
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client", "--output=json"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _parse_settings(self, config: Any) -> HyperPodEksSettings:
        """Parse HyperPod EKS settings from config."""
        eks_cfg = config.execution.hyperpod_eks
        if eks_cfg is None:
            return HyperPodEksSettings()
        ssm_relay = getattr(eks_cfg, "ssm_relay_instance_id", None)
        kube_path = getattr(eks_cfg, "kubeconfig_path", None)
        # Resolve ~ and make relative paths absolute against cwd so kubectl
        # invocations from any working directory pick up the same file.
        # No implicit default: callers using SSM-tunneled clusters must point
        # this at a real kubeconfig (e.g. intuit/examples/kubeconfigs/<cluster>.yaml).
        if kube_path:
            kube_path = str(Path(kube_path).expanduser().resolve())

        return HyperPodEksSettings(
            cluster_name=getattr(eks_cfg, "cluster_name", "") or "",
            eks_cluster_name=getattr(eks_cfg, "eks_cluster_name", "") or "",
            namespace=getattr(eks_cfg, "namespace", "kubeflow") or "kubeflow",
            image_uri=getattr(eks_cfg, "image_uri", "") or "",
            service_account=getattr(eks_cfg, "service_account", "default") or "default",
            fsx_filesystem_id=getattr(eks_cfg, "fsx_filesystem_id", "") or "",
            fsx_mount_path=getattr(eks_cfg, "fsx_mount_path", "/fsx") or "/fsx",
            storage_class=getattr(eks_cfg, "storage_class", "fsx-sc") or "fsx-sc",
            pvc_name=getattr(eks_cfg, "pvc_name", "") or "",
            node_selector=getattr(eks_cfg, "node_selector", {}) or {},
            tolerations=getattr(eks_cfg, "tolerations", []) or [],
            restart_policy=getattr(eks_cfg, "restart_policy", "OnFailure")
            or "OnFailure",
            max_restarts=getattr(eks_cfg, "max_restarts", 3) or 3,
            ssm_relay_instance_id=ssm_relay,
            kubeconfig_path=kube_path,
        )

    def _get_source_s3_uri(self, config: Any) -> str:
        """Get S3 URI for source code tarball.

        Raises a clear error if ``output.s3_path`` is not an ``s3://`` URI.
        Filesystem-style paths (e.g. ``/fsx/output/...``) used to silently
        parse to ``bucket=""``, which then crashed inside ``boto3.upload_file``
        with an opaque error. The validator now catches this at submit time;
        this method's check is a belt-and-braces defense for callers that
        bypass ``rl-train validate``.
        """
        bucket, prefix = self._parse_output_s3_path(config)
        if prefix:
            return f"s3://{bucket}/{prefix}/hyperpod-code/sourcedir.tar.gz"
        return f"s3://{bucket}/hyperpod-code/sourcedir.tar.gz"

    @staticmethod
    def _write_resolved_requirements(config: Any, source_dir: str) -> None:
        """Overwrite ``requirements.txt`` in the source bundle.

        Resolution priority (highest wins):

          1. Inline ``container.requirements`` — explicit user list.
          2. Recipe ``resolved_requirements`` — set by the schema validator
             when ``container.recipe`` is configured.
          3. HyperPod EKS default — ``qwen3`` recipe. Differs from the
             SageMaker launcher (which leaves the static file from
             ``runtime/scripts/`` alone when no recipe is set) because that
             static file targets the qwen3_5 stack (``transformers>=5.6``,
             ``trl>=1.3``). That combination crashes on import for our
             trl-0.28.x line (see comment in ``qwen3.txt``). Defaulting
             here keeps existing HyperPod configs that predate the recipe
             system working without surprise crashes.
        """
        container = getattr(config, "container", None)
        if container is None:
            return

        reqs: Optional[List[str]]
        if container.requirements is not None:
            reqs = container.requirements
            source = "inline container.requirements"
        elif container.resolved_requirements is not None:
            reqs = container.resolved_requirements
            source = f"recipe {container.recipe!r}"
        else:
            from foresightrlcore.config.recipes import CONTAINER_RECIPES

            reqs = CONTAINER_RECIPES["qwen3"]["requirements"]
            source = "default 'qwen3' (HyperPod EKS fallback)"
            logger.warning(
                "container.recipe is unset; defaulting to 'qwen3' for "
                "HyperPod EKS. Set container.recipe explicitly to silence "
                "this warning."
            )
        reqs = container.with_extras(reqs) or []

        req_path = os.path.join(source_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("\n".join(reqs) + "\n")
        logger.info("Wrote %d requirements to bundle from %s", len(reqs), source)

    @staticmethod
    def _parse_output_s3_path(config: Any) -> tuple[str, str]:
        """Split ``output.s3_path`` into ``(bucket, prefix)`` or raise."""
        s3_path = str(config.output.s3_path)
        if not s3_path.startswith("s3://"):
            raise ValueError(
                f"HyperPod EKS backend requires output.s3_path to be an "
                f"s3:// URI (source tarballs are uploaded there and workers "
                f"fetch them with `aws s3 cp`). Got: {s3_path!r}. If you "
                f"meant to use FSx for model output, that's handled "
                f"separately via FSx mounts in the pod spec — the source "
                f"tarball path must still be S3."
            )
        parts = s3_path[len("s3://") :].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        if not bucket:
            raise ValueError(f"output.s3_path is missing a bucket: {s3_path!r}")
        return bucket, prefix.rstrip("/")

    def _upload_source_to_s3(self, config: Any) -> str:
        """Package scripts + foresightrlcore package + config and upload to S3.

        Follows the same pattern as this repo's aws_sagemaker_launcher.py:
        - Copy SCRIPTS_DIR (contains train.py, requirements.txt, etc.)
        - Copy foresightrlcore package (train.py imports from it)
        - Add args.yaml with the job config
        - Tar and upload to S3

        Note: idps-sdk is pre-installed in the training Docker image
        (intuit/docker/Dockerfile.qwen3 → intuit/docker/Dockerfile.base), so
        no runtime install is needed. IDPS resolution imports
        foresightrl_intuit, which container.extra_packages stages.

        The packaged ``requirements.txt`` is overwritten with the resolved
        recipe requirements when ``container.recipe`` is set (mirrors the
        SageMaker launcher's behaviour). When no recipe is set we default
        to the ``qwen3`` recipe rather than the static fallback in
        ``runtime/scripts/requirements.txt`` (which targets the qwen3_5
        stack and is incompatible with the trl 0.28.x line — see the
        comment block in ``foresightrlcore/config/requirements/qwen3.txt``).
        """
        bucket, prefix = self._parse_output_s3_path(config)
        s3_key = (
            f"{prefix}/hyperpod-code/sourcedir.tar.gz"
            if prefix
            else "hyperpod-code/sourcedir.tar.gz"
        )

        # Create temp directory with source code
        tmp_dir = tempfile.mkdtemp(prefix="hyperpod_src_")
        try:
            # Copy runtime scripts
            shutil.copytree(
                str(SCRIPTS_DIR), tmp_dir, ignore=STAGE_IGNORE, dirs_exist_ok=True
            )

            # Copy foresightrlcore package (train.py imports from it)
            dest_pkg = os.path.join(tmp_dir, "foresightrlcore")
            shutil.copytree(
                str(PACKAGE_ROOT),
                dest_pkg,
                ignore=shutil.ignore_patterns(*STAGE_IGNORE_PATTERNS, "runtime"),
            )

            stage_extra_packages(tmp_dir, config.container.extra_packages)

            self._write_resolved_requirements(config, tmp_dir)

            # Write config as args.yaml
            # Use mode='json' to serialize enums as strings (safe_load compatible)
            config_dict = (
                config.model_dump(mode="json") if hasattr(config, "model_dump") else {}
            )
            with open(os.path.join(tmp_dir, "args.yaml"), "w") as f:
                yaml.dump(config_dict, f)

            # Create tarball
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                tar_path = f.name

            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(tmp_dir, arcname=".")

            # Upload to S3
            import boto3

            s3 = boto3.client("s3")
            s3.upload_file(tar_path, bucket, s3_key)
            os.unlink(tar_path)

            return f"s3://{bucket}/{s3_key}"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _apply_manifest(
        self, manifest: dict, settings: Optional[HyperPodEksSettings] = None
    ) -> None:
        """Apply Kubernetes manifest via kubectl."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest, f, default_flow_style=False)
            manifest_path = f.name

        try:
            logger.info(f"Applying manifest: {manifest_path}")
            result = self._run_kubectl(
                ["apply", "-f", manifest_path], settings=settings
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"kubectl apply failed: {result.stderr}\n{result.stdout}"
                )
            logger.info(f"Manifest applied successfully: {result.stdout.strip()}")
        finally:
            os.unlink(manifest_path)

    def _cleanup_vllm_deployment(
        self,
        job_name: str,
        namespace: str,
        settings: Optional[HyperPodEksSettings] = None,
    ) -> None:
        """Delete the vLLM Deployment and Service created for dedicated_vllm topology.

        Unlike a PyTorchJob, a Deployment does not self-terminate when training
        finishes. We clean it up by label so the GPU node is released.
        """
        vllm_name = f"{job_name}-vllm"
        for kind in ("deployment", "service"):
            result = self._run_kubectl(
                ["-n", namespace, "delete", kind, vllm_name, "--ignore-not-found"],
                settings=settings,
            )
            if result.returncode == 0:
                logger.info(f"Deleted {kind}/{vllm_name}")
            else:
                logger.warning(
                    f"Could not delete {kind}/{vllm_name}: {result.stderr.strip()}"
                )

    def _wait_for_job(
        self,
        job_name: str,
        namespace: str,
        settings: Optional[HyperPodEksSettings] = None,
        cleanup_vllm: bool = False,
    ) -> None:
        """Wait for job to complete, then optionally clean up vLLM Deployment."""
        crd_type = "pytorchjob"
        logger.info(f"Waiting for {crd_type}/{job_name} to complete...")

        while True:
            try:
                result = self._run_kubectl(
                    [
                        "-n",
                        namespace,
                        "get",
                        crd_type,
                        job_name,
                        "-o",
                        "jsonpath={.status.conditions[-1].type}",
                    ],
                    settings=settings,
                )
                status = result.stdout.strip()
                logger.info(f"Job status: {status}")

                if status in ("Succeeded", "Failed"):
                    if cleanup_vllm:
                        self._cleanup_vllm_deployment(job_name, namespace, settings)
                    if status == "Failed":
                        raise RuntimeError(f"Training job failed: {job_name}")
                    logger.info(f"Training job completed: {job_name}")
                    return

            except subprocess.CalledProcessError as e:
                logger.warning(f"Error checking job status: {e}")

            time.sleep(30)


register_backend("aws_hyperpod_eks", AwsHyperPodEksBackend)
