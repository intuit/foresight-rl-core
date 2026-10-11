"""SageMaker launcher for the ``evaluate`` operation.

Submits a single ``CreateTrainingJob`` via boto3, mirroring
:class:`OptimizerSageMakerLauncher`. The evaluate-specific parts:

* **Infra sizing** comes from the candidate models: API-only runs (bedrock /
  genos / echo) take a CPU instance and the lightweight image; one
  ``provider: local`` candidate forces a GPU instance and the vLLM-capable
  image, because that endpoint is a real ``vllm serve``.
* Mounts a ``model`` channel when a locally-served candidate points at S3, so
  weights are on local disk before the endpoint starts, plus an ``adapter``
  channel for a candidate that serves a LoRA adapter over those weights.
* Streams both resume checkpoints through a SageMaker ``CheckpointConfig``
  when ``output.checkpoint_s3_path`` is set.
* Entry point is ``evaluate_entrypoint.sh``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import boto3

from foresightrlcore.config.evaluate_schema import (
    local_adapter_channels,
    local_model_channels,
)
from foresightrlcore.execution.sagemaker_common import (
    STAGE_IGNORE,
    STAGE_IGNORE_PATTERNS,
    apply_inter_container_encryption,
    extra_packages_env,
    stage_extra_packages,
)

if TYPE_CHECKING:
    from foresightrlcore.config.evaluate_schema import EvaluateJobConfig

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "runtime" / "scripts"
PACKAGE_DIR = Path(__file__).resolve().parent.parent  # foresightrlcore/

_CONTAINER_CHECKPOINT_DIR = "/opt/ml/checkpoints"
_CONTAINER_CONFIG_PATH = "/opt/ml/input/data/config/args.yaml"


class EvaluateSageMakerLauncher:
    """Submit a SageMaker training job for an evaluate run."""

    def __init__(
        self,
        config: "EvaluateJobConfig",
        *,
        role: Optional[str] = None,
        session: Any = None,
    ) -> None:
        self.config = config
        self._session = session
        self._role = role
        self._s3_client: Optional[Any] = None
        self._prepared_source_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def launch(self, wait: bool = False) -> str:
        source_s3_uri = self._upload_source_dir()
        config_s3_uri = self._upload_config()

        input_channels: Dict[str, str] = {"config": _parent_prefix(config_s3_uri)}
        input_channels.update(self._build_model_inputs())

        return self._submit_run(
            source_s3_uri=source_s3_uri,
            input_channels=input_channels,
            wait=wait,
        )

    # ------------------------------------------------------------------
    # Session + role + clients
    # ------------------------------------------------------------------

    @property
    def session(self) -> Any:
        if self._session is None:
            try:
                from sagemaker.core.helper.session_helper import Session

                self._session = Session()
            except (ImportError, ModuleNotFoundError):
                from sagemaker import Session

                self._session = Session()
        return self._session

    @property
    def role(self) -> str:
        if self._role is None:
            if self.config.evaluate.role:
                self._role = self.config.evaluate.role
                return self._role

            try:
                from sagemaker.core.helper.session_helper import get_execution_role
            except (ImportError, ModuleNotFoundError):
                from sagemaker import get_execution_role

            try:
                self._role = get_execution_role()
            except ValueError:
                iam = boto3.client("iam")
                self._role = iam.get_role(RoleName="sagemaker_execution_role")["Role"][
                    "Arn"
                ]
        return self._role

    @property
    def s3_client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    # ------------------------------------------------------------------
    # Source-code staging
    # ------------------------------------------------------------------

    @property
    def _source_dir(self) -> str:
        if self._prepared_source_dir is None:
            tmp = tempfile.mkdtemp(prefix="foresight_evaluate_src_")
            shutil.copytree(
                str(SCRIPTS_DIR), tmp, ignore=STAGE_IGNORE, dirs_exist_ok=True
            )
            dest = os.path.join(tmp, "foresightrlcore")
            shutil.copytree(
                str(PACKAGE_DIR),
                dest,
                ignore=shutil.ignore_patterns(*STAGE_IGNORE_PATTERNS, "runtime"),
            )
            stage_extra_packages(tmp, self.config.container.extra_packages)
            self._write_resolved_requirements(tmp)
            self._prepared_source_dir = tmp
        return self._prepared_source_dir

    def _write_resolved_requirements(self, source_dir: str) -> None:
        """Write ``requirements.txt`` from the evaluate recipe variant."""
        container = self.config.container
        if container.requirements is not None:
            reqs: List[str] = container.requirements
        elif container.resolved_requirements_evaluate is not None:
            reqs = container.resolved_requirements_evaluate
        else:
            reqs = []
        reqs = container.with_extras(reqs) or []
        req_path = os.path.join(source_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("\n".join(reqs) + "\n")
        logger.info("Wrote %d evaluate requirements to %s", len(reqs), req_path)

    def _upload_source_dir(self) -> str:
        bucket, prefix = self._parse_s3_uri(str(self.config.output.s3_path))
        s3_key = "/".join(
            s for s in [prefix.rstrip("/"), "source", "sourcedir.tar.gz"] if s
        )
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tmp_path = f.name
        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(self._source_dir, arcname=".")
            return self._upload_file_to_s3(tmp_path, bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    def _upload_config(self) -> str:
        bucket, prefix = self._parse_s3_uri(str(self.config.output.s3_path))
        s3_key = "/".join(s for s in [prefix.rstrip("/"), "config", "args.yaml"] if s)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            self.config.to_yaml(f.name)
            tmp_path = f.name
        try:
            return self._upload_file_to_s3(tmp_path, bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Image + inputs + env
    # ------------------------------------------------------------------

    def _resolve_image_uri(self) -> str:
        """CPU image for API-only runs; a vLLM (GPU) image when serving locally."""
        container = self.config.container
        if self.config.requires_gpu():
            if container.perf_test_image_uri:
                return container.perf_test_image_uri
            if container.image_uri:
                return container.image_uri
            raise ValueError(
                "local serving requires a vLLM-capable GPU image: set "
                "container.recipe (e.g. qwen3_5), container.perf_test_image_uri, "
                "or container.image_uri."
            )
        if container.evaluate_image_uri:
            return container.evaluate_image_uri
        if container.image_uri:
            return container.image_uri
        if container.optimizer_image_uri:
            return container.optimizer_image_uri
        raise ValueError(
            "evaluate requires a container image: set container.recipe, "
            "container.evaluate_image_uri, or container.image_uri."
        )

    def _build_model_inputs(self) -> Dict[str, str]:
        """Return the channels to mount: weights per model, adapters alongside."""
        channels = {
            channel: model.weights_source
            for channel, model in local_model_channels(self.config.evaluate.models)
        }
        channels.update(
            {
                channel: model.adapter or ""
                for channel, model in local_adapter_channels(
                    self.config.evaluate.models
                )
            }
        )
        return channels

    def _build_environment(self, source_s3_uri: str) -> Dict[str, str]:
        env = {**self.config.environment}
        env.update(extra_packages_env(self.config.container.extra_packages))
        env["EVALUATE_CONFIG"] = _CONTAINER_CONFIG_PATH
        env["EVALUATE_OUTPUT_DIR"] = "/opt/ml/output/data"
        env["EVALUATE_CHECKPOINT_DIR"] = _CONTAINER_CHECKPOINT_DIR
        env["EVALUATE_SOURCE_S3"] = source_s3_uri
        return env

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _submit_run(
        self,
        *,
        source_s3_uri: str,
        input_channels: Dict[str, str],
        wait: bool,
    ) -> str:
        region = self.session.boto_session.region_name
        sm_client = boto3.client("sagemaker", region_name=region)

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name = self._build_job_name(timestamp)

        image_uri = self._resolve_image_uri()
        environment = self._build_environment(source_s3_uri)
        environment["SAGEMAKER_REGION"] = region

        input_data_config = [
            {
                "ChannelName": channel_name,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": str(s3_uri),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            }
            for channel_name, s3_uri in input_channels.items()
        ]

        # Bootstrap: pull the sourcedir tarball, extract, exec the entrypoint.
        bootstrap = (
            "set -e;"
            "D=/opt/foresight-evaluate-src;mkdir -p $D;"
            'aws s3 cp "$EVALUATE_SOURCE_S3" /tmp/s.tgz;'
            "tar -xzf /tmp/s.tgz -C $D;rm /tmp/s.tgz;"
            "chmod +x $D/evaluate_entrypoint.sh;"
            "export PYTHONPATH=$D:${PYTHONPATH:-};"
            "exec $D/evaluate_entrypoint.sh"
        )

        request: Dict[str, Any] = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["/bin/bash", "-c"],
                "ContainerArguments": [bootstrap],
            },
            "RoleArn": self.role,
            "InputDataConfig": input_data_config,
            "OutputDataConfig": {
                "S3OutputPath": str(self._run_output_s3_path()),
                "CompressionType": "NONE",
            },
            "ResourceConfig": {
                "InstanceType": self.config.resolved_instance_type(),
                "InstanceCount": self.config.evaluate.instance_count,
                "VolumeSizeInGB": self.config.evaluate.volume_size_gb,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.config.job.max_runtime_hours * 3600,
            },
            "Environment": environment,
        }

        if self.config.output.checkpoint_s3_path:
            request["CheckpointConfig"] = {
                "S3Uri": self.config.output.checkpoint_s3_path,
                "LocalPath": _CONTAINER_CHECKPOINT_DIR,
            }
            logger.info("Checkpoint S3: %s", self.config.output.checkpoint_s3_path)

        if self.config.evaluate.training_plan_arn:
            request["ResourceConfig"][
                "TrainingPlanArn"
            ] = self.config.evaluate.training_plan_arn
            logger.info("Training plan: %s", self.config.evaluate.training_plan_arn)

        if self.config.evaluate.vpc:
            request["VpcConfig"] = {
                "Subnets": self.config.evaluate.vpc.subnets,
                "SecurityGroupIds": self.config.evaluate.vpc.security_groups,
            }
            logger.info(
                "VPC: subnets=%s security_groups=%s",
                self.config.evaluate.vpc.subnets,
                self.config.evaluate.vpc.security_groups,
            )

        # ACP IS031.1: encrypt inter-container traffic for training jobs.
        apply_inter_container_encryption(
            request, self.config.evaluate.encrypt_inter_container_traffic
        )

        logger.info(
            "Submitting evaluate job: %s (instance=%s, image=%s)",
            job_name,
            self.config.resolved_instance_type(),
            image_uri,
        )
        response = sm_client.create_training_job(**request)
        logger.info(
            "Evaluate job submitted: %s (ARN: %s)",
            job_name,
            response.get("TrainingJobArn"),
        )

        if wait:
            logger.info("Waiting for evaluate job to complete...")
            waiter = sm_client.get_waiter("training_job_completed_or_stopped")
            waiter.wait(TrainingJobName=job_name)

        return job_name

    def _build_job_name(self, timestamp: str) -> str:
        base = f"eval-{self.config.job.name}".replace("_", "-")
        full = f"{base}-{timestamp}"
        return full if len(full) <= 63 else full[-63:]

    def _run_output_s3_path(self) -> str:
        return f"{str(self.config.output.s3_path or '').rstrip('/')}/"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s3_uri(uri: str) -> "tuple[str, str]":
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")
        parts = uri[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _upload_file_to_s3(self, local_path: str, bucket: str, s3_key: str) -> str:
        self.s3_client.upload_file(local_path, bucket, s3_key)
        uri = f"s3://{bucket}/{s3_key}"
        logger.info("Uploaded %s -> %s", local_path, uri)
        return uri


def _parent_prefix(s3_uri: str) -> str:
    """``s3://bucket/a/b/c.yaml`` -> ``s3://bucket/a/b/``."""
    if "/" not in s3_uri[5:]:
        return s3_uri
    return s3_uri.rsplit("/", 1)[0] + "/"


__all__ = ["EvaluateSageMakerLauncher"]
