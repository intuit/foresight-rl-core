"""SageMaker launcher for the ``perf_test`` operation.

Submits a single ``CreateTrainingJob`` request via boto3. Compared to
:class:`GRPOJobLauncher` this is much simpler -- there is no SDK v2/v3
split, no multi-node topology, no vLLM-vs-torchrun branching. The
container resolves to the pre-baked DJL inference image plus a Gradle
install at startup; the entry point is ``perf_entrypoint.sh``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import boto3

from foresightrlcore.execution.sagemaker_common import (
    STAGE_IGNORE,
    STAGE_IGNORE_PATTERNS,
    apply_inter_container_encryption,
    extra_packages_env,
    stage_extra_packages,
)

if TYPE_CHECKING:
    from foresightrlcore.config.perf_schema import PerfTestJobConfig

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "runtime" / "scripts"
PACKAGE_DIR = Path(__file__).resolve().parent.parent  # foresightrlcore/


class PerfTestSageMakerLauncher:
    """Submit a SageMaker training job for a perf-test run."""

    def __init__(
        self,
        config: "PerfTestJobConfig",
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
        """Submit the perf-test job and return the job name."""
        source_s3_uri = self._upload_source_dir()
        config_s3_uri = self._upload_config()
        run = self.config.perf_test

        input_channels: Dict[str, str] = {
            "config": _parent_prefix(config_s3_uri),
            "perf_data": run.gatling.data_s3_path,
        }
        model_input = self._build_model_input()
        if model_input is not None:
            input_channels["model"] = model_input

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
            if self.config.perf_test.role:
                self._role = self.config.perf_test.role
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
        """Staging directory that gets uploaded as ``sourcedir.tar.gz``."""
        if self._prepared_source_dir is None:
            tmp = tempfile.mkdtemp(prefix="foresight_perf_src_")
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
        """Write ``requirements.txt`` from the perf-test recipe variant."""
        c = self.config.container
        reqs: Optional[List[str]]
        if c.requirements is not None:
            reqs = c.requirements
        elif c.resolved_requirements_perf_test is not None:
            reqs = c.resolved_requirements_perf_test
        else:
            reqs = []
        reqs = c.with_extras(reqs) or []
        req_path = os.path.join(source_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write("\n".join(reqs) + "\n")
        logger.info("Wrote %d perf-test requirements to %s", len(reqs), req_path)

    def _upload_source_dir(self) -> str:
        output_bucket, output_prefix = self._parse_s3_uri(
            str(self.config.output.s3_path)
        )
        s3_key = "/".join(
            s for s in [output_prefix.rstrip("/"), "source", "sourcedir.tar.gz"] if s
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tmp_path = f.name

        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(self._source_dir, arcname=".")
            return self._upload_file_to_s3(tmp_path, output_bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    def _upload_config(self) -> str:
        output_bucket, output_prefix = self._parse_s3_uri(
            str(self.config.output.s3_path)
        )
        s3_key = "/".join(
            s for s in [output_prefix.rstrip("/"), "config", "args.yaml"] if s
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            self.config.to_yaml(f.name)
            tmp_path = f.name

        try:
            return self._upload_file_to_s3(tmp_path, output_bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Image + container env
    # ------------------------------------------------------------------

    def _resolve_image_uri(self) -> str:
        """Pick the perf-test image first.

        When a recipe is set, ContainerConfig.image_uri may resolve to the
        training DLC (e.g. pytorch-training); that is wrong for perf tests
        because it doesn't include Java. Prefer the explicit perf-test
        image. Only fall back to ``image_uri`` when the user has set it
        without a recipe (custom image path).
        """
        c = self.config.container
        if c.perf_test_image_uri:
            return c.perf_test_image_uri
        if c.image_uri and c.recipe is None:
            return c.image_uri
        if c.image_uri:
            # Recipe set but no perf-test image -- treat as explicit override.
            return c.image_uri
        raise ValueError(
            "container.image_uri or container.perf_test_image_uri must be "
            "set for perf_test (use a recipe that declares one, e.g. "
            "container.recipe: qwen3_5)."
        )

    def _build_model_input(self) -> Optional[str]:
        """Return an S3 URI to mount as a 'model' input channel, or None."""
        name = self.config.model.name
        if name.startswith("s3://"):
            return name
        return None

    def _build_environment(self, source_s3_uri: str) -> Dict[str, str]:
        run = self.config.perf_test
        env = {**self.config.environment}
        env.update(extra_packages_env(self.config.container.extra_packages))
        env["PERF_TEST_CONFIG"] = "/opt/ml/input/data/config/args.yaml"
        env["PERF_TEST_DATA_CHANNEL"] = "/opt/ml/input/data/perf_data"
        env["PERF_TEST_OUTPUT_DIR"] = "/opt/ml/output/data"
        env["PERF_TEST_MODEL_PATH"] = (
            "/opt/ml/input/data/model"
            if self._build_model_input() is not None
            else self.config.model.name
        )
        # Source tarball location. The DJL inference image has its own baked-in
        # entrypoint, so we override AlgorithmSpecification.ContainerEntrypoint
        # to a tiny bootstrap that downloads and extracts the sourcedir before
        # exec'ing perf_entrypoint.sh. The bootstrap reads this env var.
        env["PERF_TEST_SOURCE_S3"] = source_s3_uri
        if run.serving.vllm_env:
            env["PERF_TEST_VLLM_ENV_JSON"] = json.dumps(run.serving.vllm_env)
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
        run = self.config.perf_test
        region = self.session.boto_session.region_name
        sm_client = boto3.client("sagemaker", region_name=region)

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name = self._build_job_name(timestamp)

        image_uri = self._resolve_image_uri()
        environment = self._build_environment(source_s3_uri)
        environment["SAGEMAKER_REGION"] = region

        input_data_config = []
        for channel_name, s3_uri in input_channels.items():
            input_data_config.append(
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
            )

        # The DJL inference image has a baked-in ENTRYPOINT (partition.py)
        # that ignores the SageMaker DLC `sagemaker_program` hyperparameter
        # contract. Override with an explicit BYOC ContainerEntrypoint that
        # pulls the sourcedir tarball from S3, extracts it, and exec's
        # perf_entrypoint.sh. SageMaker caps ContainerArguments at 256 chars
        # per element, so the bootstrap is intentionally terse. DJL inference
        # images ship the AWS CLI.
        bootstrap = (
            "set -e;"
            "D=/opt/foresight-perf-src;mkdir -p $D;"
            'aws s3 cp "$PERF_TEST_SOURCE_S3" /tmp/s.tgz;'
            "tar -xzf /tmp/s.tgz -C $D;rm /tmp/s.tgz;"
            "chmod +x $D/perf_entrypoint.sh;"
            "export PYTHONPATH=$D:${PYTHONPATH:-};"
            "exec $D/perf_entrypoint.sh"
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
                "InstanceType": run.instance_type,
                "InstanceCount": run.instance_count,
                "VolumeSizeInGB": run.volume_size_gb,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.config.job.max_runtime_hours * 3600,
            },
            "Environment": environment,
        }

        # Run against a reserved capacity block (e.g. p5en training plan) when set,
        # mirroring the training launcher. Injected into ResourceConfig so the job
        # does not fall back to on-demand.
        if run.training_plan_arn:
            request["ResourceConfig"]["TrainingPlanArn"] = run.training_plan_arn
            logger.info("Training plan: %s", run.training_plan_arn)

        # ACP IS031.1: encrypt inter-container traffic for training jobs.
        apply_inter_container_encryption(request, run.encrypt_inter_container_traffic)

        if run.vpc:
            request["VpcConfig"] = {
                "Subnets": run.vpc.subnets,
                "SecurityGroupIds": run.vpc.security_groups,
            }
            logger.info(
                "VPC: subnets=%s security_groups=%s",
                run.vpc.subnets,
                run.vpc.security_groups,
            )

        logger.info("Submitting perf-test job: %s", job_name)
        response = sm_client.create_training_job(**request)
        logger.info(
            "Perf-test job submitted: %s (ARN: %s)",
            job_name,
            response.get("TrainingJobArn"),
        )

        if wait:
            logger.info("Waiting for perf-test job to complete...")
            waiter = sm_client.get_waiter("training_job_completed_or_stopped")
            waiter.wait(TrainingJobName=job_name)

        return job_name

    def _build_job_name(self, timestamp: str) -> str:
        run = self.config.perf_test
        base = f"perf-{self.config.job.name}-{run.name}".replace("_", "-")
        full = f"{base}-{timestamp}"
        # SageMaker limits training-job names to 63 characters.
        if len(full) <= 63:
            return full
        return full[-63:]

    def _run_output_s3_path(self) -> str:
        run = self.config.perf_test
        base = str(self.config.output.s3_path or "").rstrip("/")
        return f"{base}/{run.name}/"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

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


__all__ = ["PerfTestSageMakerLauncher"]
