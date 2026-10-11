"""SageMaker job builder and launcher for GRPO training.

Handles single-node (server/colocate vLLM) and multi-node (colocate) topologies.
Compatible with SageMaker Python SDK v2.x (Estimator API) and v3.x (ModelTrainer API).
Falls back to raw boto3 CreateTrainingJob when a training_plan_arn is set and the
installed SDK v3 does not expose training_plan on the Compute class.  The boto3 path
delivers source code via a "code" input channel (same as the SDK Estimator does)
so SageMaker infrastructure downloads it before the container starts.
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
from typing import Any, Dict, Optional

import boto3

from foresightrlcore.config.schema import (
    Topology,
)
from foresightrlcore.config.schema import TrainingJobConfig as SageMakerTrainingConfig
from foresightrlcore.config.schema import (
    VllmMode,
)
from foresightrlcore.execution.sagemaker_common import (
    STAGE_IGNORE,
    STAGE_IGNORE_PATTERNS,
    apply_inter_container_encryption,
    extra_packages_env,
    stage_extra_packages,
)

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "runtime" / "scripts"
PACKAGE_DIR = Path(__file__).resolve().parent.parent  # foresightrlcore/

_SDK_V3 = None


def _is_sdk_v3() -> bool:
    """Detect whether the installed SageMaker SDK is v3 (ModelTrainer API)."""
    global _SDK_V3
    if _SDK_V3 is None:
        try:
            from sagemaker.train.model_trainer import ModelTrainer  # noqa: F401

            _SDK_V3 = True
        except (ImportError, ModuleNotFoundError):
            _SDK_V3 = False
    return _SDK_V3


def _get_session(
    default_bucket: Optional[str] = None,
    default_bucket_prefix: Optional[str] = None,
):
    """Create a SageMaker Session, compatible with SDK v2 and v3.

    When *default_bucket* is provided, the SDK will upload source code,
    config, and outputs there instead of the auto-generated
    ``sagemaker-<region>-<account-id>`` bucket.
    """
    kwargs: Dict[str, Any] = {}
    if default_bucket:
        kwargs["default_bucket"] = default_bucket
    if default_bucket_prefix:
        kwargs["default_bucket_prefix"] = default_bucket_prefix

    if _is_sdk_v3():
        from sagemaker.core.helper.session_helper import Session

        return Session(**kwargs)
    else:
        from sagemaker import Session

        return Session(**kwargs)


def _get_execution_role():
    """Get the SageMaker execution role ARN, compatible with SDK v2 and v3."""
    if _is_sdk_v3():
        from sagemaker.core.helper.session_helper import get_execution_role

        return get_execution_role()
    else:
        from sagemaker import get_execution_role

        return get_execution_role()


def _resolve_image_uri(
    region: str,
    framework_version: str,
    instance_type: str,
    framework: str = "pytorch",
    image_uri: str | None = None,
) -> str:
    """Retrieve a DLC training container image URI.

    When *image_uri* is provided it is returned as-is, bypassing framework
    resolution entirely.  Otherwise the SageMaker SDK resolves the image for
    the requested *framework* (``pytorch`` or ``huggingface``).
    """
    if image_uri:
        return image_uri

    if _is_sdk_v3():
        from sagemaker.core import image_uris
    else:
        from sagemaker import image_uris

    return image_uris.retrieve(
        framework=framework,
        region=region,
        version=framework_version,
        instance_type=instance_type,
        image_scope="training",
    )


class GRPOJobLauncher:
    """Builds and submits SageMaker training jobs for GRPO training."""

    def __init__(
        self,
        config: SageMakerTrainingConfig,
        role: Optional[str] = None,
        session=None,
    ):
        self.config = config
        self._session = session
        self._role = role
        self._s3_client = None

    @property
    def session(self):
        if self._session is None:
            self._session = _get_session(
                default_bucket=self.config.infra.sagemaker_default_bucket,
                default_bucket_prefix=self.config.infra.sagemaker_default_bucket_prefix,
            )
        return self._session

    @property
    def role(self) -> str:
        if self._role is None:
            if self.config.infra.role:
                self._role = self.config.infra.role
            else:
                try:
                    self._role = _get_execution_role()
                except ValueError:
                    iam = boto3.client("iam")
                    self._role = iam.get_role(RoleName="sagemaker_execution_role")[
                        "Role"
                    ]["Arn"]
        return self._role

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    @property
    def _is_torchrun_mode(self) -> bool:
        return self.config.infra.vllm_mode == VllmMode.NONE

    @property
    def _is_merge_lora_only(self) -> bool:
        """True for sft_verl's merge_lora_only mode, which reads a

        checkpoint from output.checkpoint_s3_path and never touches
        training data at all — see VerlSftMethodHandler.run() and
        config/validation.py's merge_lora_only exemptions.
        """
        sft_cfg = getattr(self.config.training, "sft_verl", None)
        return (
            self.config.method.type == "sft_verl"
            and sft_cfg is not None
            and sft_cfg.merge_lora_only
        )

    @property
    def _job_name_prefix(self) -> str:
        """SageMaker job name prefix — reflects the actual method being run.

        Historically hardcoded to "grpo" from when this launcher (despite
        its class name) only ever launched GRPO jobs; now shared by every
        aws_training_job method (sft_verl, future methods, ...), so the
        prefix must reflect config.method.type instead.

        TrainingJobName only allows ``[a-zA-Z0-9-]`` (botocore pattern
        ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}``) — method type strings like
        "sft_verl" use underscores, so they must be hyphenated here. The
        SDK v3 ModelTrainer path happens to do this same substitution
        internally, but the boto3-direct and SDK v2 Estimator paths do not,
        so this can't be left to chance downstream.
        """
        return self.config.method.type.replace("_", "-")

    @property
    def _gpus_per_node(self) -> int:
        # merge_lora_only is single-process, rank-0-only by design (see
        # VerlSftMethodHandler._run_merge_lora_only's docstring) — every
        # other rank just logs "nothing to do" and returns immediately.
        # Launching it via this same torchrun entrypoint must not still
        # spawn a full GPU-count's worth of processes: examples/smoke_tests/
        # sft_verl_merge_smoke.yaml explicitly recommends a cheap CPU-only
        # instance (ml.m5.4xlarge) for this mode, which this heuristic
        # (g5.12/g6e.12 -> 4, else 8) would otherwise treat as an 8-GPU box
        # and configure Torchrun for 8 redundant processes/node — most of
        # which do nothing but still cost real startup overhead, and one of
        # several ranks finishing in lockstep so differently from the
        # others is exactly the kind of skew torchrun's own rendezvous/
        # fault-tolerance logic isn't designed around.
        if self._is_merge_lora_only:
            return 1
        it = self.config.infra.instance_type
        if "g5.12" in it or "g6e.12" in it:
            return 4
        return 8

    @property
    def _requirements_file(self) -> str:
        """requirements_torchrun.txt drops the heavy vLLM dependency."""
        return (
            "requirements_torchrun.txt"
            if self._is_torchrun_mode
            else "requirements.txt"
        )

    @property
    def _source_dir(self) -> str:
        """Temp directory with runtime scripts + the foresightrlcore package.

        The SDK ``SourceCode`` / ``Estimator`` ``source_dir`` parameter
        packages this directory into a ``sourcedir.tar.gz`` and uploads it.
        On the container the contents are extracted to the code directory
        (typically ``/opt/ml/code`` or ``/opt/ml/input/data/code``).

        Because ``train.py`` imports from ``foresightrlcore`` (config,
        methods, hooks, …), the full package must be present alongside the
        runtime scripts.
        """
        if not hasattr(self, "_prepared_source_dir"):
            tmp = tempfile.mkdtemp(prefix="foresight_src_")
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
            self._prepared_source_dir: str = tmp
        return self._prepared_source_dir

    def _job_environment(self) -> Dict[str, str]:
        """Container env: user-supplied vars plus staged-extension discovery."""
        environment = {**self.config.environment}
        environment.update(extra_packages_env(self.config.container.extra_packages))
        return environment

    def _write_resolved_requirements(self, source_dir: str) -> None:
        """Overwrite static requirements files with recipe or inline deps.

        Resolution priority (highest wins):
          1. Inline ``container.requirements`` — explicit user list (may be ``[]``).
          2. Recipe ``resolved_requirements`` — set when a recipe is configured.
          3. Custom image shortcut — when ``image_uri`` is explicitly supplied
             without a recipe the image is assumed to be self-contained.  An
             empty requirements file is written so the container entrypoint
             does not re-install packages on top of the baked-in deps.
             To install extra packages on top, provide an explicit
             ``container.requirements`` list.
          4. Static file fallback — no overwrite; the file from
             ``runtime/scripts/`` is used as-is.
        """
        c = self.config.container
        _custom_image = bool(c.image_uri and c.recipe is None)

        reqs: Optional[list[str]]
        if c.requirements is not None:
            reqs = c.requirements
        elif _custom_image:
            reqs = []
        else:
            reqs = c.resolved_requirements
        reqs = c.with_extras(reqs)

        if reqs is not None:
            req_path = os.path.join(source_dir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("\n".join(reqs) + "\n")
            logger.info("Wrote %d requirements to %s", len(reqs), req_path)

        torchrun_reqs: Optional[list[str]]
        if c.resolved_requirements_torchrun is not None:
            torchrun_reqs = c.resolved_requirements_torchrun
        elif _custom_image:
            torchrun_reqs = []
        else:
            torchrun_reqs = None

        if torchrun_reqs is not None:
            req_path = os.path.join(source_dir, "requirements_torchrun.txt")
            with open(req_path, "w") as f:
                f.write("\n".join(torchrun_reqs) + "\n")
            logger.info(
                "Wrote %d torchrun requirements to %s",
                len(torchrun_reqs),
                req_path,
            )

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        """Parse 's3://bucket/key' into (bucket, key)."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def _upload_file_to_s3(self, local_path: str, bucket: str, s3_key: str) -> str:
        """Upload a local file to S3 and return the s3:// URI."""
        self.s3_client.upload_file(local_path, bucket, s3_key)
        uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"Uploaded {local_path} -> {uri}")
        return uri

    def _upload_config(self) -> str:
        """Serialize config to YAML, upload to S3 alongside the output path."""
        output_bucket, output_prefix = self._parse_s3_uri(
            self.config.output.s3_path or ""
        )
        s3_key = "/".join(
            s
            for s in [
                output_prefix.rstrip("/"),
                "config",
                "args.yaml",
            ]
            if s
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            self.config.to_yaml(f.name)
            tmp_path = f.name

        try:
            return self._upload_file_to_s3(tmp_path, output_bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    def _upload_source_dir(self) -> str:
        """Tar up the prepared source directory and upload to S3.

        The SageMaker script-mode toolkit reads the `sagemaker_submit_directory`
        hyperparameter (an S3 URI), downloads the archive, and extracts it to
        /opt/ml/code before running the entry point.

        In torchrun mode, the DDP toolkit auto-detects ``requirements.txt`` by
        filename regardless of the ``sagemaker_requirements`` hyperparameter.
        To ensure the lighter torchrun requirements (no vLLM, pinned torch)
        are used, we package ``requirements_torchrun.txt`` as
        ``requirements.txt`` inside the tarball.
        """
        output_bucket, output_prefix = self._parse_s3_uri(
            str(self.config.output.s3_path)
        )
        s3_key = "/".join(
            s for s in [output_prefix.rstrip("/"), "source", "sourcedir.tar.gz"] if s
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tmp_path = f.name

        try:
            swap_reqs = self._is_torchrun_mode

            def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
                if not swap_reqs:
                    return tarinfo
                if tarinfo.name == "./requirements.txt":
                    return None
                if tarinfo.name == "./requirements_torchrun.txt":
                    tarinfo.name = "./requirements.txt"
                return tarinfo

            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(self._source_dir, arcname=".", filter=_tar_filter)
            return self._upload_file_to_s3(tmp_path, output_bucket, s3_key)
        finally:
            os.unlink(tmp_path)

    def _build_boto3_hyperparameters(self, source_s3_uri: str, region: str) -> dict:
        """Build HyperParameters dict for the boto3 CreateTrainingJob request.

        In torchrun mode (vllm_mode=none), the entry point is train.py
        with torch_distributed enabled so the toolkit launches via torchrun
        (not MPI). The toolkit auto-detects GPU count for nproc_per_node.
        Non-sagemaker_* hyperparameters are passed as CLI args to the script.
        """
        if self._is_torchrun_mode:
            return {
                "sagemaker_program": json.dumps("train.py"),
                "sagemaker_submit_directory": json.dumps(source_s3_uri),
                "sagemaker_region": json.dumps(region),
                "sagemaker_requirements": json.dumps(self._requirements_file),
                "sagemaker_torch_distributed_enabled": json.dumps(True),
                "config": "/opt/ml/input/data/config/args.yaml",
            }

        return {
            "sagemaker_program": json.dumps("entrypoint.sh"),
            "sagemaker_submit_directory": json.dumps(source_s3_uri),
            "sagemaker_region": json.dumps(region),
        }

    def _submit_via_boto3(
        self, image_uri: str, input_channels: dict, wait: bool
    ) -> str:
        """Submit a CreateTrainingJob request directly via boto3.

        Used when the installed SDK v3 does not expose training_plan on the Compute
        class.  TrainingPlanArn is injected via a before-call event hook to bypass
        older botocore service models that don't include the field.

        Source code is delivered via the sagemaker_program / sagemaker_submit_directory
        HyperParameters — the same script-mode mechanism the SDK Estimator uses.
        The toolkit reads these from /opt/ml/input/config/hyperparameters.json,
        downloads the tar.gz from S3, and extracts it to /opt/ml/code.
        """
        region = self.session.boto_session.region_name
        sm_client = boto3.client("sagemaker", region_name=region)

        training_plan_arn = self.config.infra.training_plan_arn
        if training_plan_arn:
            # Inject TrainingPlanArn into ResourceConfig after botocore validates the
            # body, because older botocore service models don't know this field yet.
            def _inject_training_plan(params, **kwargs):
                body = json.loads(params["body"])
                body["ResourceConfig"]["TrainingPlanArn"] = training_plan_arn
                encoded = json.dumps(body).encode("utf-8")
                params["body"] = encoded
                params["headers"]["Content-Length"] = str(len(encoded))

            sm_client.meta.events.register(
                "before-call.sagemaker.CreateTrainingJob",
                _inject_training_plan,
            )
            logger.info(f"Training plan: {training_plan_arn}")

        source_s3_uri = self._upload_source_dir()
        logger.info(f"Source code uploaded: {source_s3_uri}")

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name = f"{self._job_name_prefix}-{self.config.job.name}-{timestamp}"

        environment = self._job_environment()
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

        request: dict = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": self.role,
            "InputDataConfig": input_data_config,
            "OutputDataConfig": {
                "S3OutputPath": str(self.config.output.s3_path),
                "CompressionType": "NONE",
                # infra.kms_key_id is opt-in (absent by default, matching
                # prior behavior exactly) — see the SDK v3 OutputDataConfig
                # construction below for the full rationale. This boto3-direct
                # path is reachable from a normal (non-mixed-instance-groups)
                # sft_verl config whenever infra.training_plan_arn is set, so
                # it needs the same key, not just the SDK v3 path.
                **(
                    {"KmsKeyId": str(self.config.infra.kms_key_id)}
                    if self.config.infra.kms_key_id
                    else {}
                ),
            },
            "ResourceConfig": self._boto3_resource_config(),
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.config.job.max_runtime_hours * 3600,
            },
            # Script-mode hyperparameters: the toolkit reads these from
            # /opt/ml/input/config/hyperparameters.json, downloads the tar.gz
            # from S3, and extracts it to /opt/ml/code.
            "HyperParameters": self._build_boto3_hyperparameters(source_s3_uri, region),
            "Environment": environment,
        }

        if self.config.infra.keep_alive_seconds:
            if self.config.infra.uses_mixed_instance_types():
                logger.warning(
                    "infra.keep_alive_seconds is ignored for mixed-type "
                    "dedicated_vllm (SageMaker warm pools are incompatible "
                    "with InstanceGroups)."
                )
            else:
                request["ResourceConfig"][
                    "KeepAlivePeriodInSeconds"
                ] = self.config.infra.keep_alive_seconds

        if self.config.output.checkpoint_s3_path:
            if self.config.infra.uses_mixed_instance_types():
                # SageMaker CheckpointConfig is one S3 URI two-way-synced from
                # every instance. On mixed InstanceGroups the vLLM node's empty
                # /opt/ml/checkpoints races the training node's DeepSpeed
                # shards (LIST/PUT/DELETE SlowDown; job Failed after a
                # successful train). Spot is already rejected for this path,
                # so live checkpoint sync is unused. Final weights still go
                # to output.s3_path via /opt/ml/model.
                logger.warning(
                    "Skipping SageMaker CheckpointConfig for mixed-type "
                    "dedicated_vllm (two-way sync from every InstanceGroup "
                    "races DeepSpeed shards). Training ranks will aws s3 "
                    "sync /opt/ml/checkpoints to checkpoint_s3_path "
                    "themselves; the vLLM node does not write that prefix."
                )
            else:
                request["CheckpointConfig"] = {
                    "S3Uri": str(self.config.output.checkpoint_s3_path),
                    "LocalPath": "/opt/ml/checkpoints",
                }

        if self.config.infra.use_spot_instances:
            request["EnableManagedSpotTraining"] = True
            request["StoppingCondition"][
                "MaxWaitTimeInSeconds"
            ] = self._spot_max_wait_seconds()
            logger.info(
                "Managed Spot Training enabled (boto3): MaxWaitTimeInSeconds=%ss",
                request["StoppingCondition"]["MaxWaitTimeInSeconds"],
            )

        if self.config.infra.vpc:
            request["VpcConfig"] = {
                "Subnets": self.config.infra.vpc.subnets,
                "SecurityGroupIds": self.config.infra.vpc.security_groups,
            }

        # ACP IS031.1: encrypt inter-container traffic for training jobs.
        apply_inter_container_encryption(
            request, self.config.infra.encrypt_inter_container_traffic
        )

        logger.info(f"Submitting training job via boto3: {job_name}")
        response = sm_client.create_training_job(**request)
        logger.info(
            f"Training job submitted: {job_name} "
            f"(ARN: {response.get('TrainingJobArn')})"
        )

        if wait:
            logger.info("Waiting for training job to complete...")
            waiter = sm_client.get_waiter("training_job_completed_or_stopped")
            waiter.wait(TrainingJobName=job_name)

        return job_name

    def _boto3_resource_config(self) -> dict:
        """Build ResourceConfig for CreateTrainingJob.

        Homogeneous jobs use InstanceType + InstanceCount. Mixed dedicated_vllm
        uses InstanceGroups (mutually exclusive with the top-level pair).
        """
        infra = self.config.infra
        resource: dict = {"VolumeSizeInGB": infra.volume_size_gb}
        if infra.uses_mixed_instance_types():
            assert infra.vllm_instance_type is not None
            resource["InstanceGroups"] = [
                {
                    "InstanceGroupName": "vllm",
                    "InstanceType": infra.vllm_instance_type,
                    "InstanceCount": infra.vllm_instance_count,
                },
                {
                    "InstanceGroupName": "training",
                    "InstanceType": infra.instance_type,
                    "InstanceCount": infra.mixed_training_instance_count(),
                },
            ]
            return resource
        resource["InstanceType"] = infra.instance_type
        resource["InstanceCount"] = infra.instance_count
        return resource

    def _spot_max_wait_seconds(self) -> int:
        """MaxWaitTimeInSeconds for managed spot training.

        Defaults to max_runtime_hours when job.max_wait_hours is unset. Validation
        guarantees max_wait_hours >= max_runtime_hours when both are set.
        """
        hours = self.config.job.max_wait_hours or self.config.job.max_runtime_hours
        return hours * 3600

    # ── SDK v3 (ModelTrainer) builders ──────────────────────────────────────

    def _build_model_trainer(self, image_uri: str, input_data):
        """Build a ModelTrainer (SDK v3).

        In torchrun mode (vllm_mode=none), the entry point is train.py
        with SageMaker-native Torchrun distribution so EFA is set up properly.
        """
        from sagemaker.train.configs import (
            CheckpointConfig,
            Compute,
            Networking,
            OutputDataConfig,
            SourceCode,
            StoppingCondition,
        )
        from sagemaker.train.model_trainer import ModelTrainer

        if self._is_torchrun_mode:
            from sagemaker.train.distributed import Torchrun

            entry_script = "train.py"
        else:
            entry_script = "entrypoint.sh"

        source_code = SourceCode(
            source_dir=self._source_dir,
            entry_script=entry_script,
            requirements=self._requirements_file,
        )

        compute_kwargs: dict = {
            "instance_type": self.config.infra.instance_type,
            "instance_count": self.config.infra.instance_count,
            "volume_size_in_gb": self.config.infra.volume_size_gb,
            "keep_alive_period_in_seconds": self.config.infra.keep_alive_seconds
            or None,
        }
        stopping_kwargs: dict = {
            "max_runtime_in_seconds": self.config.job.max_runtime_hours * 3600,
        }
        if self.config.infra.use_spot_instances:
            compute_kwargs["enable_managed_spot_training"] = True
            stopping_kwargs["max_wait_time_in_seconds"] = self._spot_max_wait_seconds()
            logger.info(
                "Managed Spot Training enabled (SDK v3): max_wait=%ss",
                stopping_kwargs["max_wait_time_in_seconds"],
            )

        compute = Compute(**compute_kwargs)
        stopping_condition = StoppingCondition(**stopping_kwargs)

        output_data_config = OutputDataConfig(
            s3_output_path=self.config.output.s3_path,
            compression_type="NONE",
            # infra.kms_key_id is opt-in (None by default, matching prior
            # behavior exactly) — set it when the destination bucket's own
            # default encryption key isn't one your training role can use.
            # SageMaker has no separate CheckpointConfig KMS field;
            # checkpoints inherit this same key.
            **(
                {"kms_key_id": self.config.infra.kms_key_id}
                if self.config.infra.kms_key_id
                else {}
            ),
        )

        trainer_kwargs = {
            "training_image": image_uri,
            "source_code": source_code,
            "role": self.role,
            "compute": compute,
            "sagemaker_session": self.session,
            "base_job_name": f"{self._job_name_prefix}-{self.config.job.name}",
            "stopping_condition": stopping_condition,
            "output_data_config": output_data_config,
        }

        if self._is_torchrun_mode:
            trainer_kwargs["distributed"] = Torchrun(
                process_count_per_node=self._gpus_per_node,
            )
            trainer_kwargs["hyperparameters"] = {
                "config": "/opt/ml/input/data/config/args.yaml",
            }
            logger.info(
                f"Torchrun distribution: {self._gpus_per_node} processes/node, "
                f"entry_script={entry_script}"
            )

        if self.config.output.checkpoint_s3_path:
            trainer_kwargs["checkpoint_config"] = CheckpointConfig(
                s3_uri=self.config.output.checkpoint_s3_path,
                local_path="/opt/ml/checkpoints",
            )

        job_environment = self._job_environment()
        if job_environment:
            trainer_kwargs["environment"] = job_environment

        # ACP IS031.1: always attach a Networking config carrying the
        # inter-container traffic encryption flag; fold in VPC subnets/SGs
        # when a VPC is also configured.
        networking_kwargs: dict = {
            "enable_inter_container_traffic_encryption": (
                self.config.infra.encrypt_inter_container_traffic
            ),
        }
        if self.config.infra.vpc:
            networking_kwargs["subnets"] = self.config.infra.vpc.subnets
            networking_kwargs["security_group_ids"] = (
                self.config.infra.vpc.security_groups
            )
            logger.info(
                f"VPC configured: subnets={self.config.infra.vpc.subnets}, "
                f"security_groups={self.config.infra.vpc.security_groups}"
            )
        trainer_kwargs["networking"] = Networking(**networking_kwargs)
        logger.info(
            "Inter-container traffic encryption (SDK v3): %s",
            self.config.infra.encrypt_inter_container_traffic,
        )

        return ModelTrainer(**trainer_kwargs), input_data

    def _build_input_data_v3(self, input_channels: dict):
        """Convert input channels dict to SDK v3 InputData list."""
        from sagemaker.train.configs import InputData

        input_data_config = []
        for channel_name, data_source in input_channels.items():
            input_data_config.append(
                InputData(
                    channel_name=channel_name,
                    data_source=data_source,
                )
            )
        return input_data_config

    # ── SDK v2 (Estimator) builders ─────────────────────────────────────────

    @staticmethod
    def _get_estimator_class():
        """Return the best available Estimator class.

        Prefers sagemaker.pytorch.PyTorch (SDK v2). Falls back to the base
        sagemaker.estimator.Estimator if PyTorch is not importable (some SDK v3
        stripped installs remove the framework-specific subclasses).
        """
        try:
            from sagemaker.pytorch import PyTorch

            return PyTorch
        except (ImportError, ModuleNotFoundError):
            from sagemaker.estimator import Estimator

            return Estimator

    def _build_estimator(self, image_uri: str):
        """Build an Estimator (SDK v2).

        In torchrun mode (vllm_mode=none), enables pytorchddp distribution
        and points the entry_point at train.py so SageMaker manages torchrun.
        """
        EstimatorClass = self._get_estimator_class()

        job_name = f"{self._job_name_prefix}-{self.config.job.name}"

        if self._is_torchrun_mode:
            entry_point = "train.py"
        else:
            entry_point = "entrypoint.sh"

        estimator_kwargs = {
            "image_uri": image_uri,
            "source_dir": self._source_dir,
            "entry_point": entry_point,
            "role": self.role,
            "instance_type": self.config.infra.instance_type,
            "instance_count": self.config.infra.instance_count,
            "volume_size": self.config.infra.volume_size_gb,
            "sagemaker_session": self.session,
            "base_job_name": job_name,
            "output_path": self.config.output.s3_path,
            "max_run": self.config.job.max_runtime_hours * 3600,
            "keep_alive_period_in_seconds": self.config.infra.keep_alive_seconds,
            "disable_output_compression": True,
            # ACP IS031.1: encrypt inter-container traffic for training jobs.
            "encrypt_inter_container_traffic": (
                self.config.infra.encrypt_inter_container_traffic
            ),
        }

        # infra.kms_key_id is opt-in (absent by default, matching prior
        # behavior exactly) — this SDK v2 path is reachable whenever the
        # installed sagemaker package predates ModelTrainer (see
        # _is_sdk_v3()), independent of which method is training, so it
        # needs the same key as the SDK v3 path, not just that one.
        if self.config.infra.kms_key_id:
            estimator_kwargs["output_kms_key"] = self.config.infra.kms_key_id

        if self.config.infra.use_spot_instances:
            estimator_kwargs["use_spot_instances"] = True
            estimator_kwargs["max_wait"] = self._spot_max_wait_seconds()
            logger.info(
                "Managed Spot Training enabled (SDK v2): max_wait=%ss",
                estimator_kwargs["max_wait"],
            )

        if self._is_torchrun_mode:
            estimator_kwargs["distribution"] = {"torch_distributed": {"enabled": True}}
            estimator_kwargs["hyperparameters"] = {
                "config": "/opt/ml/input/data/config/args.yaml",
            }
            logger.info(
                f"Torch distributed (torchrun): entry_point={entry_point}, "
                f"instance_count={self.config.infra.instance_count}"
            )

        if self.config.infra.training_plan_arn:
            estimator_kwargs["training_plan"] = self.config.infra.training_plan_arn
            logger.info(f"Training plan: {self.config.infra.training_plan_arn}")

        job_environment = self._job_environment()
        if job_environment:
            estimator_kwargs["environment"] = job_environment

        if self.config.output.checkpoint_s3_path:
            estimator_kwargs["checkpoint_s3_uri"] = (
                self.config.output.checkpoint_s3_path
            )
            estimator_kwargs["checkpoint_local_path"] = "/opt/ml/checkpoints"

        if self.config.infra.vpc:
            estimator_kwargs["subnets"] = self.config.infra.vpc.subnets
            estimator_kwargs["security_group_ids"] = (
                self.config.infra.vpc.security_groups
            )

        return EstimatorClass(**estimator_kwargs)

    def _build_multi_node_estimator(self, image_uri: str):
        """Build Estimator with heterogeneous instance groups for multi-node topology (SDK v2)."""
        EstimatorClass = self._get_estimator_class()
        from sagemaker.instance_group import InstanceGroup

        mn = self.config.infra.multi_node
        if mn is None:
            raise ValueError("multi_node config required for multi_node topology")

        vllm_group = InstanceGroup(
            "vllm", mn.vllm_instance_type, mn.vllm_instance_count
        )
        training_group = InstanceGroup(
            "training", mn.training_instance_type, mn.training_instance_count
        )

        job_name = f"{self._job_name_prefix}-{self.config.job.name}"

        estimator_kwargs = {
            "image_uri": image_uri,
            "source_dir": self._source_dir,
            "entry_point": "entrypoint.sh",
            "role": self.role,
            "instance_groups": [vllm_group, training_group],
            "volume_size": self.config.infra.volume_size_gb,
            "sagemaker_session": self.session,
            "base_job_name": job_name,
            "output_path": self.config.output.s3_path,
            "max_run": self.config.job.max_runtime_hours * 3600,
            # ACP IS031.1: encrypt inter-container traffic for training jobs.
            "encrypt_inter_container_traffic": (
                self.config.infra.encrypt_inter_container_traffic
            ),
        }

        # infra.kms_key_id is opt-in (absent by default, matching prior
        # behavior exactly) — this heterogeneous-InstanceGroups path is
        # reachable for any method using topology: multi_node (e.g. GRPO
        # with a dedicated vLLM instance group), independent of which SDK
        # version is installed, so it needs the same key as the other
        # three launch paths, not just those.
        if self.config.infra.kms_key_id:
            estimator_kwargs["output_kms_key"] = self.config.infra.kms_key_id

        if self.config.infra.training_plan_arn:
            estimator_kwargs["training_plan"] = self.config.infra.training_plan_arn
            logger.info(f"Training plan: {self.config.infra.training_plan_arn}")

        job_environment = self._job_environment()
        if job_environment:
            estimator_kwargs["environment"] = job_environment

        if self.config.output.checkpoint_s3_path:
            estimator_kwargs["checkpoint_s3_uri"] = (
                self.config.output.checkpoint_s3_path
            )
            estimator_kwargs["checkpoint_local_path"] = "/opt/ml/checkpoints"

        if self.config.infra.vpc:
            estimator_kwargs["subnets"] = self.config.infra.vpc.subnets
            estimator_kwargs["security_group_ids"] = (
                self.config.infra.vpc.security_groups
            )

        return EstimatorClass(**estimator_kwargs)

    # ── Launch ──────────────────────────────────────────────────────────────

    def launch(self, wait: bool = False) -> str:
        """Submit the SageMaker training job.

        Args:
            wait: If True, block until the job completes.

        Returns:
            The training job name.
        """
        sdk_version = "v3 (ModelTrainer)" if _is_sdk_v3() else "v2 (Estimator)"
        logger.info(f"SageMaker SDK: {sdk_version}")
        logger.info(
            f"Launching {self.config.method.type} training job: {self.config.job.name}"
        )
        logger.info(f"Topology: {self.config.infra.topology.value}")
        logger.info(f"vLLM mode: {self.config.infra.vllm_mode.value}")
        logger.info(f"Model: {self.config.model.name}")
        if self.config.infra.uses_mixed_instance_types():
            train_count = self.config.infra.mixed_training_instance_count()
            logger.info(
                "Mixed dedicated_vllm InstanceGroups: "
                f"vLLM {self.config.infra.vllm_instance_type} "
                f"x{self.config.infra.vllm_instance_count}, "
                f"training {self.config.infra.instance_type} x{train_count}"
            )
        else:
            logger.info(
                f"Instance: {self.config.infra.instance_type} "
                f"x{self.config.infra.instance_count}"
            )
        if self._is_torchrun_mode:
            logger.info(
                f"Torchrun mode: SageMaker-managed distribution, "
                f"{self._gpus_per_node} GPUs/node, entry_point=train.py"
            )

        region = self.session.boto_session.region_name
        fw_version = self.config.container.framework_version or "2.8.0"
        image_instance_type = self.config.infra.instance_type
        if self.config.infra.uses_mixed_instance_types():
            from foresightrlcore.config.profiles import preferred_image_instance_type

            image_instance_type = preferred_image_instance_type(
                self.config.infra.instance_type,
                self.config.infra.vllm_instance_type or self.config.infra.instance_type,
            )
        image_uri = _resolve_image_uri(
            region=region,
            framework_version=fw_version,
            instance_type=image_instance_type,
            framework=self.config.container.framework,
            image_uri=self.config.container.image_uri,
        )
        logger.info(f"Container image: {image_uri}")

        if self.config.secrets and self.config.secrets.entries:
            logger.info(
                "Secrets configured via IDPS provider (%d entries). "
                "Resolution will happen at container runtime.",
                len(self.config.secrets.entries),
            )

        config_s3_uri = self._upload_config()

        input_channels = {"config": config_s3_uri}
        # merge_lora_only reads its checkpoint via output.checkpoint_s3_path
        # (SageMaker's own CheckpointConfig sync, not an input channel) and
        # never touches training data at all — config.data.train_path is
        # legitimately unset for this mode, so it must not be mounted as
        # the 'train' channel (an unset/non-S3 value here would fail at
        # submit time even though config validation allows it).
        if not self._is_merge_lora_only:
            input_channels["train"] = str(self.config.data.train_path)
        if self.config.data.eval_path:
            input_channels["eval"] = self.config.data.eval_path

        # Pre-staged model: if model.name is an s3:// prefix, mount it as a
        # SageMaker 'model' input channel. SageMaker downloads the prefix to
        # /opt/ml/input/data/model BEFORE the entrypoint runs, so vLLM and
        # transformers both load from local disk and never touch HF Hub —
        # this eliminates the multi-rank Hub-cache race that crashes
        # from_pretrained() with "shard N not found" on large models.
        if self.config.model.is_s3_uri:
            input_channels["model"] = self.config.model.name
            logger.info(
                f"Pre-staged model: '{self.config.model.name}' will be "
                f"mounted at /opt/ml/input/data/model (skipping HF Hub)"
            )

        if self.config.environment:
            logger.info(
                f"Environment variables: {list(self.config.environment.keys())}"
            )

        # Mixed dedicated_vllm uses SageMaker InstanceGroups. SDK v3
        # ModelTrainer has known bugs injecting instance_type alongside
        # instance_groups, so this path always submits via boto3.
        if self.config.infra.uses_mixed_instance_types():
            logger.info(
                "vllm_instance_type is set; submitting via boto3 "
                "CreateTrainingJob with ResourceConfig.InstanceGroups."
            )
            return self._submit_via_boto3(image_uri, input_channels, wait)

        # SDK v3 ModelTrainer does not expose training_plan on the Compute class.
        # Use direct boto3 submission which injects TrainingPlanArn natively and
        # delivers source code via a "code" input channel (same as the SDK Estimator).
        if self.config.infra.training_plan_arn and _is_sdk_v3():
            logger.info(
                "training_plan_arn is set; using direct boto3 CreateTrainingJob "
                "because SDK v3 ModelTrainer does not support training_plan."
            )
            return self._submit_via_boto3(image_uri, input_channels, wait)

        if _is_sdk_v3():
            return self._launch_v3(image_uri, input_channels, wait)
        else:
            return self._launch_v2(image_uri, input_channels, wait)

    def _launch_v3(self, image_uri: str, input_channels: dict, wait: bool) -> str:
        """Launch using SDK v3 ModelTrainer API."""
        input_data = self._build_input_data_v3(input_channels)
        trainer, _ = self._build_model_trainer(image_uri, input_data)

        train_kwargs = {"input_data_config": input_data, "wait": wait}

        trainer.train(**train_kwargs)

        job_name = self.config.job.name
        if hasattr(trainer, "_latest_training_job") and trainer._latest_training_job:
            job_name = trainer._latest_training_job.training_job_name

        logger.info(f"Training job submitted: {job_name}")
        return job_name

    def _launch_v2(self, image_uri: str, input_channels: dict, wait: bool) -> str:
        """Launch using SDK v2 Estimator API."""
        if self.config.infra.topology == Topology.MULTI_NODE:
            estimator = self._build_multi_node_estimator(image_uri)
        else:
            estimator = self._build_estimator(image_uri)

        estimator.fit(input_channels, wait=wait)

        job_name = self.config.job.name
        if hasattr(estimator, "latest_training_job") and estimator.latest_training_job:
            job_name = estimator.latest_training_job.name

        logger.info(f"Training job submitted: {job_name}")
        return job_name
