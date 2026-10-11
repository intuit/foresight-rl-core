"""AWS SageMaker Training Job execution backend.

Uses the GRPOJobLauncher (now in foresightrlcore.execution.aws_sagemaker_launcher)
to submit jobs. All SageMaker-specific logic (SDK v2/v3, boto3, role resolution,
source upload) is in the launcher module.
"""

from __future__ import annotations

import logging
from typing import Any

from foresightrlcore.execution import register_backend

logger = logging.getLogger(__name__)


class AwsTrainingJobBackend:
    """Execution backend for AWS SageMaker Training Jobs."""

    backend_type = "aws_training_job"

    def validate(self, config: Any) -> list[str]:
        errors: list[str] = []

        try:
            import boto3  # noqa: F401
        except ImportError:
            errors.append(
                "boto3 is not installed. Install with: uv pip install -e '.[aws]'"
            )
            return errors

        try:
            import sagemaker  # noqa: F401
        except ImportError:
            errors.append(
                "sagemaker SDK is not installed. Install with: uv pip install -e '.[aws]'"
            )
            return errors

        # merge_lora_only reads its checkpoint via output.checkpoint_s3_path
        # (SageMaker's own CheckpointConfig sync, not an input channel) and
        # never touches training data at all — data.train_path is
        # legitimately unset for that mode. Same exemption as
        # config/validation.py's _validate_data_required/_validate_execution
        # and the launcher's input-channel construction.
        sft_cfg = getattr(config.training, "sft_verl", None)
        is_merge_only = (
            config.method.type == "sft_verl"
            and sft_cfg is not None
            and sft_cfg.merge_lora_only
        )

        if not is_merge_only:
            train_path = str(config.data.train_path)
            if not train_path.startswith("s3://"):
                errors.append(
                    f"AWS backend requires s3:// data paths, got: {train_path}"
                )

        output_path = str(config.output.s3_path)
        if not output_path.startswith("s3://"):
            errors.append(
                f"AWS backend requires s3:// output paths, got: {output_path}"
            )

        return errors

    def launch(self, config: Any, wait: bool = False) -> str:
        """Launch a SageMaker training job."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        launcher = GRPOJobLauncher(config=config)
        job_name = launcher.launch(wait=wait)
        logger.info(f"SageMaker job launched: {job_name}")
        return job_name

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Query SageMaker for job status."""
        try:
            import boto3

            sm = boto3.client("sagemaker")
            response = sm.describe_training_job(TrainingJobName=job_id)
            return {
                "job_id": job_id,
                "status": response["TrainingJobStatus"],
                "secondary_status": response.get("SecondaryStatus"),
                "failure_reason": response.get("FailureReason"),
            }
        except Exception as e:
            return {"job_id": job_id, "status": "UNKNOWN", "error": str(e)}


register_backend("aws_training_job", AwsTrainingJobBackend)
