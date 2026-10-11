"""AWS SageMaker Serverless Model Customization backend.

Wraps SageMaker SDK v3's serverless trainers (DPOTrainer, RLVRTrainer,
RLAIFTrainer) behind the ExecutionBackend protocol. Uses get-or-create
semantics for AI Registry resources: datasets, evaluators, and model
package groups are resolved by name first; if they don't exist and a
source is provided, they are created automatically.

Trainer imports are deferred to ``launch()`` because the SDK triggers
``boto3.client("sagemaker")`` at import time (needs AWS_DEFAULT_REGION).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from foresightrlcore.execution import register_backend
from foresightrlcore.execution.sagemaker_common import (
    serverless_inter_container_encryption,
)

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = frozenset({"dpo", "rlvr", "rlaif"})

_TRAINER_MAP = {
    "dpo": "sagemaker.train.dpo_trainer.DPOTrainer",
    "rlvr": "sagemaker.train.rlvr_trainer.RLVRTrainer",
    "rlaif": "sagemaker.train.rlaif_trainer.RLAIFTrainer",
}


class AwsServerlessBackend:
    """Execution backend for SageMaker serverless model customization."""

    backend_type = "aws_serverless"

    def validate(self, config: Any) -> list[str]:
        errors: list[str] = []

        try:
            import sagemaker  # noqa: F401
        except ImportError:
            errors.append(
                "sagemaker SDK is not installed. "
                "Install with: uv pip install -e '.[aws]'"
            )
            return errors

        try:
            from importlib.metadata import version

            sm_version = version("sagemaker")
            major = int(sm_version.split(".")[0])
            if major < 3:
                errors.append(
                    f"sagemaker SDK v3+ is required for serverless training, "
                    f"found v{sm_version}. Upgrade with: "
                    f"uv pip install 'sagemaker>=3.0.0'"
                )
                return errors
        except Exception:
            pass

        sl = config.execution.serverless
        if sl is None:
            errors.append(
                "execution.serverless config section is required "
                "when execution.type is 'aws_serverless'."
            )
            return errors

        method = config.method.type
        if method not in _SUPPORTED_METHODS:
            errors.append(
                f"Serverless backend supports methods {sorted(_SUPPORTED_METHODS)}, "
                f"got '{method}'."
            )

        if not sl.accept_eula:
            errors.append(
                "execution.serverless.accept_eula must be true to run "
                "serverless model customization."
            )

        has_dataset_name = bool(sl.training_dataset_name)
        has_train_path = bool(config.data and config.data.train_path)
        if not has_dataset_name and not has_train_path:
            errors.append(
                "Either execution.serverless.training_dataset_name or "
                "data.train_path must be set."
            )

        if method == "rlaif":
            # AWS guidance: pass `judge_model_id` as a hyperparameter
            # rather than `reward_model_id` (which the SDK validates
            # against an allowlist). Either one satisfies the
            # judge-model requirement.
            sl_hp = (
                getattr(config.training, "serverless", None)
                if config.training
                else None
            )
            extra = sl_hp.extra if sl_hp else {}
            has_judge = bool(sl.reward_model_id) or "judge_model_id" in extra
            if not has_judge:
                errors.append(
                    "RLAIF requires a judge model: set either "
                    "execution.serverless.reward_model_id or "
                    "training.serverless.extra.judge_model_id."
                )
            if not sl.reward_prompt_name:
                errors.append(
                    "execution.serverless.reward_prompt_name is required for RLAIF."
                )

        return errors

    def launch(self, config: Any, wait: bool = False) -> str:
        """Resolve resources, construct the trainer, and submit the job."""
        self._ensure_region()

        from sagemaker.core.helper.session_helper import Session

        session = Session()
        sl = config.execution.serverless
        method = config.method.type

        role = (config.infra.role if config.infra else None) or self._get_role(session)

        technique = self._get_customization_technique(sl, method)
        train_ds = self._resolve_dataset(
            sl.training_dataset_name, sl.training_dataset_source, technique, session
        )
        val_ds = self._resolve_dataset(
            sl.validation_dataset_name,
            sl.validation_dataset_source,
            technique,
            session,
        )
        mpg = self._resolve_model_package_group(sl, session)
        reward_prompt = (
            self._resolve_reward_prompt(sl, session) if method == "rlaif" else None
        )
        custom_reward_fn = (
            self._resolve_evaluator(sl, session) if method == "rlvr" else None
        )

        s3_output = (
            config.output.s3_path if config.output and config.output.s3_path else None
        )

        trainer = self._build_trainer(
            method=method,
            sl=sl,
            role=role,
            session=session,
            train_ds=train_ds,
            val_ds=val_ds,
            mpg=mpg,
            reward_prompt=reward_prompt,
            custom_reward_fn=custom_reward_fn,
            s3_output=s3_output,
        )

        self._apply_hyperparameters(trainer, config.training)

        logger.info(
            "Submitting serverless %s training job for model %s",
            method.upper(),
            sl.model_id,
        )
        train_kwargs: dict[str, Any] = {"wait": wait}
        if sl.wait_timeout is not None:
            train_kwargs["wait_timeout"] = sl.wait_timeout
        # ACP IS031.1: encrypt inter-container traffic. The serverless
        # trainers hardcode their TrainingJob.create() arguments, so the
        # flag has to be injected around the submit call.
        with serverless_inter_container_encryption(sl.encrypt_inter_container_traffic):
            training_job = trainer.train(**train_kwargs)
        job_name = training_job.training_job_name
        logger.info("Serverless job submitted: %s", job_name)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_region() -> None:
        """Set AWS_DEFAULT_REGION if not already set.

        Required because SageMaker SDK v3 trainer imports trigger
        ``boto3.client("sagemaker")`` at import time.
        """
        if not os.environ.get("AWS_DEFAULT_REGION"):
            os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
            logger.debug("Set AWS_DEFAULT_REGION=us-west-2 (was unset)")

    @staticmethod
    def _get_role(session: Any) -> str:
        """Resolve the SageMaker execution role via session or STS fallback."""
        try:
            return session.get_caller_identity_arn()
        except Exception:
            pass

        import boto3

        sts = boto3.client("sts")
        return sts.get_caller_identity()["Arn"]

    @staticmethod
    def _resolve_dataset(
        name: str | None,
        source: str | None,
        technique: Any,
        session: Any,
    ) -> Any:
        """Get-or-create an AI Registry dataset.

        Tries ``DataSet.get`` first. If the dataset does not exist and
        ``source`` is provided (S3 URI or local path), creates it via
        ``DataSet.create``.
        """
        if not name:
            return None
        from sagemaker.ai_registry.dataset import DataSet

        logger.info("Resolving dataset: %s", name)
        try:
            return DataSet.get(name=name, sagemaker_session=session)
        except Exception:
            if not source:
                raise
            logger.info(
                "Dataset '%s' not found; creating from source: %s", name, source
            )
            return DataSet.create(
                name=name,
                source=source,
                customization_technique=technique,
                wait=True,
                sagemaker_session=session,
            )

    @staticmethod
    def _resolve_model_package_group(sl: Any, session: Any) -> Any:
        if not sl.model_package_group:
            return None
        from sagemaker.core.resources import ModelPackageGroup

        name = sl.model_package_group
        logger.info("Resolving model package group: %s", name)
        try:
            return ModelPackageGroup.get(model_package_group_name=name)
        except Exception:
            desc = (
                sl.model_package_group_description
                or "Store models from SageMaker serverless customization"
            )
            logger.info("Creating model package group: %s", name)
            try:
                return ModelPackageGroup.create(
                    model_package_group_name=name,
                    model_package_group_description=desc,
                )
            except Exception:
                return ModelPackageGroup.get(model_package_group_name=name)

    @staticmethod
    def _resolve_reward_prompt(sl: Any, session: Any) -> Any:
        """Get-or-create an AI Registry Evaluator for the reward prompt.

        Tries ``Evaluator.get`` first. On failure, creates the evaluator
        from ``reward_prompt_source`` (S3/local path) or
        ``reward_prompt_text`` (inline text written to a temp file).
        """
        if not sl.reward_prompt_name:
            return None
        from sagemaker.ai_registry.evaluator import Evaluator

        logger.info("Resolving reward prompt evaluator: %s", sl.reward_prompt_name)
        try:
            return Evaluator.get(name=sl.reward_prompt_name, sagemaker_session=session)
        except Exception:
            source = sl.reward_prompt_source
            if not source and sl.reward_prompt_text:
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="reward_prompt_")
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(sl.reward_prompt_text)
                    source = tmp_path
                except Exception:
                    os.close(fd)
                    raise
            if not source:
                raise
            logger.info(
                "Evaluator '%s' not found; creating from source: %s",
                sl.reward_prompt_name,
                source,
            )
            from sagemaker.ai_registry.air_constants import REWARD_PROMPT

            return Evaluator.create(
                name=sl.reward_prompt_name,
                type=REWARD_PROMPT,
                source=source,
                wait=True,
                sagemaker_session=session,
            )

    @staticmethod
    def _resolve_evaluator(sl: Any, session: Any) -> Any:
        """Get-or-create an AI Registry Evaluator for the RLVR custom reward
        function.

        Resolution order
        ----------------

        1. ``Evaluator.get(custom_reward_function_name)`` — return the
           existing AI Registry evaluator if one is already published
           under that name.
        2. ``custom_reward_function_source`` set — create from the
           explicit file path / S3 URI (escape hatch for one-off
           Lambdas or external teams). Multi-file zips use
           :meth:`_create_lambda_with_extras`.
        3. Otherwise, look the name up in
           :data:`foresightrlcore.serverless.registry.LAMBDAS` and
           build a multi-file zip from the registered ``LambdaSpec``.

        When both ``source`` and a registry entry resolve, ``source``
        wins (explicit beats implicit) and a warning is logged.
        """
        if not sl.custom_reward_function_name:
            return None
        from sagemaker.ai_registry.evaluator import Evaluator

        logger.info(
            "Resolving custom reward function evaluator: %s",
            sl.custom_reward_function_name,
        )
        try:
            return Evaluator.get(
                name=sl.custom_reward_function_name, sagemaker_session=session
            )
        except Exception:
            return AwsServerlessBackend._create_evaluator_from_recipe(sl, session)

    @staticmethod
    def _create_evaluator_from_recipe(sl: Any, session: Any) -> Any:
        """Create an AI Registry Evaluator from ``source`` or the registry.

        Resolution order for the **Lambda runtime config** (runtime,
        timeout, memory, env vars):

        1. If ``custom_reward_function_name`` matches a registered
           :class:`LambdaSpec`, use that spec's values. This is the
           common case for both step 2 ("source" override of a known
           name — the user is shipping the same Lambda code with one
           file moved around, not changing what the Lambda needs at
           runtime) and step 3 ("registry" — the spec is the only
           source of truth).
        2. Otherwise (one-off Lambda with a fresh name and an explicit
           ``source``), fall back to the package-wide Bedrock-friendly
           defaults from :mod:`foresightrlcore.serverless.registry`.
           This avoids the silent-failure mode where a new
           Bedrock-judged Lambda gets AWS's 3s / 128MB defaults and
           times out on the first rollout.
        """
        from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
        from sagemaker.ai_registry.evaluator import Evaluator

        from foresightrlcore.serverless import registry

        name = sl.custom_reward_function_name
        env_defaults: dict[str, str] = {}
        runtime = registry.DEFAULT_LAMBDA_RUNTIME
        timeout_s = registry.DEFAULT_LAMBDA_TIMEOUT_S
        memory_mb = registry.DEFAULT_LAMBDA_MEMORY_MB

        if sl.custom_reward_function_source:
            if name in registry.LAMBDAS:
                spec = registry.LAMBDAS[name]
                logger.warning(
                    "custom_reward_function_source is set AND %r is registered "
                    "in foresightrlcore.serverless.registry.LAMBDAS — using the "
                    "explicit source (--source beats registry), but inheriting "
                    "runtime config (runtime=%s, timeout=%ds, memory=%dMB, "
                    "env_defaults) from the registered LambdaSpec.",
                    name,
                    spec.runtime,
                    spec.timeout_s,
                    spec.memory_mb,
                )
                env_defaults = dict(spec.env_defaults)
                runtime = spec.runtime
                timeout_s = spec.timeout_s
                memory_mb = spec.memory_mb
            main_source = sl.custom_reward_function_source
            extra_files = list(sl.custom_reward_function_extra_files or [])
        else:
            registry.discover_tasks()
            if name not in registry.LAMBDAS:
                raise ValueError(
                    f"Evaluator {name!r} not found in AI Registry and not "
                    f"registered locally. Either register the lambda in "
                    f"foresightrlcore/tasks/<name>/__init__.py via "
                    f"register_lambda(...), or set "
                    f"execution.serverless.custom_reward_function_source "
                    f"to an explicit .py path / S3 URI. "
                    f"Registered: {sorted(registry.LAMBDAS)!r}."
                )
            spec = registry.LAMBDAS[name]
            main_source, extra_files = registry.resolve_lambda(name)
            env_defaults = dict(spec.env_defaults)
            runtime = spec.runtime
            timeout_s = spec.timeout_s
            memory_mb = spec.memory_mb

        logger.info(
            "Evaluator %r not found in AI Registry; creating from %s "
            "(extra files: %d, runtime=%s, timeout=%ds, memory=%dMB)",
            name,
            main_source,
            len(extra_files),
            runtime,
            timeout_s,
            memory_mb,
        )

        if extra_files or env_defaults:
            lambda_arn = AwsServerlessBackend._create_lambda_with_extras(
                name=name,
                main_source=main_source,
                extra_files=extra_files,
                role=sl.custom_reward_function_role,
                env_vars=env_defaults,
                runtime=runtime,
                timeout_s=timeout_s,
                memory_mb=memory_mb,
            )
            return Evaluator.create(
                name=name,
                type=REWARD_FUNCTION,
                source=lambda_arn,
                role=sl.custom_reward_function_role,
                wait=True,
                sagemaker_session=session,
            )

        return Evaluator.create(
            name=name,
            type=REWARD_FUNCTION,
            source=main_source,
            role=sl.custom_reward_function_role,
            wait=True,
            sagemaker_session=session,
        )

    @staticmethod
    def _create_lambda_with_extras(
        name: str,
        main_source: str,
        extra_files: list[str],
        role: Any,
        env_vars: dict[str, str] | None = None,
        runtime: str | None = None,
        timeout_s: int | None = None,
        memory_mb: int | None = None,
    ) -> str:
        """Create a Lambda with a multi-file zip and return its ARN.

        Builds the zip from ``main_source`` plus every path in
        ``extra_files``. The main source is placed at the zip root as
        ``lambda_function.py`` (matching the SDK's hardcoded handler
        ``lambda_function.lambda_handler``); each extra is placed at
        the zip root using its file basename, so the main source can
        ``import <basename>`` it as a top-level module.

        When ``env_vars`` is provided the keys/values are forwarded to
        ``create_function``'s ``Environment.Variables`` so reward
        Lambdas can read them via ``os.environ.get(...)`` without
        any code change. Values from ``training.serverless.extra`` are
        not yet propagated here — that's a deferred roadmap item.

        ``runtime``, ``timeout_s``, and ``memory_mb`` default to the
        Bedrock-friendly values in :mod:`foresightrlcore.serverless.registry`
        (``python3.12`` / 900s / 1024MB). Earlier versions of this
        helper passed neither ``Timeout`` nor ``MemorySize`` to
        ``CreateFunction``, leaving them at AWS's global defaults
        (3s / 128MB) — a value that silently fails for any
        Bedrock-judged reward function.

        The generated ``FunctionName`` is automatically truncated to
        AWS's 64-char limit (preserving the ``SageMaker-evaluator-``
        prefix and timestamp suffix) when the spec name is long.
        """
        import io
        import zipfile
        from datetime import datetime

        import boto3

        from foresightrlcore.serverless import registry

        if runtime is None:
            runtime = registry.DEFAULT_LAMBDA_RUNTIME
        if timeout_s is None:
            timeout_s = registry.DEFAULT_LAMBDA_TIMEOUT_S
        if memory_mb is None:
            memory_mb = registry.DEFAULT_LAMBDA_MEMORY_MB

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(main_source, "lambda_function.py")
            for extra in extra_files:
                zf.write(extra, os.path.basename(extra))
        zip_buffer.seek(0)

        raw_function_name = (
            f"SageMaker-evaluator-{name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        function_name = registry.truncate_lambda_function_name(raw_function_name)
        logger.info(
            "Creating Lambda %s with %d extra files: %s "
            "(runtime=%s, timeout=%ds, memory=%dMB)",
            function_name,
            len(extra_files),
            [os.path.basename(p) for p in extra_files],
            runtime,
            timeout_s,
            memory_mb,
        )

        create_kwargs: dict[str, Any] = {
            "FunctionName": function_name,
            "Runtime": runtime,
            "Role": role,
            "Handler": "lambda_function.lambda_handler",
            "Code": {"ZipFile": zip_buffer.read()},
            "Timeout": timeout_s,
            "MemorySize": memory_mb,
        }
        if env_vars:
            create_kwargs["Environment"] = {"Variables": dict(env_vars)}

        lambda_client = boto3.client("lambda")
        response = lambda_client.create_function(**create_kwargs)
        return response["FunctionArn"]

    @staticmethod
    def _get_customization_technique(sl: Any, method_type: str) -> Any:
        """Derive the CustomizationTechnique enum for DataSet.create."""
        technique_str = sl.customization_technique or method_type
        try:
            from sagemaker.ai_registry.dataset_utils import CustomizationTechnique

            return CustomizationTechnique(technique_str.upper())
        except (ImportError, ValueError):
            return None

    def _build_trainer(
        self,
        method: str,
        sl: Any,
        role: str,
        session: Any,
        train_ds: Any,
        val_ds: Any,
        mpg: Any,
        reward_prompt: Any,
        custom_reward_fn: Any,
        s3_output: str | None,
    ) -> Any:
        """Construct the appropriate SageMaker trainer."""
        import importlib

        from sagemaker.train.common import TrainingType

        dotted_path = _TRAINER_MAP[method]
        module_path, cls_name = dotted_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        trainer_cls = getattr(mod, cls_name)

        tt_map = {"lora": TrainingType.LORA, "full": TrainingType.FULL}
        training_type = tt_map.get(sl.training_type.lower(), sl.training_type)

        common_kwargs: dict[str, Any] = {
            "model": sl.model_id,
            "training_type": training_type,
            "training_dataset": train_ds,
            "model_package_group": mpg,
            "accept_eula": sl.accept_eula,
            "role": role,
            "sagemaker_session": session,
        }
        if val_ds is not None:
            common_kwargs["validation_dataset"] = val_ds
        if s3_output:
            common_kwargs["s3_output_path"] = s3_output
        if sl.mlflow_experiment_name:
            common_kwargs["mlflow_experiment_name"] = sl.mlflow_experiment_name
        if sl.mlflow_run_name:
            common_kwargs["mlflow_run_name"] = sl.mlflow_run_name
        if sl.mlflow_resource_arn:
            common_kwargs["mlflow_resource_arn"] = sl.mlflow_resource_arn
        if sl.base_job_name:
            common_kwargs["base_job_name"] = sl.base_job_name
        if sl.kms_key_id:
            common_kwargs["kms_key_id"] = sl.kms_key_id
        if sl.networking:
            common_kwargs["networking"] = self._build_vpc_config(sl.networking)
        if sl.max_runtime_seconds is not None:
            common_kwargs["stopping_condition"] = self._build_stopping_condition(
                sl.max_runtime_seconds
            )
        if sl.tags:
            common_kwargs["tags"] = [{"Key": k, "Value": v} for k, v in sl.tags.items()]

        if method == "rlvr" and custom_reward_fn is not None:
            common_kwargs["custom_reward_function"] = custom_reward_fn
        elif method == "rlaif":
            # Only pass reward_model_id when explicitly set. Per AWS
            # guidance, when using a non-allowlisted judge (e.g.
            # Anthropic Claude inference profiles) the model should be
            # passed via the `judge_model_id` hyperparameter instead;
            # passing `reward_model_id` then triggers the SDK's
            # allowlist validation and rejects valid Bedrock IDs.
            if sl.reward_model_id:
                common_kwargs["reward_model_id"] = sl.reward_model_id
            common_kwargs["reward_prompt"] = reward_prompt

        trainer = trainer_cls(**common_kwargs)
        return trainer

    @staticmethod
    def _build_vpc_config(vpc: Any) -> Any:
        """Convert our VpcConfig Pydantic model to the SDK's VpcConfig shape."""
        from sagemaker.core.shapes import VpcConfig as SdkVpcConfig

        return SdkVpcConfig(
            subnets=vpc.subnets,
            security_group_ids=vpc.security_groups,
        )

    @staticmethod
    def _build_stopping_condition(max_runtime_seconds: int) -> Any:
        from sagemaker.train.configs import StoppingCondition

        return StoppingCondition(max_runtime_in_seconds=max_runtime_seconds)

    @staticmethod
    def _set_hp(hp: Any, name: str, value: Any) -> None:
        """Assign a hyperparameter value, widening the SDK's client-side
        enum first when the requested value is outside it.

        Background: SageMaker SDK v3's ``FineTuningOptions`` (see
        ``sagemaker.train.common.FineTuningOptions.__setattr__``) raises
        ``ValueError`` when a value is outside the recipe's ``enum`` list
        (e.g. ``global_batch_size`` is constrained to
        ``[128, 256, 512, 1024]``). AWS support confirmed (2026-05-11) that
        the server-side accepts values outside the SDK enum — empirically
        verified by submitting a job with ``global_batch_size: 64``
        (``batch64-probe-20260511224338``), which AWS accepted and stored
        verbatim. The SDK enum is therefore a UI/SDK convenience and not a
        true service constraint.

        This helper widens ``hp._specs[name]['enum']`` in place when the
        requested value isn't in it, then falls through to normal
        ``setattr``. Trainers/objects without a ``_specs`` dict (e.g. test
        mocks) are unaffected.
        """
        specs = getattr(hp, "_specs", None)
        if isinstance(specs, dict):
            spec = specs.get(name)
            if isinstance(spec, dict) and "enum" in spec and value not in spec["enum"]:
                spec["enum"] = list(spec["enum"]) + [value]
        setattr(hp, name, value)

    @classmethod
    def _apply_hyperparameters(cls, trainer: Any, training_config: Any) -> None:
        """Apply hyperparameter overrides from training.serverless to the trainer."""
        sl_hp = getattr(training_config, "serverless", None)
        if sl_hp is None:
            return

        hp = trainer.hyperparameters
        if sl_hp.global_batch_size is not None:
            cls._set_hp(hp, "global_batch_size", sl_hp.global_batch_size)
        if sl_hp.max_epochs is not None:
            cls._set_hp(hp, "max_epochs", sl_hp.max_epochs)
        if sl_hp.learning_rate is not None:
            cls._set_hp(hp, "learning_rate", sl_hp.learning_rate)

        for key, value in sl_hp.extra.items():
            cls._set_hp(hp, key, value)


register_backend("aws_serverless", AwsServerlessBackend)
