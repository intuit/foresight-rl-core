"""GRPO method handler.

Thin subclass of :class:`TrlMethodHandler` that adds GRPO-specific
validation and trainer arguments.  All shared logic (task loading,
datasets, LoRA, hooks, train/save) lives in the base class.

For SageMaker containers, the entry point is foresightrlcore/runtime/scripts/train.py.
This handler is used by the local backend and future execution backends.
"""

from __future__ import annotations

from typing import Any

from foresightrlcore.methods import register_method
from foresightrlcore.methods.base import TrlMethodHandler


class GrpoMethodHandler(TrlMethodHandler):
    """GRPO training method handler."""

    method_type = "grpo"

    def __init__(self) -> None:
        super().__init__(
            method_type="grpo",
            default_trainer_path="trl.GRPOTrainer",
            default_config_path="trl.GRPOConfig",
        )

    def validate(self, config: Any) -> list[str]:
        """Validate GRPO-specific configuration."""
        errors = super().validate(config)
        if errors:
            return errors

        grpo_cfg = getattr(config.training, "grpo", None)
        if grpo_cfg is None:
            errors.append("training.grpo section is missing.")
            return errors

        if grpo_cfg.num_generations < 2:
            errors.append(
                f"GRPO requires num_generations >= 2, got {grpo_cfg.num_generations}"
            )

        return errors

    def build_trainer_args(self, config: Any) -> dict[str, Any]:
        """Build GRPOConfig kwargs from the unified config schema.

        Adds GRPO-specific parameters on top of the common trainer args
        provided by the base class.
        """
        args = super().build_trainer_args(config)
        grpo = config.training.grpo

        args.update(
            {
                "num_generations": grpo.num_generations,
                "max_completion_length": grpo.max_completion_length,
                "beta": grpo.beta,
                "temperature": grpo.temperature,
                "top_p": grpo.top_p,
                "top_k": grpo.top_k,
                "num_iterations": grpo.num_iterations,
                "epsilon": grpo.epsilon,
                "epsilon_high": grpo.epsilon_high,
                "scale_rewards": grpo.scale_rewards,
                "loss_type": grpo.loss_type,
                "mask_truncated_completions": grpo.mask_truncated_completions,
                "vllm_importance_sampling_correction": grpo.vllm_importance_sampling_correction,
                "repetition_penalty": grpo.repetition_penalty,
                "sync_ref_model": grpo.sync_ref_model,
                "ref_model_mixup_alpha": grpo.ref_model_mixup_alpha,
                "ref_model_sync_steps": grpo.ref_model_sync_steps,
                "multi_objective_aggregation": grpo.multi_objective_aggregation,
                "top_entropy_quantile": grpo.top_entropy_quantile,
                "vllm_importance_sampling_cap": grpo.vllm_importance_sampling_cap,
                "use_bias_correction_kl": grpo.use_bias_correction_kl,
                "disable_dropout": grpo.disable_dropout,
                "cast_lm_head_to_fp32": grpo.cast_lm_head_to_fp32,
                "ds3_gather_for_generation": grpo.ds3_gather_for_generation,
                "shuffle_dataset": grpo.shuffle_dataset,
            }
        )

        _optional = {
            "importance_sampling_level": grpo.importance_sampling_level,
            "reward_weights": grpo.reward_weights,
            "min_p": grpo.min_p,
            "generation_kwargs": grpo.generation_kwargs,
            "generation_batch_size": grpo.generation_batch_size,
            "num_generations_eval": grpo.num_generations_eval,
            "delta": grpo.delta,
            "vllm_importance_sampling_mode": grpo.vllm_importance_sampling_mode,
            "off_policy_mask_threshold": grpo.off_policy_mask_threshold,
            "max_tool_calling_iterations": grpo.max_tool_calling_iterations,
        }
        for k, v in _optional.items():
            if v is not None:
                args[k] = v

        chat_template_kwargs: dict[str, Any] = {
            "enable_thinking": config.model.enable_thinking,
        }
        if config.model.thinking_budget is not None:
            chat_template_kwargs["thinking_budget"] = config.model.thinking_budget
        args["chat_template_kwargs"] = chat_template_kwargs

        return args

    def _build_trainer_kwargs(
        self,
        *,
        config: Any,
        model_name: str,
        trl_config: Any,
        train_dataset: Any,
        eval_dataset: Any,
        peft_config: Any,
        all_callbacks: list,
        reward_fn: Any,
        task_callbacks: list,
    ) -> dict[str, Any]:
        """GRPO passes ``reward_funcs`` to the trainer constructor."""
        kwargs = super()._build_trainer_kwargs(
            config=config,
            model_name=model_name,
            trl_config=trl_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            all_callbacks=all_callbacks,
            reward_fn=reward_fn,
            task_callbacks=task_callbacks,
        )
        kwargs["reward_funcs"] = reward_fn
        return kwargs


register_method("grpo", GrpoMethodHandler)
