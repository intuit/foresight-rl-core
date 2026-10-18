"""Factory dispatching a model config to a generation client.

Takes any object satisfying
:class:`~foresightrlcore.generation.interfaces.ModelSpec` -- both
``OptimizerModelConfig`` and ``EvaluateModelConfig`` do -- so a new operation
reuses every provider without the factory knowing its schema.
"""

from __future__ import annotations

import logging

from foresightrlcore.generation.interfaces import GenerationClient, ModelSpec

logger = logging.getLogger(__name__)


def make_generation_client(
    model: ModelSpec,
    *,
    max_retries: int = 10,
    base_delay: float = 3.0,
) -> GenerationClient:
    """Return a :class:`GenerationClient` for ``model`` based on its provider."""
    provider = model.provider
    if provider == "genos":
        try:
            from foresightrl_intuit.optimizer.generation.genos_client import (
                make_genos_generation_client,
            )
        except ImportError as exc:
            raise ValueError(
                "genos generation provider requires the foresightrl-intuit "
                "extension (uv pip install -e intuit/)"
            ) from exc

        return make_genos_generation_client(
            model.resolved_model_id,
            temperature=model.temperature,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    if provider == "bedrock":
        from foresightrlcore.generation.bedrock_client import BedrockGenerationClient

        return BedrockGenerationClient(
            model.resolved_model_id,
            region=model.region,
            temperature=model.temperature,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    if provider == "local":
        from foresightrlcore.config.perf_schema import VllmServingConfig
        from foresightrlcore.generation.local_vllm_client import (
            LocalVllmGenerationClient,
        )

        serving = model.serving or VllmServingConfig()
        return LocalVllmGenerationClient(
            model.name,
            serving,
            max_retries=max_retries,
            base_delay=base_delay,
            truncate_prompt_tokens=model.truncate_prompt_tokens,
        )

    raise ValueError(f"unknown generation provider: {provider!r}")


__all__ = ["make_generation_client"]
