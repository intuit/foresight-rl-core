"""Generation-layer contracts, shared by the ``optimizer`` and ``evaluate`` ops.

These carriers and Protocols used to live in
:mod:`foresightrlcore.optimizer.interfaces`. They were promoted here so a second
operation can reuse the generation clients without importing the optimizer.
``optimizer.interfaces`` re-exports them, so its public surface is unchanged.

Stdlib only, so importing the package costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class CallUsage:
    """Token usage + latency for a single LLM call.

    Args:
        tag: Logical call name (e.g. ``business_summary``, ``mcq``).
        prompt_tokens: Input tokens (from provider usage when available).
        completion_tokens: Output tokens.
        latency_ms: Wall-clock latency in milliseconds.
    """

    tag: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@runtime_checkable
class GenerationClient(Protocol):
    """Minimal **synchronous** LLM client returning text + usage + latency.

    Implementations wrap the foresight-rl-core client managers (genos/bedrock)
    or a locally served vLLM endpoint. ``generate`` performs one call and
    returns ``(text, CallUsage)``; ``close`` releases any owned resources
    (e.g. shuts down a local ``vllm serve`` subprocess).
    """

    def generate(
        self, prompt: str, *, tag: str, max_tokens: int
    ) -> "Tuple[str, CallUsage]":
        """Call the model once; return ``(text, CallUsage)``."""
        ...

    def close(self) -> None:
        """Release any owned resources (no-op for API clients)."""
        ...


@runtime_checkable
class ModelSpec(Protocol):
    """The slice of a model config :func:`make_generation_client` reads.

    Both ``OptimizerModelConfig`` and ``EvaluateModelConfig`` satisfy this, so
    the factory serves both operations without importing either schema.
    """

    name: str
    provider: str
    max_tokens: int
    region: Optional[str]
    serving: Optional[Any]
    truncate_prompt_tokens: Optional[int]

    @property
    def temperature(self) -> Optional[float]:
        """Candidate temperature, or ``None`` to omit it from the request."""
        ...

    @property
    def resolved_model_id(self) -> str:
        """Provider-specific model id (falls back to ``name``)."""
        ...

    @property
    def serving_path(self) -> str:
        """What a local server loads weights from (falls back to ``name``).

        Separate from ``name`` because the two diverge once a job stages
        weights to local disk: the server needs the path, while reports and
        resume keys need the stable label.
        """
        ...

    @property
    def lora_adapter(self) -> Optional[str]:
        """LoRA adapter to serve on top of the weights, if any."""
        ...


__all__ = ["CallUsage", "GenerationClient", "ModelSpec"]
