"""Provider-agnostic generation clients (genos / bedrock / local vLLM / stub).

Shared by the ``optimizer`` and ``evaluate`` operations. Every client satisfies
:class:`~foresightrlcore.generation.interfaces.GenerationClient`, so an
operation picks a provider by config alone.

``foresightrlcore.optimizer.generation`` aliases this package for backwards
compatibility.
"""

from foresightrlcore.generation.factory import make_generation_client
from foresightrlcore.generation.interfaces import (
    CallUsage,
    GenerationClient,
    ModelSpec,
)

__all__ = [
    "make_generation_client",
    "CallUsage",
    "GenerationClient",
    "ModelSpec",
]
