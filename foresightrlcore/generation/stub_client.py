"""Deterministic offline generation client for tests / dry runs.

Produces reproducible text without any network or GPU, so the runner's
cross-product, checkpoint, and aggregation logic can be exercised in unit tests.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

from foresightrlcore.generation.base import approx_tokens
from foresightrlcore.generation.interfaces import CallUsage


class StubGenerationClient:
    """Return a deterministic echo derived from the prompt.

    Args:
        model_name: Included in the output for disambiguation across candidates.
        template: Optional format string with ``{model}``, ``{tag}``, ``{digest}``
            placeholders. When unset a default structured stub is returned.
    """

    def __init__(self, model_name: str = "stub", template: str | None = None) -> None:
        self.model_name = model_name
        self.template = template

    def generate(
        self, prompt: str, *, tag: str, max_tokens: int
    ) -> Tuple[str, CallUsage]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        if self.template is not None:
            text = self.template.format(model=self.model_name, tag=tag, digest=digest)
        else:
            text = (
                f"[stub:{self.model_name}] tag={tag} digest={digest}\n"
                f"{prompt[:200]}"
            )
        return text, CallUsage(
            tag=tag,
            prompt_tokens=approx_tokens(prompt),
            completion_tokens=approx_tokens(text),
            latency_ms=0.1,
        )

    def close(self) -> None:
        return None


__all__ = ["StubGenerationClient"]
