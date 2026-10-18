"""Generic single-prompt mode.

Renders one template (section ``single``, key ``prompt``) with all of the
record's inputs as substitution variables, then makes a single generation call.
This is the simplest reusable mode for use cases that do not need a chain.
"""

from __future__ import annotations

from foresightrlcore.optimizer.interfaces import (
    GenerationClient,
    GenResult,
    PromptSet,
    Record,
)
from foresightrlcore.optimizer.registry import MODES

_SECTION = "single"
_KEY = "prompt"


class SingleCallMode:
    """One prompt, one call. Inputs are substituted into the template."""

    name = "single_call"

    def generate(
        self,
        record: Record,
        client: GenerationClient,
        prompt_set: PromptSet,
        *,
        max_tokens: int,
    ) -> GenResult:
        prompt = prompt_set.render(_SECTION, _KEY, **record.inputs)
        text, usage = client.generate(prompt, tag="single_call", max_tokens=max_tokens)
        return GenResult(text=text, calls=[usage])


@MODES.register("single_call")
def _make() -> SingleCallMode:
    return SingleCallMode()


__all__ = ["SingleCallMode"]
