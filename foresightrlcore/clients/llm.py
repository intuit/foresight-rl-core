"""Provider-agnostic LLM client resolver.

Tasks never branch on provider strings or own provider-specific
credential loading. They validate the YAML block via
:class:`foresightrlcore.config.schema.LLMClientConfig` and call
:func:`resolve_llm_manager` to get a manager with a uniform interface.
"""

from __future__ import annotations

from typing import Union

from foresightrlcore.clients.bedrock import BedrockClientManager
from foresightrlcore.config.schema import LLMClientConfig
from foresightrlcore.plugins import get_llm_provider, load_plugins

LLMClientManager = Union[BedrockClientManager, object]


def resolve_llm_manager(cfg: LLMClientConfig) -> LLMClientManager:
    """Return a configured manager for the provider selected in ``cfg``."""
    load_plugins()
    factory = get_llm_provider(cfg.provider)
    sub = getattr(cfg, cfg.provider, None)
    if sub is None:
        raise ValueError(
            f"llm_clients provider={cfg.provider!r} but no "
            f"'{cfg.provider}:' subkey was provided"
        )
    return factory(cfg)
