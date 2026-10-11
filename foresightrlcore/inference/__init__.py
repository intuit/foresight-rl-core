"""Inference substrate: any model, any provider, one OpenAI-compatible URL.

This is the piece that makes externally-authored benchmarks portable. A
benchmark never learns which provider it is talking to -- it receives a
``base_url`` and an ``api_key`` and issues ordinary OpenAI requests.

```python
from foresightrlcore.inference import endpoint_for

with endpoint_for(model) as endpoint:
    subprocess.run(cmd, env={**os.environ, **endpoint.env()})
```

Provider coverage:

* ``local``   -- ``vllm serve``, already OpenAI-native
  (:mod:`foresightrlcore.inference.local_vllm`)
* ``bedrock`` -- stdlib proxy over ``boto3`` (OSS)
* ``genos``   -- the same proxy; the client arrives with ``foresightrl-intuit``
* ``echo``    -- deterministic offline backend for tests and dry runs

A provider that is *reached over HTTP* needs no change here: register an LLM
provider, add its name to ``PROXIED_PROVIDERS``, and the proxy finds it through
:func:`~foresightrlcore.clients.llm.resolve_llm_manager`.

A provider that *owns a process* is the other kind, and it is a core edit
rather than a registration -- :class:`~foresightrlcore.inference.local_vllm.
VllmServerEndpoint` is the worked example. Implement
:class:`~foresightrlcore.inference.interfaces.EndpointHandle`, add a branch to
:func:`make_endpoint_handle`, and name the provider in ``MODEL_PROVIDERS``
(plus ``GPU_PROVIDERS`` if it needs hardware). This is the shape an agent
server would take: the benchmark still sees only a URL, so nothing downstream
learns that an agent rather than a model is answering.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from foresightrlcore.generation.interfaces import ModelSpec
from foresightrlcore.inference.backends import (
    EchoChatBackend,
    ManagerChatBackend,
    build_llm_client_config,
)
from foresightrlcore.inference.client import ChatClient, EndpointError
from foresightrlcore.inference.interfaces import (
    ChatBackend,
    ChatMessage,
    ChatResult,
    EndpointHandle,
    InferenceEndpoint,
)
from foresightrlcore.inference.proxy import (
    PLACEHOLDER_API_KEY,
    OpenAICompatProxy,
)

logger = logging.getLogger(__name__)

#: Providers served by the in-process proxy rather than by a server we start.
#: :func:`make_endpoint_handle` dispatches on this, so it cannot drift from the
#: behaviour it describes.
PROXIED_PROVIDERS = ("bedrock", "genos", "echo")


def make_endpoint_handle(
    model: ModelSpec,
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    host: str = "127.0.0.1",
    port: int = 0,
) -> EndpointHandle:
    """Return an unstarted :class:`EndpointHandle` for ``model``.

    Prefer :func:`endpoint_for`, which brackets the lifecycle.

    Raises:
        ValueError: If ``model.provider`` has no endpoint implementation.
    """
    provider = model.provider

    if provider == "local":
        from foresightrlcore.config.perf_schema import VllmServingConfig
        from foresightrlcore.inference.local_vllm import VllmServerEndpoint

        return VllmServerEndpoint(
            model.serving_path,
            model.serving or VllmServingConfig(),
            model_name=model.name,
            lora_adapter=model.lora_adapter,
        )

    if provider in PROXIED_PROVIDERS:
        backend: ChatBackend = (
            EchoChatBackend(model.name)
            if provider == "echo"
            else ManagerChatBackend(
                build_llm_client_config(
                    provider,
                    model.resolved_model_id,
                    region=model.region,
                    temperature=model.temperature,
                ),
                max_retries=max_retries,
                base_delay=base_delay,
            )
        )
        return OpenAICompatProxy(
            backend,
            model_id=model.resolved_model_id,
            model_name=model.name,
            host=host,
            port=port,
        )

    raise ValueError(
        f"no inference endpoint for provider {provider!r}; "
        f"known: {sorted(('local',) + PROXIED_PROVIDERS)}"
    )


@contextmanager
def endpoint_for(model: ModelSpec, **kwargs: object) -> Iterator[InferenceEndpoint]:
    """Bring an endpoint up for ``model`` and tear it down on exit.

    Args:
        model: Any object satisfying
            :class:`~foresightrlcore.generation.interfaces.ModelSpec`.
        **kwargs: Forwarded to :func:`make_endpoint_handle`.

    Yields:
        A live :class:`InferenceEndpoint`.
    """
    handle = make_endpoint_handle(model, **kwargs)  # type: ignore[arg-type]
    endpoint = handle.start()
    try:
        yield endpoint
    finally:
        handle.stop()


__all__ = [
    "PLACEHOLDER_API_KEY",
    "PROXIED_PROVIDERS",
    "ChatBackend",
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "EchoChatBackend",
    "EndpointError",
    "EndpointHandle",
    "InferenceEndpoint",
    "ManagerChatBackend",
    "OpenAICompatProxy",
    "build_llm_client_config",
    "endpoint_for",
    "make_endpoint_handle",
]
