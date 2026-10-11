"""Bedrock LLM client — IAM-role auth and a LangChain-compatible shim.

Provides a reusable Bedrock client for any task that needs an LLM
judge / scorer / classifier without bringing in a new pip dependency
such as ``langchain-aws``. The shim exposes the same surface the
reward pipeline already consumes::

    client = manager.get()
    structured = client.with_structured_output(SomePydanticModel)
    result: SomePydanticModel = structured.invoke(messages)

Internally, the shim calls ``bedrock-runtime.converse(...)`` with the
Pydantic schema attached as a forced ``toolUse``. The response's
``toolUse.input`` is parsed back into the schema via
``SomePydanticModel.model_validate(...)``.

Auth is via the standard ``boto3`` credential chain — typically the
SageMaker execution role inside a training container, or the user's
local AWS profile during development. **No app secret is required.**
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_REGION_ENV_VAR = "AWS_DEFAULT_REGION"

_STRUCTURED_TOOL_NAME = "emit_structured_response"


# ---------------------------------------------------------------------------
# Message + schema translation helpers
# ---------------------------------------------------------------------------


def _split_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Translate OpenAI-style messages into Bedrock Converse format.

    Returns a ``(system_blocks, conversation)`` tuple where:
      * ``system_blocks`` is a list of ``{"text": "..."}`` items.
      * ``conversation`` is a list of ``{"role": ..., "content": [...]}``
        items with content as a list of typed parts (currently ``text``).
    """
    system_blocks: List[Dict[str, Any]] = []
    conversation: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")
        text = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)

        if role == "system":
            system_blocks.append({"text": text})
            continue

        # Bedrock Converse only supports 'user' / 'assistant' roles.
        normalized_role = "assistant" if role == "assistant" else "user"
        conversation.append({"role": normalized_role, "content": [{"text": text}]})

    return system_blocks, conversation


def text_from_converse(response: Dict[str, Any]) -> str:
    """Concatenate the text blocks of a Bedrock Converse response."""
    output = response.get("output", {}) or {}
    message = output.get("message", {}) or {}
    parts = message.get("content", []) or []
    return "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and "text" in part
    )


def token_usage_from_converse(response: Dict[str, Any]) -> Tuple[int, int]:
    """Return ``(input_tokens, output_tokens)`` as reported, zeros if absent."""
    usage = response.get("usage", {}) or {}
    return (
        int(usage.get("inputTokens", 0) or 0),
        int(usage.get("outputTokens", 0) or 0),
    )


def _pydantic_to_tool_spec(schema_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Convert any Pydantic ``BaseModel`` subclass into a Bedrock toolSpec."""
    schema = schema_cls.model_json_schema()
    return {
        "toolSpec": {
            "name": _STRUCTURED_TOOL_NAME,
            "description": (
                f"Emit a structured response matching the {schema_cls.__name__} "
                f"schema. Always call this tool exactly once."
            ),
            "inputSchema": {"json": schema},
        }
    }


def _extract_tool_args(response: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the ``toolUse.input`` payload out of a Bedrock Converse response."""
    output = response.get("output", {}) or {}
    message = output.get("message", {}) or {}
    content_parts = message.get("content", []) or []

    for part in content_parts:
        tool_use = part.get("toolUse") if isinstance(part, dict) else None
        if tool_use:
            args = tool_use.get("input") or {}
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    pass
            if isinstance(args, dict):
                return args

    raise ValueError(
        "Bedrock response did not contain a toolUse block. "
        f"Raw output keys: {list(output.keys())}"
    )


# ---------------------------------------------------------------------------
# LangChain-compatible shim
# ---------------------------------------------------------------------------


class _BedrockStructured:
    """Mimics LangChain's ``with_structured_output(Schema).invoke(messages)``.

    Holds a reference to the parent ``_BedrockChat`` and the target
    Pydantic schema; performs the Bedrock Converse call with the schema
    attached as a forced ``toolUse``.
    """

    def __init__(self, chat: "_BedrockChat", schema_cls: Type[BaseModel]):
        self._chat = chat
        self._schema_cls = schema_cls
        self._tool_spec = _pydantic_to_tool_spec(schema_cls)

    def invoke(self, messages: List[Dict[str, Any]]) -> BaseModel:
        system_blocks, conversation = _split_messages(messages)

        kwargs: Dict[str, Any] = {
            "modelId": self._chat.model_id,
            "messages": conversation,
            "toolConfig": {
                "tools": [self._tool_spec],
                "toolChoice": {"tool": {"name": _STRUCTURED_TOOL_NAME}},
            },
            "inferenceConfig": {"temperature": self._chat.temperature},
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        response = self._chat.client.converse(**kwargs)
        args = _extract_tool_args(response)
        return self._schema_cls.model_validate(args)


class _BedrockMessage:
    """The subset of a LangChain ``AIMessage`` that callers here read."""

    def __init__(self, content: str, usage_metadata: Dict[str, int]) -> None:
        self.content = content
        self.usage_metadata = usage_metadata


class _BedrockBound:
    """A chat client with per-call inference parameters already applied.

    LangChain's ``chat.bind(**kwargs).invoke(messages)`` is the shape the
    inference proxy uses for every provider. Genos gets it from ``ChatOpenAI``;
    this supplies the same two methods for Bedrock so the proxy never has to
    ask which provider it is talking to.
    """

    def __init__(self, chat: "_BedrockChat", params: Dict[str, Any]) -> None:
        self._chat = chat
        self._params = params

    def invoke(self, messages: List[Dict[str, Any]]) -> _BedrockMessage:
        system_blocks, conversation = _split_messages(messages)

        inference: Dict[str, Any] = {}
        # Anthropic rejects a request that carries both temperature and topP.
        # The client's own default must therefore yield to an explicit topP,
        # or every call for a candidate that configured only top_p fails. A
        # caller that binds both has asked for both and gets the error.
        temperature = self._params.get("temperature")
        if temperature is None and self._params.get("top_p") is None:
            temperature = self._chat.temperature
        if temperature is not None:
            inference["temperature"] = temperature
        if self._params.get("max_tokens") is not None:
            inference["maxTokens"] = int(self._params["max_tokens"])
        if self._params.get("top_p") is not None:
            inference["topP"] = float(self._params["top_p"])
        if self._params.get("stop"):
            inference["stopSequences"] = list(self._params["stop"])

        kwargs: Dict[str, Any] = {
            "modelId": self._chat.model_id,
            "messages": conversation,
            "inferenceConfig": inference,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if self._params.get("extra_body"):
            kwargs["additionalModelRequestFields"] = self._params["extra_body"]

        response = self._chat.client.converse(**kwargs)
        prompt_tokens, completion_tokens = token_usage_from_converse(response)
        return _BedrockMessage(
            text_from_converse(response),
            {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
        )


class _BedrockChat:
    """Minimal LangChain-compatible chat client wrapping a boto3 client."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        temperature: float,
    ):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature

    def with_structured_output(self, schema_cls: Type[BaseModel]) -> _BedrockStructured:
        return _BedrockStructured(self, schema_cls)

    def bind(self, **kwargs: Any) -> _BedrockBound:
        """Return a client carrying ``kwargs`` as inference parameters."""
        return _BedrockBound(self, kwargs)

    def invoke(self, messages: List[Dict[str, Any]]) -> _BedrockMessage:
        """Generate with the client's default parameters."""
        return _BedrockBound(self, {}).invoke(messages)


# ---------------------------------------------------------------------------
# Public factory + manager
# ---------------------------------------------------------------------------


def get_bedrock_llm_client(
    model_id: str,
    region: Optional[str] = None,
    temperature: float = 0.0,
) -> _BedrockChat:
    """Create a Bedrock chat client wrapping a ``boto3.bedrock-runtime`` client.

    Auth is via the standard ``boto3`` credential chain — typically the
    SageMaker execution role or a local AWS profile. ``region`` falls back
    to the ``AWS_DEFAULT_REGION`` environment variable.

    The client is built with an explicit botocore ``Config``: the default
    60s read timeout is too short for long ``converse`` generations (a single
    high-``maxTokens`` completion routinely exceeds it, which surfaces as a
    "Read timeout on endpoint URL" error and drives the caller into long retry
    backoff). We use ``read_timeout=120`` / ``connect_timeout=10`` with adaptive
    retries, matching the proven :class:`BedrockJudge` helper.
    """
    import boto3
    from botocore.config import Config

    resolved_region = region or os.environ.get(DEFAULT_BEDROCK_REGION_ENV_VAR)
    if not resolved_region:
        raise RuntimeError(
            "Bedrock region is not configured. Set llm_clients.<name>.bedrock.region "
            "in the YAML or the AWS_DEFAULT_REGION environment variable."
        )

    boto_config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        read_timeout=120,
        connect_timeout=10,
    )
    client = boto3.client(
        "bedrock-runtime", region_name=resolved_region, config=boto_config
    )
    return _BedrockChat(client=client, model_id=model_id, temperature=temperature)


class BedrockClientManager:
    """Thread-safe Bedrock LLM client (cached, no token refresh needed).

    Mirrors :class:`foresightrlcore.clients.genos.GenosClientManager`'s
    public surface (``.get()`` / ``.force_refresh()``) so callers can
    treat both managers interchangeably.

    Because IAM credentials are managed by ``boto3`` (rotation handled
    by the SDK / instance metadata service), there is no token TTL to
    track here; ``force_refresh`` simply rebuilds the underlying boto3
    client to recover from transient errors.
    """

    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self._model_id = model_id
        self._region = region
        self._temperature = temperature
        self._client: Optional[_BedrockChat] = None
        self._lock = threading.Lock()

    def get(self) -> _BedrockChat:
        """Return a (cached) Bedrock chat client, creating one on first call."""
        with self._lock:
            if self._client is None:
                logger.info(
                    "Creating Bedrock LLM client (model_id=%s, region=%s)",
                    self._model_id,
                    self._region or os.environ.get(DEFAULT_BEDROCK_REGION_ENV_VAR),
                )
                self._client = get_bedrock_llm_client(
                    model_id=self._model_id,
                    region=self._region,
                    temperature=self._temperature,
                )
        return self._client

    def force_refresh(self) -> _BedrockChat:
        """Drop the cached client and build a fresh one (e.g. after errors)."""
        logger.info("Force-refreshing Bedrock LLM client")
        with self._lock:
            self._client = None
        return self.get()
