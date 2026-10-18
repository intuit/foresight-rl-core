"""Shared helpers for generation clients (token accounting)."""

from __future__ import annotations

from typing import Any, Tuple


def approx_tokens(text: str) -> int:
    """Rough fallback token estimate (~4 chars/token), matching the reference."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def usage_from_langchain_message(
    message: Any, prompt: str, text: str
) -> Tuple[int, int]:
    """Extract ``(prompt_tokens, completion_tokens)`` from a LangChain AIMessage.

    Prefers the provider-reported ``usage_metadata``; when absent (or zero),
    falls back to the ~4-chars/token approximation used by the reference.
    """
    meta = getattr(message, "usage_metadata", None) or {}
    prompt_tokens = int(meta.get("input_tokens", 0) or 0)
    completion_tokens = int(meta.get("output_tokens", 0) or 0)
    if prompt_tokens == 0 and completion_tokens == 0:
        return approx_tokens(prompt), approx_tokens(text)
    return prompt_tokens, completion_tokens


def text_from_langchain_message(message: Any) -> str:
    """Return the text content from a LangChain AIMessage (or its str form)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)


__all__ = [
    "approx_tokens",
    "usage_from_langchain_message",
    "text_from_langchain_message",
]
