"""Thinking/reasoning tag utilities.

Shared helpers for stripping and measuring <think>/<reasoning> tags
produced by reasoning models (Qwen3, etc.). Used by reward functions
that apply thinking-length penalties.
"""

from __future__ import annotations

import re
from typing import Any


def completion_to_str(completion: Any) -> str:
    """Normalize a TRL completion (str or list[{role, content}]) to text.

    TRL passes completions either as raw strings (text format) or as a
    list of ``{"role", "content"}`` message dicts (conversational format).
    Shared by every task reward function so the normalization lives in one
    place.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return ""


def strip_thinking_tags(content: str) -> str:
    """Remove thinking/reasoning tags from reasoning model output."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    if "</think>" in content and "<think>" not in content:
        content = content.rsplit("</think>", 1)[-1]
    else:
        content = re.sub(r"</think>", "", content)
    return content.strip()


def extract_thinking_length(content: str) -> int:
    """Extract the length (in chars) of the thinking/reasoning section."""
    match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if match:
        return len(match.group(1))
    if "</think>" in content and "<think>" not in content:
        return len(content.split("</think>")[0])
    return 0


def parse_json_from_response(response_text: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    import json

    response_text = strip_thinking_tags(response_text)
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
