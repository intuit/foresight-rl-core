"""Generic SageMaker RLVR Lambda-event parsing helpers.

These functions take the raw ``event`` dict the AWS Lambda runtime
passes in and shred it into the few primitives every reward Lambda
needs:

* :func:`coerce_batch` — the rollout batch (list of samples).
* :func:`extract_response` — the model's rollout text for one sample.
* :func:`extract_state` — the ground-truth / verification context.
* :func:`extract_assistant_content` — last assistant message in a chat.

The various wire shapes accepted are the ones SageMaker Serverless RLVR
has emitted across SDK versions and per-sample / batched invocations
(verified against the Mailchimp evaluator). All helpers are pure data
shuffling — no boto3, no network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

_RESPONSE_KEYS: Tuple[str, ...] = (
    "response",
    "completion",
    "solution_str",
    "output",
    "generation",
)

_SAMPLE_FINGERPRINT_KEYS: Tuple[str, ...] = (
    "prompt",
    "messages",
    "reward_model",
    "reference_answer",
    "response",
    "completion",
    "solution_str",
)


def _content_to_str(content: Any) -> str:
    """Normalize a message ``content`` field to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def extract_assistant_content(messages: List[Dict[str, Any]]) -> str:
    """Return the content of the *last* assistant message."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return _content_to_str(msg.get("content", ""))
    return ""


def extract_response(sample: Dict[str, Any]) -> str:
    """Extract the model's rollout from a serverless RLVR sample.

    Tries (in order) top-level string fields, the last assistant turn in
    ``messages``, and the last assistant turn appended to ``prompt``.
    Returns ``""`` if none are populated.
    """
    for key in _RESPONSE_KEYS:
        value = sample.get(key)
        if value:
            text = _content_to_str(value)
            if text:
                return text

    messages = sample.get("messages")
    if isinstance(messages, list):
        text = extract_assistant_content(messages)
        if text:
            return text

    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        text = extract_assistant_content(prompt)
        if text:
            return text

    return ""


def extract_state(sample: Dict[str, Any]) -> str:
    """Extract the ground-truth / verification context from a sample.

    Preferred source is ``reference_answer.text`` (the way SageMaker's
    RLVR trainer surfaces ``reward_model.ground_truth``). Falls back to
    ``reward_model.ground_truth`` directly, then to the last user
    message content.
    """
    ref = sample.get("reference_answer")
    if isinstance(ref, dict):
        text = ref.get("text") or ref.get("state")
        if isinstance(text, str) and text.strip():
            return text

    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if isinstance(gt, str) and gt.strip():
            return gt

    for collection_key in ("messages", "prompt"):
        collection = sample.get(collection_key)
        if isinstance(collection, list):
            for msg in reversed(collection):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    text = _content_to_str(msg.get("content", ""))
                    if text:
                        return text
    return ""


def _looks_like_sample(obj: Any) -> bool:
    """True if ``obj`` looks like a single RLVR sample (not a wrapper)."""
    return isinstance(obj, dict) and any(k in obj for k in _SAMPLE_FINGERPRINT_KEYS)


def coerce_batch(event: Any) -> List[Dict[str, Any]]:
    """Pull the rollout batch out of various Lambda invocation shapes.

    Accepts (in priority order):

    * a JSON list of samples,
    * ``{"batch": [...]}`` or ``{"input": [...]}``,
    * ``{"body": "<json-string with batch>"}`` (API Gateway shape used
      during smoke tests),
    * a single sample dict (per-sample invocation) — detected by
      sample-shape fingerprints.
    """
    if isinstance(event, list):
        return [item for item in event if isinstance(item, dict)]
    if not isinstance(event, dict):
        return []
    if isinstance(event.get("batch"), list):
        return [item for item in event["batch"] if isinstance(item, dict)]
    body = event.get("body")
    if isinstance(body, str):
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict) and isinstance(decoded.get("batch"), list):
            return [item for item in decoded["batch"] if isinstance(item, dict)]
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
        if _looks_like_sample(decoded):
            return [decoded]
    if isinstance(event.get("input"), list):
        return [item for item in event["input"] if isinstance(item, dict)]
    if _looks_like_sample(event.get("input")):
        return [event["input"]]
    if _looks_like_sample(event):
        return [event]
    return []
