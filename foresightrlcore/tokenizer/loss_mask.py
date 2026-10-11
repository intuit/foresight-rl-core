"""Template-agnostic per-turn loss masking via incremental tokenization.

HuggingFace's own ``apply_chat_template(..., return_assistant_tokens_mask=True)``
requires the model's chat template to wrap assistant turns in a Jinja
``{% generation %}...{% endgeneration %}`` block. Neither Qwen3.5's nor
Gemma 4's real shipped templates have this (confirmed by grepping both
templates directly), so that mechanism doesn't work here without first
patching the template string — fragile, model-specific string surgery that
has to be re-derived per model family.

This module avoids needing the template to cooperate at all: it re-renders
the growing message prefix one turn at a time and diffs consecutive
prefixes to find exactly which token span each turn contributed. Verified
empirically against the real Gemma 4 tokenizer: each prefix's tokens are an
exact prefix of the full conversation's tokens, so the per-turn spans
partition the full sequence with no gaps or overlaps, for any chat template.
"""

from __future__ import annotations

from typing import Any


def tokenize_with_loss_mask(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize a full conversation and compute a per-token loss mask.

    Loss is 1 for every token contributed by an ``assistant`` turn (the
    "full assistant turn" policy — matches
    :mod:`foresightrlcore.methods.verl_sft_dataset`), 0 otherwise.

    Raises ``ValueError`` if the incrementally-reconstructed length doesn't
    match a single full-conversation tokenization — this would mean the
    tokenizer/template isn't deterministic across prefix lengths, which
    breaks the whole approach; fail loudly rather than silently misalign
    the mask.
    """
    kwargs = chat_template_kwargs or {}

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **kwargs
    )["input_ids"]

    loss_mask: list[int] = []
    prev_len = 0
    for i in range(1, len(messages) + 1):
        prefix_ids = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=False, **kwargs
        )["input_ids"]
        # Check actual token *content*, not just length: a tokenizer/template
        # could render a shorter-or-equal-length-but-different prefix (e.g. a
        # marker token that depends on whether a message is the last one),
        # which a length-only check would miss entirely.
        if prefix_ids != full_ids[: len(prefix_ids)]:
            raise ValueError(
                f"Prefix tokenization for messages[:{i}] does not match the "
                "corresponding slice of the full conversation's tokenization "
                "— the tokenizer/chat template is not producing a stable "
                "prefix across incremental calls, so per-turn loss masking "
                "cannot be trusted for this model without further "
                "investigation."
            )
        span_len = len(prefix_ids) - prev_len
        is_assistant = messages[i - 1]["role"] == "assistant"
        loss_mask.extend([1 if is_assistant else 0] * span_len)
        prev_len = len(prefix_ids)

    if prev_len != len(full_ids):
        raise ValueError(
            f"Incremental tokenization length ({prev_len}) does not match "
            f"full-conversation tokenization length ({len(full_ids)}) — "
            "the tokenizer/chat template is not producing a stable prefix "
            "across incremental calls, so per-turn loss masking cannot be "
            "trusted for this model without further investigation."
        )

    return full_ids, loss_mask
