import pytest

from foresightrlcore.tokenizer.loss_mask import tokenize_with_loss_mask


class StubTokenizer:
    """Deterministic, prefix-consistent fake chat template: each message
    contributes a role tag token + one token per character. Purely
    concatenative, so any prefix of the message list renders as an exact
    prefix of the full conversation's tokens — exactly the property
    tokenize_with_loss_mask relies on for real tokenizers.
    """

    _ROLE_TAGS = {"system": 100, "user": 200, "assistant": 300}

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **kwargs
    ):
        ids = []
        for m in messages:
            ids.append(self._ROLE_TAGS[m["role"]])
            ids.extend(ord(c) % 50 + 1 for c in m["content"])
        return {"input_ids": ids}


class UnstableTokenizer:
    """Fake tokenizer whose rendering is NOT prefix-consistent: the last
    token's *value* depends on whether that message is the final one in
    the list passed in, so a prefix render's tail diverges in content from
    the corresponding slice of the full render — same total length, wrong
    content. Used to verify the loud-failure path catches content
    divergence, not just a length mismatch.
    """

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **kwargs
    ):
        ids = list(range(len(messages)))
        ids[-1] = 999
        return {"input_ids": ids}


def test_masks_only_assistant_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    ids, mask = tokenize_with_loss_mask(StubTokenizer(), messages)

    assert len(ids) == len(mask)
    full = StubTokenizer().apply_chat_template(messages, tokenize=True)["input_ids"]
    assert ids == full

    # role tag + 3 chars for "sys" -> 4 tokens, all loss_mask=0
    assert mask[:4] == [0, 0, 0, 0]
    # role tag + 2 chars for "hi" -> 3 tokens, all loss_mask=0
    assert mask[4:7] == [0, 0, 0]
    # role tag + 5 chars for "hello" -> 6 tokens, all loss_mask=1
    assert mask[7:] == [1, 1, 1, 1, 1, 1]


def test_multi_turn_masks_every_assistant_span():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "bb"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "dd"},
    ]
    ids, mask = tokenize_with_loss_mask(StubTokenizer(), messages)

    # role tag + content per message: [user:1+1=2][asst:1+2=3][user:1+1=2][asst:1+2=3]
    assert mask == [0, 0] + [1, 1, 1] + [0, 0] + [1, 1, 1]
    assert sum(mask) == 6
    assert len(ids) == len(mask) == 10


def test_no_assistant_turns_gives_all_zero_mask():
    messages = [{"role": "user", "content": "only a question"}]
    ids, mask = tokenize_with_loss_mask(StubTokenizer(), messages)
    assert sum(mask) == 0
    assert len(mask) == len(ids)


def test_raises_when_tokenization_is_not_prefix_stable():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    with pytest.raises(ValueError, match="does not match"):
        tokenize_with_loss_mask(UnstableTokenizer(), messages)


def test_chat_template_kwargs_forwarded():
    seen = {}

    class RecordingTokenizer(StubTokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False, **kwargs
        ):
            seen.update(kwargs)
            return super().apply_chat_template(
                messages, tokenize, add_generation_prompt, **kwargs
            )

    messages = [{"role": "user", "content": "hi"}]
    tokenize_with_loss_mask(RecordingTokenizer(), messages, {"enable_thinking": True})
    assert seen == {"enable_thinking": True}
