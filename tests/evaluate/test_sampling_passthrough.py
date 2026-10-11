"""Sampling parameters reaching the endpoint.

A score is only comparable to a published one if the sampling that produced it
can be reproduced. ``temperature`` alone does not pin a distribution, so
``top_p``, ``top_k`` and ``repetition_penalty`` have to survive the trip from
YAML through the runner to the request body -- and stay absent when nobody asked
for them, so existing benchmarks keep their current behaviour.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest import mock

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark
from foresightrlcore.evaluate.interfaces import PromptRequest
from foresightrlcore.evaluate.runner import EvaluateRunner
from foresightrlcore.inference.interfaces import InferenceEndpoint
from tests.evaluate.conftest import FakeChatClient

RECORDS = [
    {
        "id": "q1",
        "question": "Capital of France?",
        "choices": ["Berlin", "Paris"],
        "answer": "Paris",
    }
]


def _config(tmp_path, **model_fields: Any) -> EvaluateJobConfig:
    model: Dict[str, Any] = {"name": "m1", "provider": "echo"}
    model.update(model_fields)
    return EvaluateJobConfig.model_validate(
        {
            "job": {"name": "t"},
            "execution": {"type": "local"},
            "output": {},
            "evaluate": {
                "benchmarks": [
                    {
                        "name": "mcq",
                        "type": "mcq",
                        "metrics": ["choice_match"],
                        "dataset": {"loader": "inline", "records": RECORDS},
                    }
                ],
                "models": [model],
                "concurrency": 1,
                "reporters": [],
            },
        }
    )


@contextmanager
def _fake_endpoint(model):
    yield InferenceEndpoint(
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
        model_id=model.name,
        model_name=model.name,
    )


def _run(tmp_path, config) -> FakeChatClient:
    """Run one record and hand back the client, for inspecting what was sent."""
    client = FakeChatClient({"*": "B"})
    runner = EvaluateRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        workspace_root=tmp_path / "ws",
        endpoint_factory=_fake_endpoint,
    )
    with mock.patch("foresightrlcore.evaluate.runner.ChatClient", return_value=client):
        runner.run()
    return client


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestConfigSurface:
    def test_the_sampling_fields_default_to_unset(self, tmp_path):
        model = _config(tmp_path).evaluate.models[0]
        assert model.top_p is None
        assert model.top_k is None
        assert model.repetition_penalty is None
        assert model.extra_body == {}

    def test_the_published_matching_settings_are_expressible(self, tmp_path):
        """The exact sampling block the published 1000-row runs used."""
        model = _config(
            tmp_path,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.1,
        ).evaluate.models[0]

        assert (model.temperature, model.top_p) == (0.6, 0.95)
        assert (model.top_k, model.repetition_penalty) == (20, 1.1)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("top_p", 0.0),
            ("top_p", 1.5),
            ("top_k", 0),
            ("top_k", -2),
            ("repetition_penalty", 0.0),
        ],
    )
    def test_out_of_range_sampling_is_rejected(self, tmp_path, field, value):
        with pytest.raises(ValueError):
            _config(tmp_path, **{field: value})

    def test_minus_one_is_vllms_no_limit_and_is_sent_as_is(self, tmp_path):
        # Seven of the leaderboard configs set top_k: -1 for vLLM-served models.
        assert _config(tmp_path, top_k=-1).evaluate.models[0].top_k == -1

    def test_extra_body_carries_arbitrary_provider_options(self, tmp_path):
        """reasoning_effort has no field of its own and rides in extra_body."""
        model = _config(
            tmp_path,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
        ).evaluate.models[0]

        assert model.extra_body["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


# ---------------------------------------------------------------------------
# Runner -> client
# ---------------------------------------------------------------------------


#: Captured before any patching, so the stand-in below can still build the real
#: prompt and change only the sampling.
_REAL_BUILD_PROMPT = McqBenchmark.build_prompt


def _insisting_build_prompt(self, record):
    """A benchmark that dictates its own sampling, as a task occasionally must."""
    request = _REAL_BUILD_PROMPT(self, record)
    request.repetition_penalty = 1.5
    return request


class TestRunnerForwardsSampling:
    def test_unset_sampling_is_forwarded_as_none(self, tmp_path):
        """So the client omits the fields rather than sending defaults."""
        sent = _run(tmp_path, _config(tmp_path)).call_kwargs[0]

        assert sent["top_p"] is None
        assert sent["top_k"] is None
        assert sent["repetition_penalty"] is None
        assert sent["extra"] == {}

    def test_model_config_sampling_reaches_the_client(self, tmp_path):
        config = _config(
            tmp_path, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1
        )
        sent = _run(tmp_path, config).call_kwargs[0]

        assert sent["temperature"] == 0.6
        assert sent["top_p"] == 0.95
        assert sent["top_k"] == 20
        assert sent["repetition_penalty"] == 1.1

    def test_extra_body_reaches_the_client(self, tmp_path):
        config = _config(tmp_path, extra_body={"seed": 7})
        assert _run(tmp_path, config).call_kwargs[0]["extra"] == {"seed": 7}


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    """A benchmark overrides the model where it has an opinion."""

    def test_a_benchmark_overrides_the_model(self, tmp_path):
        config = _config(tmp_path, repetition_penalty=1.1)
        with mock.patch.object(McqBenchmark, "build_prompt", _insisting_build_prompt):
            sent = _run(tmp_path, config).call_kwargs[0]

        assert sent["repetition_penalty"] == 1.5

    def test_the_model_is_used_when_the_benchmark_is_silent(self, tmp_path):
        config = _config(tmp_path, repetition_penalty=1.1)
        assert _run(tmp_path, config).call_kwargs[0]["repetition_penalty"] == 1.1

    def test_neither_set_stays_unset(self, tmp_path):
        config = _config(tmp_path)
        assert _run(tmp_path, config).call_kwargs[0]["repetition_penalty"] is None


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class TestRequestBody:
    """What actually goes on the wire, captured at urlopen."""

    @staticmethod
    def _payload(**kwargs: Any) -> Dict[str, Any]:
        from foresightrlcore.inference.client import ChatClient

        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()

        captured: List[Any] = []

        class _Response:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc):
                return False

            def read(self_inner):
                return body

        def _urlopen(request, timeout=None):
            captured.append(request)
            return _Response()

        endpoint = InferenceEndpoint(
            base_url="http://127.0.0.1:9/v1",
            api_key="k",
            model_id="m",
            model_name="m",
        )
        with mock.patch("urllib.request.urlopen", _urlopen):
            ChatClient(endpoint).chat([{"role": "user", "content": "hi"}], **kwargs)
        return json.loads(captured[0].data.decode())

    def test_unset_sampling_is_absent_from_the_body(self):
        """Not sent as null: a provider may reject an explicit null."""
        payload = self._payload(max_tokens=16)

        assert "top_p" not in payload
        assert "top_k" not in payload
        assert "repetition_penalty" not in payload

    def test_set_sampling_is_present_in_the_body(self):
        payload = self._payload(
            max_tokens=16, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1
        )

        assert payload["temperature"] == 0.6
        assert payload["top_p"] == 0.95
        assert payload["top_k"] == 20
        assert payload["repetition_penalty"] == 1.1

    def test_top_k_and_repetition_penalty_go_top_level(self):
        """vLLM reads them there; this builds the body itself, so they survive."""
        payload = self._payload(max_tokens=16, top_k=20, repetition_penalty=1.1)

        assert "extra_body" not in payload
        assert {"top_k", "repetition_penalty"} <= set(payload)

    def test_extra_is_merged_into_the_body(self):
        payload = self._payload(
            max_tokens=16,
            extra={"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
        )

        assert payload["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}

    def test_extra_wins_over_a_named_field(self):
        """The escape hatch for a provider that spells something differently."""
        payload = self._payload(max_tokens=16, top_p=0.95, extra={"top_p": 0.1})

        assert payload["top_p"] == 0.1


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestFingerprint:
    """Sampling is part of what was measured, so changing it invalidates rows."""

    @staticmethod
    def _fingerprint(config) -> str:
        from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
        from foresightrlcore.evaluate.runner import measurement_fingerprint

        catalog = load_catalog(config.evaluate.catalog_paths)
        spec = resolve_benchmark(config.evaluate.benchmarks[0], catalog)
        return measurement_fingerprint(spec, config.evaluate.models[0])

    def test_changing_top_p_invalidates_the_measurement(self, tmp_path):
        before = self._fingerprint(_config(tmp_path, top_p=0.95))
        after = self._fingerprint(_config(tmp_path, top_p=0.9))

        assert before != after

    def test_changing_repetition_penalty_invalidates_the_measurement(self, tmp_path):
        before = self._fingerprint(_config(tmp_path, repetition_penalty=1.1))
        after = self._fingerprint(_config(tmp_path, repetition_penalty=1.0))

        assert before != after

    def test_an_unchanged_sampling_block_still_resumes(self, tmp_path):
        a = _config(tmp_path, top_p=0.95, top_k=20, repetition_penalty=1.1)
        b = _config(tmp_path, top_p=0.95, top_k=20, repetition_penalty=1.1)

        assert self._fingerprint(a) == self._fingerprint(b)
