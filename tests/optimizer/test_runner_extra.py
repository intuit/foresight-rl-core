"""Runner paths not exercised by the happy-path e2e test.

Covers: the public ``run_optimizer`` wrapper, the container ``main`` CLI entry,
the S3 output-sync + checkpoint-S3 branches, and the local-serving lifecycle
(``client.start()`` bracketing a candidate's batch).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import foresightrlcore.optimizer.datasets  # noqa: F401,E402
import foresightrlcore.optimizer.judges  # noqa: F401,E402
import foresightrlcore.optimizer.modes  # noqa: F401,E402
import foresightrlcore.optimizer.prompts  # noqa: F401,E402
from foresightrlcore.config.optimizer_schema import OptimizerJobConfig  # noqa: E402
from foresightrlcore.generation.stub_client import (  # noqa: E402
    StubGenerationClient,
)
from foresightrlcore.optimizer import runner as runner_mod  # noqa: E402
from foresightrlcore.optimizer.interfaces import Record  # noqa: E402
from foresightrlcore.optimizer.registry import DATASETS  # noqa: E402
from foresightrlcore.optimizer.runner import (  # noqa: E402
    OptimizerRunner,
    main,
    run_optimizer,
)


class _MemDataset:
    def records(self, spec, sample):
        n = int(sample.get("size", 2))
        return [
            Record(
                realm_id=f"r{i}",
                epoch_date="2024-01-01",
                inputs={
                    "evidence_bundle": f"rev 10{i}",
                    "start_date": "",
                    "end_date": "d",
                },
                reference="x" * 300,
                meta={"dataset": spec.get("name", "memextra"), "shard": i % 2},
            )
            for i in range(n)
        ]


if "memextra" not in DATASETS:

    @DATASETS.register("memextra")
    def _make_memextra() -> _MemDataset:
        return _MemDataset()


def _config(**out) -> OptimizerJobConfig:
    output = {"s3_path": "/tmp/ignored"}
    output.update(out)
    return OptimizerJobConfig.model_validate(
        {
            "job": {"name": "opt-extra"},
            "optimizer": {
                "sample": {"size": 2, "seed": 13, "stratify_by": "shard"},
                "datasets": [{"name": "memextra"}],
                "prompt_set": "bts_v1",
                "models": [{"name": "cand", "provider": "genos", "max_tokens": 500}],
                "modes": ["summary_state"],
                "judges": [{"name": "length", "weight": 0}],
                "concurrency": 1,
                "checkpoint": {"enabled": True, "every_n_rows": 1},
            },
            "execution": {"type": "local"},
            "output": output,
        }
    )


@pytest.mark.requires_intuit
def test_run_optimizer_wrapper(tmp_path):
    config = _config()
    import foresightrlcore.generation as gen_pkg

    with patch.object(
        gen_pkg,
        "make_generation_client",
        side_effect=lambda model, **kw: StubGenerationClient(model_name=model.name),
    ):
        result = run_optimizer(
            config, output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "ckpt"
        )
    assert result.n_rows == 2
    assert (tmp_path / "out" / "results.parquet").exists()


def test_main_cli_entry(tmp_path, monkeypatch):
    cfg_path = tmp_path / "opt.yaml"
    _config().to_yaml(cfg_path)

    fake_result = MagicMock(n_rows=2)
    with patch.object(runner_mod, "run_optimizer", return_value=fake_result) as run_opt:
        rc = main(
            [
                "--config",
                str(cfg_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--checkpoint-dir",
                str(tmp_path / "ckpt"),
            ]
        )
    assert rc == 0
    run_opt.assert_called_once()


@pytest.mark.requires_intuit
def test_checkpoint_s3_and_output_sync(tmp_path):
    config = _config(s3_path="s3://bucket/out/", checkpoint_s3_path="s3://bucket/ckpt/")

    uploads = []
    syncs = []

    def fake_upload(local, uri):
        uploads.append(uri)

    def fake_sync(local, uri):
        syncs.append(uri)

    with (
        patch.object(runner_mod.optim_io, "s3_upload", fake_upload),
        patch.object(runner_mod.optim_io, "s3_sync_dir", fake_sync),
    ):
        runner = OptimizerRunner(
            config,
            output_dir=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
            gen_client_factory=lambda m: StubGenerationClient(model_name=m.name),
        )
        result = runner.run()

    assert result.n_rows == 2
    # Checkpoint uploaded to the derived per-file S3 key.
    assert any(u.endswith("optimizer_checkpoint.jsonl") for u in uploads)
    # Output directory synced to the configured S3 prefix.
    assert syncs and syncs[0] == "s3://bucket/out"


def _llm_config(**over) -> OptimizerJobConfig:
    d = {
        "job": {"name": "opt-llm"},
        "optimizer": {
            "sample": {"size": 1, "seed": 13, "stratify_by": "shard"},
            "datasets": [{"name": "memextra"}],
            "prompt_set": "bts_v1",
            "models": [{"name": "cand", "provider": "genos", "max_tokens": 500}],
            "modes": ["summary_state"],
            "judges": [{"name": "correctness", "weight": 1.0}],
            "judge_clients": {
                "default": {"provider": "genos", "genos": {"model_id": "m"}}
            },
            "concurrency": 1,
        },
        "execution": {"type": "local"},
        "output": {"s3_path": "/tmp/ignored"},
    }
    d.update(over)
    return OptimizerJobConfig.model_validate(d)


def test_default_manager_resolver_requires_clients(tmp_path):
    config = _config()  # no judge_clients
    runner = OptimizerRunner(config, output_dir=tmp_path / "o")
    with pytest.raises(ValueError, match="judge_clients is empty"):
        runner._default_manager_resolver("default")


def test_default_manager_resolver_dispatches(tmp_path):
    config = _llm_config()
    runner = OptimizerRunner(config, output_dir=tmp_path / "o")
    with patch(
        "foresightrlcore.clients.llm.resolve_llm_manager", return_value="mgr"
    ) as resolve:
        assert runner._default_manager_resolver("default") == "mgr"
    resolve.assert_called_once()


def test_build_judges_wires_llm_judge_and_retry(tmp_path):
    config = _llm_config()
    fake_manager = MagicMock()
    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "o",
        manager_resolver=lambda name: fake_manager,
    )
    judges, weights = runner._build_judges()
    assert "correctness" in judges
    # Retry config threaded onto the LLM judge.
    assert judges["correctness"]._max_retries == config.optimizer.retry.max_retries
    assert weights["correctness"] == 1.0


@pytest.mark.requires_intuit
def test_resolves_dataset_path_and_prompts_path(tmp_path):
    from foresightrl_intuit.optimizer.prompts.bts_v1 import _DEFAULT_BTS_V1_PATH

    data_file = tmp_path / "data.parquet"
    data_file.write_text("placeholder")  # _MemDataset ignores contents
    config = _config()
    config.optimizer.datasets[0].path = str(data_file)
    config.optimizer.prompts_path = str(_DEFAULT_BTS_V1_PATH)

    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=lambda m: StubGenerationClient(model_name=m.name),
    )
    result = runner.run()
    assert result.n_rows == 2


@pytest.mark.requires_intuit
def test_local_serving_lifecycle_is_bracketed(tmp_path):
    config = _config()

    events = []

    class _LocalClient:
        def __init__(self, name):
            self.model_name = name

        def start(self):
            events.append("start")
            return 0.0

        def generate(self, prompt, *, tag, max_tokens):
            return StubGenerationClient(self.model_name).generate(
                prompt, tag=tag, max_tokens=max_tokens
            )

        def close(self):
            events.append("close")

    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=lambda m: _LocalClient(m.name),
    )
    runner.run()
    assert events == ["start", "close"]
