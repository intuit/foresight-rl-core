"""End-to-end runner test using the deterministic stub client (no network/GPU).

Exercises the cross-product loop, per-judge scoring, output writing, and the
checkpoint/resume path. pandas/pyarrow are required (optimizer extra); skipped
if unavailable.
"""

from __future__ import annotations

import json
import time
from typing import List
from unittest import mock

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

# Registry side-effect imports (datasets/judges/modes/prompts).
import foresightrlcore.optimizer.datasets  # noqa: F401,E402
import foresightrlcore.optimizer.judges  # noqa: F401,E402
import foresightrlcore.optimizer.modes  # noqa: F401,E402
import foresightrlcore.optimizer.prompts  # noqa: F401,E402
from foresightrlcore.config.optimizer_schema import OptimizerJobConfig  # noqa: E402
from foresightrlcore.generation.stub_client import (  # noqa: E402
    StubGenerationClient,
)
from foresightrlcore.optimizer.interfaces import Record  # noqa: E402
from foresightrlcore.optimizer.registry import DATASETS  # noqa: E402
from foresightrlcore.optimizer.runner import OptimizerRunner  # noqa: E402

# Every case here drives the runner through the BTS `bts_v1` prompt set, which
# ships with the Intuit extension.
pytestmark = pytest.mark.requires_intuit


class _MemDataset:
    """In-memory dataset provider yielding N synthetic BTS-shaped records."""

    def records(self, spec, sample):
        n = int(sample.get("size", 3))
        out = []
        for i in range(n):
            out.append(
                Record(
                    realm_id=f"realm{i}",
                    epoch_date="2024-01-01",
                    inputs={
                        "evidence_bundle": f"revenue 100{i} expenses 50{i}",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                    reference="x" * 500,
                    meta={"dataset": spec.get("name", "memtest"), "shard": i % 2},
                )
            )
        return out


# Register once (guard against re-import).
if "memtest" not in DATASETS:

    @DATASETS.register("memtest")
    def _make_memtest() -> _MemDataset:
        return _MemDataset()


def _config(size: int = 3) -> OptimizerJobConfig:
    return OptimizerJobConfig.model_validate(
        {
            "job": {"name": "opt-test"},
            "optimizer": {
                "sample": {"size": size, "seed": 13, "stratify_by": "shard"},
                "datasets": [{"name": "memtest"}],
                "prompt_set": "bts_v1",
                "models": [
                    {"name": "cand-a", "provider": "genos", "max_tokens": 2000},
                    {"name": "cand-b", "provider": "genos", "max_tokens": 2000},
                ],
                "modes": ["summary_state"],
                "judges": [
                    {"name": "sections", "weight": 0},
                    {"name": "no_recommendations", "weight": 0},
                    {"name": "numeric_overlap", "weight": 0},
                    {"name": "length", "weight": 0},
                ],
                "concurrency": 2,
                "checkpoint": {"enabled": True, "every_n_rows": 1},
            },
            "execution": {"type": "local"},
            "output": {"s3_path": "/tmp/ignored"},
        }
    )


def _stub_factory(model):
    return StubGenerationClient(model_name=model.name)


class _FailingGenerationClient:
    """A generation client whose every call raises (simulates a failed model)."""

    def __init__(self, model_name: str = "boom") -> None:
        self.model_name = model_name

    def generate(self, prompt, *, tag, max_tokens):
        raise RuntimeError("simulated generation failure")

    def close(self):
        return None


def _failing_factory(model):
    return _FailingGenerationClient(model_name=model.name)


def test_rows_are_checkpointed_as_they_are_produced(tmp_path):
    """A row must reach the checkpoint when it is judged, not when the batch
    for its model ends.

    `_run_model_batch` evaluates the whole cross-product for one model and can
    run for hours. Checkpointing only once it returns means an interrupted
    batch discards every row it produced -- the same defect found and fixed in
    the evaluate runner, which this mirrors.

    Row counts alone cannot see this: with `every_n_rows: 1` the old code also
    synced one row at a time, just all of them after generation had finished.
    What separates the two is *when* the first sync happens relative to
    generation, so the test records how many generations had completed by
    then. Old code: all five, because the batch was over. New code: one or
    two, the second only because a worker may start the next row before the
    main thread processes the first result.
    """
    from foresightrlcore.common.io import CheckpointWriter

    generated = 0
    generated_at_first_sync: List[int] = []
    real_sync = CheckpointWriter.sync

    def spy(self: CheckpointWriter) -> None:
        generated_at_first_sync.append(generated)
        real_sync(self)

    def counting_factory(model):
        client = StubGenerationClient(model_name=model.name)
        inner = client.generate

        def generate(*args, **kwargs):
            nonlocal generated
            # The stub answers instantly, which erases the very signal under
            # test: every row finishes before the main thread observes any of
            # them, so old and new code look alike. A small delay restores the
            # ordering that a real provider would impose anyway.
            time.sleep(0.01)
            result = inner(*args, **kwargs)
            generated += 1
            return result

        client.generate = generate  # type: ignore[method-assign]
        return client

    # One model and no parallelism, so there is exactly one batch and the
    # count at the first sync is unambiguous.
    config = _config(size=5)
    config.optimizer.models = [config.optimizer.models[0]]
    config.optimizer.concurrency = 1

    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=counting_factory,
    )
    with mock.patch.object(CheckpointWriter, "sync", spy):
        runner.run()

    assert generated_at_first_sync, "the checkpoint was never synced"
    # Fixed, this reads [3, 6, 9, 12, 15, ...] -- one sync per row, three
    # generation calls apiece. Broken, it reads [15, 15, ...]: nothing is
    # persisted until every row has been generated.
    assert generated_at_first_sync[0] < generated, (
        "the first sync happened only after every row had been generated, so "
        "an interrupted batch would have salvaged nothing: "
        f"{generated_at_first_sync[0]} of {generated} generated at first sync"
    )


def test_runner_writes_all_outputs(tmp_path):
    config = _config(size=3)
    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=_stub_factory,
    )
    result = runner.run()

    # 3 records x 2 models x 1 mode = 6 rows.
    assert result.n_rows == 6
    out = tmp_path / "out"
    assert (out / "results.parquet").exists()
    assert (out / "summary.md").exists()
    assert (out / "optimizer_results.json").exists()

    doc = json.loads((out / "optimizer_results.json").read_text())
    assert doc["job_name"] == "opt-test"
    assert doc["n_rows"] == 6
    # Manifest is self-contained: every row is embedded...
    assert len(doc["rows"]) == 6
    assert {r["model"] for r in doc["rows"]} == {"cand-a", "cand-b"}
    # ...alongside a structured report.
    assert doc["summary"]["overall"]["total"] == 6
    assert len(doc["summary"]["by_model_mode"]) == 2
    judge_names = {j["judge"] for j in doc["summary"]["per_judge"]}
    assert {"sections", "length"} <= judge_names

    summary = (out / "summary.md").read_text()
    assert "GenAI Optimizer" in summary
    assert "cand-a" in summary and "cand-b" in summary


def test_results_have_per_judge_columns(tmp_path):
    import pandas as pd

    config = _config(size=2)
    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=_stub_factory,
    )
    runner.run()

    df = pd.read_parquet(tmp_path / "out" / "results.parquet")
    for judge in ("sections", "no_recommendations", "numeric_overlap", "length"):
        assert f"{judge}_score" in df.columns
        assert f"{judge}_passed" in df.columns
    assert {"model", "mode", "primary_metric_value", "cost_usd"} <= set(df.columns)


def test_checkpoint_resume_skips_completed_rows(tmp_path):
    config = _config(size=3)
    ckpt = tmp_path / "ckpt"

    calls = {"n": 0}

    def counting_factory(model):
        calls["n"] += 1
        return StubGenerationClient(model_name=model.name)

    r1 = OptimizerRunner(
        config,
        output_dir=tmp_path / "out1",
        checkpoint_dir=ckpt,
        gen_client_factory=counting_factory,
    )
    result1 = r1.run()
    assert result1.n_rows == 6
    first_pass_factory_calls = calls["n"]
    assert first_pass_factory_calls == 2  # one client per model

    # Second run reuses the same checkpoint dir -> everything already done.
    r2 = OptimizerRunner(
        config,
        output_dir=tmp_path / "out2",
        checkpoint_dir=ckpt,
        gen_client_factory=counting_factory,
    )
    result2 = r2.run()
    # All 6 combos already present in the checkpoint -> still reported.
    assert result2.n_rows == 6
    # No model batch had outstanding work -> factory not called again.
    assert calls["n"] == first_pass_factory_calls


def test_generation_error_is_not_counted_as_pass(tmp_path):
    import pandas as pd

    config = _config(size=3)
    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=_failing_factory,
    )
    result = runner.run()
    assert result.n_rows == 6  # 3 records x 2 models x 1 mode

    df = pd.read_parquet(tmp_path / "out" / "results.parquet")
    # Every row errored during generation.
    assert (df["generation_error"] != "").all()
    # Errored rows are marked "error" - never "pass".
    assert (df["evaluation_status"] == "error").all()
    assert not (df["evaluation_status"] == "pass").any()
    # No judging happened on empty output: no per-judge columns, zero primary.
    assert "sections_passed" not in df.columns
    assert (df["primary_metric_value"] == 0.0).all()
    assert (df["total_tokens"] == 0).all()

    summary = (tmp_path / "out" / "summary.md").read_text()
    # pass% is 0 and error% is 100 in the model x mode table.
    assert "error%" in summary
    assert "generation errors" in summary

    import json

    doc = json.loads((tmp_path / "out" / "optimizer_results.json").read_text())
    # Manifest summary reflects the errors: none successful, all errored.
    assert doc["summary"]["overall"]["successful"] == 0
    assert doc["summary"]["overall"]["errored"] == 6
    assert doc["summary"]["overall"]["error_rate"] == 100.0
    # Errored rows are embedded with status "error" and no judging.
    assert len(doc["rows"]) == 6
    assert all(r["evaluation_status"] == "error" for r in doc["rows"])
    # per-judge entries judged zero rows (all generation failed).
    assert all(j["n_judged"] == 0 for j in doc["summary"]["per_judge"])


def test_cost_computation_uses_pricing(tmp_path):
    import pandas as pd

    config = OptimizerJobConfig.model_validate(
        {
            "job": {"name": "opt-cost"},
            "optimizer": {
                "sample": {"size": 1, "seed": 1, "stratify_by": "shard"},
                "datasets": [{"name": "memtest"}],
                "prompt_set": "bts_v1",
                "models": [
                    {
                        "name": "priced",
                        "provider": "genos",
                        "price_in": 1.0,
                        "price_out": 2.0,
                        "max_tokens": 2000,
                    }
                ],
                "modes": ["summary_state"],
                "judges": [{"name": "length", "weight": 0}],
                "concurrency": 1,
            },
            "execution": {"type": "local"},
            "output": {"s3_path": "/tmp/ignored"},
        }
    )
    runner = OptimizerRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        gen_client_factory=_stub_factory,
    )
    runner.run()
    df = pd.read_parquet(tmp_path / "out" / "results.parquet")
    row = df.iloc[0]
    expected = (row["prompt_tokens"] / 1000.0) * 1.0 + (
        row["completion_tokens"] / 1000.0
    ) * 2.0
    assert row["cost_usd"] == pytest.approx(round(expected, 6))
