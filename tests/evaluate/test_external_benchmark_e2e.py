"""End-to-end proof: a benchmark from another repo runs unmodified.

This is the claim the whole operation is built on, so it is asserted rather
than described. The benchmark below is written to a temp directory as a
standalone script. It imports nothing from ``foresightrlcore``, knows no
provider, and reads only the documented endpoint contract -- exactly the
position a third-party repository is in.

Everything downstream of the config is the real thing: the real operation, the
real runner, the real proxy over real HTTP, the real subprocess, the real
reporters. Only the model behind the endpoint is synthetic (``provider: echo``),
because a real one would make this a paid network test rather than a proof that
the plumbing holds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.config.evaluate_validation import validate_evaluate_config
from foresightrlcore.operations.evaluate import EvaluateOperation

# A benchmark that talks OpenAI over HTTP and grades its own answers. Written
# the way an outside project would write it: argparse, urllib, a JSON result.
EXTERNAL_BENCHMARK = '''\
"""A benchmark that has never heard of foresightrlcore."""

import argparse
import csv
import json
import os
import urllib.request


def ask(base_url, api_key, model, question):
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(
            {"model": model, "messages": [{"role": "user", "content": question}]}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("EVALUATE_MODEL_ID"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=os.environ.get("EVALUATE_OUTPUT_FILE"))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    with open(args.data) as handle:
        rows = list(csv.DictReader(handle))

    passed, by_topic = 0, {}
    for row in rows:
        answer = ask(args.endpoint, api_key, args.model, row["question"])
        ok = row["expects"] in answer
        passed += ok
        by_topic.setdefault(row["topic"], []).append(ok)

    # Proof the workspace is writable and ours: an artifact only this trial
    # could have produced.
    with open(os.path.join(os.environ["EVALUATE_WORKSPACE"], "trace.txt"), "w") as f:
        f.write(f"{len(rows)} questions against {args.model}\\n")

    with open(args.out, "w") as handle:
        json.dump(
            {
                "score": passed / len(rows),
                "n_samples": len(rows),
                "sub_scores": {
                    topic: sum(v) / len(v) for topic, v in by_topic.items()
                },
            },
            handle,
        )


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def external_repo(tmp_path: Path) -> Path:
    """A miniature benchmark repository, complete with its own data."""
    repo = tmp_path / "someone-elses-benchmark"
    repo.mkdir()
    (repo / "run_bench.py").write_text(EXTERNAL_BENCHMARK)

    with (repo / "questions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "expects", "topic"])
        # The echo backend replies "[echo:<model>] <prompt>", so a question
        # containing its own expected substring passes and one that does not
        # fails. That gives a score strictly between 0 and 1 -- enough to prove
        # the number came from the benchmark and was not defaulted.
        writer.writerow(["say alpha", "alpha", "greek"])
        writer.writerow(["say beta", "beta", "greek"])
        writer.writerow(["say gamma", "not-in-the-echo", "latin"])
    return repo


def _config(repo: Path, output_dir: Path, **run_overrides) -> EvaluateJobConfig:
    run = {
        "benchmarks": [
            {
                "name": "someone_elses_benchmark",
                "type": "external_process",
                "source": {"path": str(repo)},
                "command": [
                    "python",
                    "run_bench.py",
                    "--data=questions.csv",
                    "--endpoint=$OPENAI_BASE_URL",
                    "--model=$EVALUATE_MODEL_ID",
                    "--out=$EVALUATE_OUTPUT_FILE",
                ],
                "timeout_seconds": 120,
            }
        ],
        "models": [{"name": "echo-model", "provider": "echo"}],
        "reporters": ["table", "json"],
    }
    run.update(run_overrides)
    return EvaluateJobConfig.model_validate(
        {
            "job": {"name": "external-proof"},
            "execution": {"type": "local", "local": {"output_dir": str(output_dir)}},
            "evaluate": run,
        }
    )


def test_an_unmodified_external_benchmark_runs_and_reports(tmp_path, external_repo):
    output_dir = tmp_path / "out"
    config = _config(external_repo, output_dir)

    assert validate_evaluate_config(config) == []
    assert EvaluateOperation().run(config) == ["external-proof"]

    payload = json.loads((output_dir / "evaluate_results.json").read_text())
    row = payload["results"][0]

    assert row["benchmark"] == "someone_elses_benchmark"
    assert row["model"] == "echo-model"
    # 2 of 3 -- computed by the benchmark, carried back through the contract.
    assert row["score"] == pytest.approx(2 / 3)
    assert row["n_samples"] == 3


def test_the_benchmarks_own_breakdown_survives_the_round_trip(tmp_path, external_repo):
    output_dir = tmp_path / "out"
    EvaluateOperation().run(_config(external_repo, output_dir))

    payload = json.loads((output_dir / "evaluate_results.json").read_text())
    sub_scores = payload["aggregates"][0]["sub_scores"]

    assert sub_scores["greek"] == 1.0
    assert sub_scores["latin"] == 0.0


def test_the_leaderboard_and_summary_are_written(tmp_path, external_repo):
    output_dir = tmp_path / "out"
    EvaluateOperation().run(_config(external_repo, output_dir))

    rows = list(csv.reader((output_dir / "benchmark_scores.csv").open()))
    assert rows[0] == ["benchmark", "echo-model"]
    assert rows[1][0] == "someone_elses_benchmark"

    summary = (output_dir / "summary.md").read_text()
    assert "someone_elses_benchmark" in summary
    assert "0.6667" in summary


def test_repeated_trials_produce_an_interval(tmp_path, external_repo):
    output_dir = tmp_path / "out"

    EvaluateOperation().run(_config(external_repo, output_dir, trials=3))

    payload = json.loads((output_dir / "evaluate_results.json").read_text())
    aggregate = payload["aggregates"][0]

    assert aggregate["trials"] == 3
    assert aggregate["scores"] == [pytest.approx(2 / 3)] * 3
    # Deterministic benchmark: the interval collapses onto the mean.
    assert aggregate["ci_low"] == pytest.approx(aggregate["ci_high"])


def test_a_second_run_resumes_instead_of_re_paying(tmp_path, external_repo):
    output_dir = tmp_path / "out"
    checkpoint_dir = tmp_path / "ckpt"
    config = _config(external_repo, output_dir)
    config.execution.local = {
        **(config.execution.local or {}),
        "checkpoint_dir": str(checkpoint_dir),
    }

    EvaluateOperation().run(config)
    first = json.loads((output_dir / "evaluate_results.json").read_text())

    checkpoint = checkpoint_dir / "evaluate_checkpoint.jsonl"
    assert len(checkpoint.read_text().splitlines()) == 1

    EvaluateOperation().run(config)
    second = json.loads((output_dir / "evaluate_results.json").read_text())

    # The score is carried forward from the checkpoint; the benchmark process
    # was never started again, so no second trial row was appended.
    assert second["results"][0]["score"] == first["results"][0]["score"]
    assert len(checkpoint.read_text().splitlines()) == 1


def test_a_benchmark_that_fails_is_reported_as_unmeasured_not_as_zero(
    tmp_path, external_repo
):
    """A failed benchmark scoring 0.0 would silently corrupt a leaderboard."""
    output_dir = tmp_path / "out"
    config = _config(external_repo, output_dir)
    config.evaluate.benchmarks[0].command = [
        "python",
        "run_bench.py",
        "--data=gone.csv",
    ]

    EvaluateOperation().run(config)

    payload = json.loads((output_dir / "evaluate_results.json").read_text())
    assert payload["results"][0]["score"] == -1.0
    assert payload["aggregates"][0]["valid"] is False

    summary = (output_dir / "summary.md").read_text()
    assert "## Failures" in summary
    assert "n/a" in summary
