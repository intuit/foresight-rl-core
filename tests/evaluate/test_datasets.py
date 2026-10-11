"""Dataset loaders: tabular, scenario directories, and inline records."""

from __future__ import annotations

import json

import pytest

from foresightrlcore.evaluate.registry import DATASET_LOADERS


def _loader(name: str):
    return DATASET_LOADERS.get(name)()


# ---------------------------------------------------------------------------
# CSV / JSONL
# ---------------------------------------------------------------------------


def test_csv_reads_numbered_choice_columns(tmp_path):
    path = tmp_path / "mcq.csv"
    path.write_text(
        "prompt,answer,1,2,3,4\n"
        "What is 2+2?,4,3,4,5,6\n"
        "Capital of France?,Paris,Berlin,Paris,Rome,Madrid\n",
        encoding="utf-8",
    )

    records = _loader("csv").records({"path": str(path)})

    assert [r.id for r in records] == ["0", "1"]
    assert records[0].inputs["question"] == "What is 2+2?"
    assert records[0].gold == "4"
    assert records[0].choices == ["3", "4", "5", "6"]


def test_csv_honours_number_of_choices_and_limit(tmp_path):
    path = tmp_path / "mcq.csv"
    path.write_text(
        "prompt,answer,1,2,3,4\na,1,w,x,y,z\nb,2,w,x,y,z\n", encoding="utf-8"
    )

    records = _loader("csv").records(
        {"path": str(path), "number_of_choices": 2, "limit": 1}
    )

    assert len(records) == 1
    assert records[0].choices == ["w", "x"]


def test_csv_uses_explicit_id_and_column_overrides(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("qid,q,a\nq7,Why?,Because\n", encoding="utf-8")

    records = _loader("csv").records(
        {
            "path": str(path),
            "id_column": "qid",
            "question_column": "q",
            "answer_column": "a",
        }
    )

    assert records[0].id == "q7"
    assert records[0].inputs["question"] == "Why?"
    assert records[0].gold == "Because"


def test_csv_without_a_question_column_is_an_error(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="question column"):
        _loader("csv").records({"path": str(path)})


def test_missing_dataset_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader("csv").records({"path": str(tmp_path / "nope.csv")})


def test_jsonl_reads_a_choices_list(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"id": "a", "question": "Q?", "answer": "x", "choices": ["x", "y"]})
        + "\n"
        # A malformed line is skipped rather than failing the whole dataset.
        + "{not json}\n",
        encoding="utf-8",
    )

    records = _loader("jsonl").records({"path": str(path)})

    assert len(records) == 1
    assert records[0].choices == ["x", "y"]


def test_sampling_is_deterministic(tmp_path):
    path = tmp_path / "data.csv"
    rows = "\n".join(f"q{i},a{i}" for i in range(50))
    path.write_text(f"question,answer\n{rows}\n", encoding="utf-8")

    sample = {"size": 10, "seed": 7, "stratify_by": "category"}
    first = _loader("csv").records({"path": str(path)}, sample)
    second = _loader("csv").records({"path": str(path)}, sample)

    assert len(first) == 10
    assert [r.id for r in first] == [r.id for r in second]


# ---------------------------------------------------------------------------
# scenario_dir
# ---------------------------------------------------------------------------


def _scenario(root, name, *, manifest=None, expected=None):
    directory = root / name
    directory.mkdir(parents=True)
    if manifest is not None:
        (directory / "manifest.json").write_text(json.dumps(manifest))
    if expected is not None:
        (directory / "expected.json").write_text(json.dumps(expected))
    return directory


def test_scenario_dir_reads_manifest_and_expected(tmp_path):
    root = tmp_path / "scenarios"
    _scenario(
        root,
        "s1",
        manifest={"category": "billing", "prompt": "Close the books."},
        expected={"status": "closed"},
    )
    _scenario(root, "s2")

    records = _loader("scenario_dir").records({"path": str(root)})

    assert [r.id for r in records] == ["s1", "s2"]
    assert records[0].meta["category"] == "billing"
    assert records[0].inputs["question"] == "Close the books."
    assert records[0].gold == {"status": "closed"}
    # A scenario with no manifest still yields a usable record.
    assert records[1].gold is None
    assert records[1].inputs["scenario_dir"].endswith("s2")


def test_scenario_dir_can_select_a_subset(tmp_path):
    root = tmp_path / "scenarios"
    for name in ("s1", "s2", "s3"):
        _scenario(root, name)

    records = _loader("scenario_dir").records(
        {"path": str(root), "scenarios": ["s1", "s3"]}
    )

    assert [r.id for r in records] == ["s1", "s3"]


def test_scenario_dir_tolerates_broken_manifest(tmp_path):
    root = tmp_path / "scenarios"
    directory = root / "s1"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{broken")

    records = _loader("scenario_dir").records({"path": str(root)})

    assert len(records) == 1
    assert records[0].meta["category"] == "default"


def test_scenario_dir_requires_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader("scenario_dir").records({"path": str(tmp_path / "missing")})


# ---------------------------------------------------------------------------
# inline
# ---------------------------------------------------------------------------


def test_inline_records_carry_extras_into_meta():
    records = _loader("inline").records(
        {
            "records": [
                {
                    "id": "q1",
                    "question": "Q?",
                    "answer": "A",
                    "choices": ["A", "B"],
                    "difficulty": "easy",
                }
            ]
        }
    )

    assert records[0].id == "q1"
    assert records[0].choices == ["A", "B"]
    assert records[0].meta["difficulty"] == "easy"


def test_inline_requires_records():
    with pytest.raises(ValueError, match="requires a 'records' list"):
        _loader("inline").records({})


def test_inline_rejects_a_non_mapping_record():
    with pytest.raises(ValueError, match="must be a mapping"):
        _loader("inline").records({"records": ["just a string"]})


# ---------------------------------------------------------------------------
# Real-world CSV quirks
#
# Both cases below come from production leaderboard files, not from
# imagination: a table sized for its longest question, and a table carrying a
# `choices` column that is a Python repr rather than JSON.
# ---------------------------------------------------------------------------


def test_blank_choice_columns_are_not_rendered_as_options(tmp_path):
    """A table wide enough for 5 options leaves blanks on 3-option rows."""
    path = tmp_path / "ragged.csv"
    path.write_text(
        "prompt,answer,1,2,3,4,5\n"
        "Where?,1,piano,desk,booth,,\n"
        "Which?,0,alpha,beta,gamma,delta,epsilon\n",
        encoding="utf-8",
    )

    records = _loader("csv").records({"path": str(path), "number_of_choices": 5})

    assert records[0].choices == ["piano", "desk", "booth"]
    assert records[1].choices == ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_choice_columns_are_stripped(tmp_path):
    path = tmp_path / "padded.csv"
    path.write_text(
        "prompt,1,2,answer\nTrue or false?, True , False ,0\n", encoding="utf-8"
    )

    records = _loader("csv").records({"path": str(path), "number_of_choices": 2})

    assert records[0].choices == ["True", "False"]


def test_an_unparseable_choices_column_falls_through_to_numbered_columns(tmp_path):
    """A `['a', "b's"]` repr is not JSON, and is not one option either."""
    path = tmp_path / "repr.csv"
    path.write_text(
        "prompt,choices,answer,1,2,3\n"
        '"Where?","[\'piano\', ""desk""]",1,piano,desk,booth\n',
        encoding="utf-8",
    )

    records = _loader("csv").records({"path": str(path), "number_of_choices": 3})

    assert records[0].choices == ["piano", "desk", "booth"]


def test_a_pipe_separated_choices_column_is_still_honoured(tmp_path):
    path = tmp_path / "piped.csv"
    path.write_text(
        "prompt,choices,answer\nWhere?,piano|desk|booth,1\n", encoding="utf-8"
    )

    records = _loader("csv").records({"path": str(path)})

    assert records[0].choices == ["piano", "desk", "booth"]


# ---------------------------------------------------------------------------
# Question-column selection
# ---------------------------------------------------------------------------


def _passage_table(path):
    """A table shaped like LogiQA2/MuSR/FinQA: bare question *and* full prompt."""
    path.write_text(
        "id,question,prompt,answer,1,2\n"
        '1,"Which follows?","Passage: all frogs are green. '
        'Question: Which follows?",1,"Frogs are green","Frogs are blue"\n',
        encoding="utf-8",
    )
    return {"path": str(path), "number_of_choices": 2}


def test_a_question_column_that_drops_the_passage_is_warned_about(tmp_path, caplog):
    """The question alone is unanswerable, and the loader must say so.

    Column precedence prefers `question` over `prompt`, so a table carrying
    both silently asks the model to reason about a passage it never sees. On
    the real LogiQA2 table that produced a below-chance score and 57% of
    responses replying that the passage was missing.
    """
    spec = _passage_table(tmp_path / "passage.csv")

    with caplog.at_level("WARNING"):
        records = _loader("csv").records(spec)

    assert records[0].inputs["question"] == "Which follows?"
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "'question'" in m and "'prompt'" in m for m in messages
    ), f"expected a truncation warning, got {messages}"


def test_an_explicit_question_column_is_not_second_guessed(tmp_path, caplog):
    """Naming the column is the fix, so it must not also nag about it."""
    spec = _passage_table(tmp_path / "passage.csv")
    spec["question_column"] = "prompt"

    with caplog.at_level("WARNING"):
        records = _loader("csv").records(spec)

    assert records[0].inputs["question"].startswith("Passage: all frogs are green.")
    assert not [r for r in caplog.records if "question_column" in r.getMessage()]


def test_columns_of_similar_length_are_not_flagged(tmp_path):
    """Two phrasings of the same question are not a truncation."""
    path = tmp_path / "similar.csv"
    path.write_text(
        "question,prompt,answer,1,2\n"
        "Which follows here?,Which one follows here?,1,Yes,No\n",
        encoding="utf-8",
    )

    records = _loader("csv").records({"path": str(path), "number_of_choices": 2})

    assert records[0].inputs["question"] == "Which follows here?"
