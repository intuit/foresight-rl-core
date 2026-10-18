"""The benchmark catalog: benchmarks as data."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.evaluate_schema import BenchmarkSpec
from foresightrlcore.evaluate import catalog as catalog_mod
from foresightrlcore.evaluate.catalog import (
    BUILTIN_CATALOG,
    load_catalog,
    read_catalog_file,
    register_catalog,
    resolve_benchmark,
)


def _write(path, entries):
    path.write_text(yaml.safe_dump({"benchmarks": entries}), encoding="utf-8")
    return path


@pytest.fixture
def isolated_registry(monkeypatch):
    """Keep catalog registrations from leaking between tests."""
    monkeypatch.setattr(catalog_mod, "_REGISTERED_CATALOGS", [])


def test_the_builtin_catalog_ships_both_tiers():
    catalog = read_catalog_file(BUILTIN_CATALOG)

    assert set(catalog) == {"demo_mcq", "demo_scenarios"}
    assert catalog["demo_mcq"].type == "mcq"
    assert catalog["demo_scenarios"].type == "scenario_suite"
    # Both run offline: a smoke test must not need a file or a network.
    assert all(
        spec.dataset is not None and spec.dataset.loader == "inline"
        for spec in catalog.values()
    )


def test_a_catalog_benchmark_needs_no_other_config(isolated_registry):
    entry = BenchmarkSpec(name="demo_mcq")
    resolved = resolve_benchmark(entry, load_catalog())

    assert resolved.type == "mcq"
    assert resolved.metrics == ["choice_match"]
    assert resolved.dataset is not None


def test_an_entry_overrides_only_the_fields_it_sets(isolated_registry):
    entry = BenchmarkSpec(name="demo_mcq", trials=5)
    resolved = resolve_benchmark(entry, load_catalog())

    assert resolved.trials == 5
    assert resolved.type == "mcq"  # inherited
    assert resolved.metrics == ["choice_match"]  # inherited


def test_a_benchmark_defined_inline_needs_a_type(isolated_registry):
    inline = BenchmarkSpec(name="mine", type="mcq")
    assert resolve_benchmark(inline, load_catalog()).name == "mine"

    with pytest.raises(KeyError, match="not in the catalog"):
        resolve_benchmark(BenchmarkSpec(name="mine"), load_catalog())


def test_an_extension_catalog_is_merged_in(tmp_path, isolated_registry):
    path = _write(tmp_path / "ext.yaml", [{"name": "ext_bench", "type": "mcq"}])
    register_catalog(path)

    assert "ext_bench" in load_catalog()


def test_registering_the_same_catalog_twice_is_a_no_op(tmp_path, isolated_registry):
    path = _write(tmp_path / "ext.yaml", [{"name": "ext_bench", "type": "mcq"}])
    register_catalog(path)
    register_catalog(path)

    assert len(catalog_mod.registered_catalogs()) == 1


def test_a_later_catalog_wins_for_the_same_name(tmp_path, isolated_registry):
    extension = _write(
        tmp_path / "ext.yaml", [{"name": "demo_mcq", "type": "generative"}]
    )
    user = _write(tmp_path / "user.yaml", [{"name": "demo_mcq", "type": "plugin"}])
    register_catalog(extension)

    assert load_catalog([user])["demo_mcq"].type == "plugin"


def test_a_missing_catalog_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_catalog_file(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "body, message",
    [
        ("- just\n- a list\n", "must be a mapping at the top level"),
        ("benchmarks: 3\n", "must be a list"),
        ("benchmarks:\n  - 'a string'\n", "must be a mapping"),
        ("benchmarks:\n  - type: mcq\n", "is invalid"),
    ],
)
def test_a_malformed_catalog_names_the_file_and_the_entry(tmp_path, body, message):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_catalog_file(path)


def test_duplicate_names_within_one_file_are_rejected(tmp_path):
    path = _write(
        tmp_path / "dup.yaml",
        [{"name": "b", "type": "mcq"}, {"name": "b", "type": "mcq"}],
    )

    with pytest.raises(ValueError, match="duplicate benchmark name"):
        read_catalog_file(path)


def test_an_empty_catalog_is_valid(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("benchmarks: []\n", encoding="utf-8")

    assert read_catalog_file(path) == {}
