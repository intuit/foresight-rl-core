"""Small-module coverage: registry, prompts, single_call mode, mode helpers,
spec row keys, interfaces token math, and the stub client template branch."""

from __future__ import annotations

import pytest

from foresightrlcore.generation.stub_client import StubGenerationClient
from foresightrlcore.optimizer.interfaces import CallUsage, GenResult, Record
from foresightrlcore.optimizer.modes.base import cap_tokens, strip_advisory_sections
from foresightrlcore.optimizer.modes.single_call import SingleCallMode
from foresightrlcore.optimizer.prompts import YamlPromptSet, make_prompt_set
from foresightrlcore.optimizer.registry import Registry
from foresightrlcore.optimizer.spec import RowResult


class TestRegistry:
    def test_register_and_get(self):
        reg: Registry = Registry("thing")
        reg.register("a")(lambda: 1)
        assert reg.get("a")() == 1
        assert "a" in reg
        assert len(reg) == 1
        assert reg.list() == ["a"]

    def test_empty_name_rejected(self):
        reg: Registry = Registry("thing")
        with pytest.raises(ValueError, match="non-empty string"):
            reg.register("")

    def test_duplicate_registration_rejected(self):
        reg: Registry = Registry("thing")
        reg.register("a")(lambda: 1)
        with pytest.raises(ValueError, match="duplicate"):
            reg.register("a")(lambda: 2)

    def test_get_unknown_raises(self):
        reg: Registry = Registry("thing")
        with pytest.raises(KeyError, match="unknown thing"):
            reg.get("missing")


class TestYamlPromptSet:
    def test_template_and_render(self, tmp_path):
        p = tmp_path / "prompts.yaml"
        p.write_text("single:\n  prompt: 'Hello {name}, brace {{literal}}'\n")
        ps = YamlPromptSet(p)
        assert ps.template("single", "prompt").startswith("Hello {name}")
        rendered = ps.render("single", "prompt", name="World")
        assert "Hello World" in rendered
        assert "{{literal}}" in rendered  # untouched

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="prompts file not found"):
            YamlPromptSet(tmp_path / "nope.yaml").template("s", "k")

    def test_non_mapping_file_raises(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="not a mapping"):
            YamlPromptSet(p).template("s", "k")

    def test_missing_section_raises(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("single:\n  prompt: hi\n")
        with pytest.raises(KeyError, match="section"):
            YamlPromptSet(p).template("absent", "prompt")

    def test_missing_key_raises(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("single:\n  prompt: hi\n")
        with pytest.raises(KeyError, match="not found"):
            YamlPromptSet(p).template("single", "absent")

    def test_render_none_value_becomes_empty(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("single:\n  prompt: 'x={val}'\n")
        assert YamlPromptSet(p).render("single", "prompt", val=None) == "x="

    def test_make_prompt_set_path_override(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("single:\n  prompt: custom\n")
        ps = make_prompt_set("bts_v1", prompts_path=str(p))
        assert ps.template("single", "prompt") == "custom"


class _FakePromptSet:
    def render(self, section, key, **variables):
        return f"{section}.{key}: {variables.get('evidence_bundle', '')}"


class TestSingleCallMode:
    def test_generate_makes_one_call(self):
        mode = SingleCallMode()
        record = Record(
            realm_id="r1",
            epoch_date="2024",
            inputs={"evidence_bundle": "rev 100"},
            reference="",
        )
        client = StubGenerationClient(model_name="m")
        result = mode.generate(record, client, _FakePromptSet(), max_tokens=100)
        assert isinstance(result, GenResult)
        assert result.text
        assert len(result.calls) == 1


class TestModeHelpers:
    def test_strip_advisory_removes_heading_and_after(self):
        text = "# Overview\nfacts\n## Recommendations\ndo x\ndo y"
        out = strip_advisory_sections(text)
        assert "Recommendations" not in out
        assert "Overview" in out

    def test_strip_advisory_noop_without_heading(self):
        text = "# Overview\njust facts"
        assert strip_advisory_sections(text) == text

    def test_cap_tokens_clamps(self):
        assert cap_tokens(16000, 4096) == 4096

    def test_cap_tokens_zero_max_returns_requested(self):
        assert cap_tokens(16000, 0) == 16000


class TestStubTemplate:
    def test_template_branch(self):
        client = StubGenerationClient(model_name="m", template="{model}|{tag}|{digest}")
        text, usage = client.generate("prompt", tag="t", max_tokens=10)
        assert text.startswith("m|t|")
        assert usage.tag == "t"


class TestSpecAndInterfaces:
    def test_row_result_resume_key(self):
        row = RowResult(
            dataset="ds",
            realm_id="r1",
            epoch_date="2024-01-01",
            model="m",
            mode="summary_state",
        )
        assert row.resume_key() == "ds|r1|2024-01-01|m|summary_state"

    def test_call_usage_total_tokens(self):
        usage = CallUsage(tag="t", prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15
