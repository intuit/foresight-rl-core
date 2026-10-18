"""Tests for foresightrlcore.config.recipes — container and local-GPU recipes."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.recipes import (
    _FALLBACK_FRAMEWORK_VERSION,
    CONTAINER_RECIPES,
    FLASH_ATTN_WHEELS,
    GPU_FAMILIES,
    LOCAL_GPU_RECIPES,
    generate_local_gpu_requirements,
    resolve_flash_attn_wheel,
    resolve_gpu_family,
)
from foresightrlcore.config.schema import ContainerConfig, TrainingJobConfig


class TestRecipeRegistry:
    """Verify recipe definitions."""

    def test_known_recipes_exist(self):
        expected = {"qwen3", "qwen3_5", "verl_sft"}
        assert set(CONTAINER_RECIPES.keys()) == expected

    def test_all_recipes_have_required_keys(self):
        required = {"framework_version", "framework", "requirements"}
        for name, recipe in CONTAINER_RECIPES.items():
            missing = required - set(recipe.keys())
            assert not missing, f"Recipe '{name}' missing keys: {missing}"

    def test_all_recipes_have_nonempty_requirements(self):
        for name, recipe in CONTAINER_RECIPES.items():
            assert recipe["requirements"], f"Recipe '{name}' has empty requirements"

    def test_qwen3_framework_version(self):
        assert CONTAINER_RECIPES["qwen3"]["framework_version"] == "2.8.0"

    def test_qwen3_5_framework_version(self):
        assert CONTAINER_RECIPES["qwen3_5"]["framework_version"] == "2.9.0"

    def test_qwen3_has_torchrun_requirements(self):
        assert CONTAINER_RECIPES["qwen3"]["requirements_torchrun"] is not None
        assert len(CONTAINER_RECIPES["qwen3"]["requirements_torchrun"]) > 0

    def test_qwen3_5_torchrun_is_none(self):
        assert CONTAINER_RECIPES["qwen3_5"]["requirements_torchrun"] is None

    def test_verl_sft_is_torchrun_only(self):
        """verl_sft (method.type: sft_verl) never starts vLLM, so the
        vllm-mode and torchrun-mode requirements are intentionally identical."""
        recipe = CONTAINER_RECIPES["verl_sft"]
        assert recipe["requirements_torchrun"] is not None
        assert recipe["requirements_torchrun"] == recipe["requirements"]
        assert any(r.startswith("verl") for r in recipe["requirements"])


class TestContainerConfigRecipeResolution:
    """ContainerConfig model_validator applies recipe defaults."""

    def test_recipe_sets_framework_version(self):
        c = ContainerConfig(recipe="qwen3")
        assert c.framework_version == "2.8.0"

    def test_recipe_sets_resolved_requirements(self):
        c = ContainerConfig(recipe="qwen3")
        assert c.resolved_requirements is not None
        assert "trl[vllm]==0.28.0" in c.resolved_requirements

    def test_recipe_qwen3_5_resolved(self):
        c = ContainerConfig(recipe="qwen3_5")
        assert c.framework_version == "2.9.0"
        assert c.resolved_requirements is not None
        assert "trl==1.6.0" in c.resolved_requirements

    def test_explicit_framework_version_overrides_recipe(self):
        c = ContainerConfig(recipe="qwen3", framework_version="2.7.0")
        assert c.framework_version == "2.7.0"
        assert c.resolved_requirements is not None

    def test_inline_requirements_override_recipe(self):
        custom_reqs = ["trl>=2.0.0", "transformers>=6.0.0"]
        c = ContainerConfig(recipe="qwen3", requirements=custom_reqs)
        assert c.requirements == custom_reqs
        assert c.resolved_requirements is None

    def test_no_recipe_fallback_framework_version(self):
        c = ContainerConfig()
        assert c.framework_version == _FALLBACK_FRAMEWORK_VERSION

    def test_explicit_framework_version_without_recipe(self):
        c = ContainerConfig(framework_version="2.7.0")
        assert c.framework_version == "2.7.0"
        assert c.resolved_requirements is None

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError, match="Unknown container recipe"):
            ContainerConfig(recipe="nonexistent")

    def test_recipe_sets_torchrun_requirements(self):
        c = ContainerConfig(recipe="qwen3")
        assert c.resolved_requirements_torchrun is not None
        assert "trl==0.29.1" in c.resolved_requirements_torchrun

    def test_recipe_qwen3_5_no_torchrun(self):
        c = ContainerConfig(recipe="qwen3_5")
        assert c.resolved_requirements_torchrun is None

    def test_recipe_verl_sft_resolved(self):
        c = ContainerConfig(recipe="verl_sft")
        assert c.framework_version == "2.9.0"
        assert c.resolved_requirements is not None
        assert c.resolved_requirements_torchrun == c.resolved_requirements

    def test_resolved_fields_excluded_from_serialization(self):
        c = ContainerConfig(recipe="qwen3")
        dumped = c.model_dump()
        assert "resolved_requirements" not in dumped
        assert "resolved_requirements_torchrun" not in dumped


class TestRecipeViaYaml:
    """Recipe resolution through full YAML config loading."""

    def test_recipe_in_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "Qwen/Qwen3-14B"},
                "container": {"recipe": "qwen3"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.container.framework_version == "2.8.0"
        assert config.container.resolved_requirements is not None

    def test_recipe_qwen3_5_in_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "Qwen/Qwen3.5-27B"},
                "container": {"recipe": "qwen3_5"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.container.framework_version == "2.9.0"

    def test_invalid_recipe_in_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {"recipe": "nonexistent"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        with pytest.raises(Exception, match="Unknown container recipe"):
            TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_no_recipe_backward_compat(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {"framework_version": "2.8.0"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.container.framework_version == "2.8.0"
        assert config.container.recipe is None
        assert config.container.resolved_requirements is None

    def test_inline_requirements_in_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {
                    "framework_version": "2.9.0",
                    "requirements": ["trl>=1.3.0", "vllm>=0.20.0"],
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.container.requirements == ["trl>=1.3.0", "vllm>=0.20.0"]
        assert config.container.resolved_requirements is None

    def test_recipe_roundtrip_excludes_resolved(self, tmp_yaml, tmp_path):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {"recipe": "qwen3_5"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        out = tmp_path / "out.yaml"
        config.to_yaml(out)

        with open(out) as f:
            raw = yaml.safe_load(f)
        assert "resolved_requirements" not in raw.get("container", {})
        assert "resolved_requirements_torchrun" not in raw.get("container", {})
        assert raw["container"]["recipe"] == "qwen3_5"


# ===================================================================
# Local-GPU recipe tests
# ===================================================================


class TestLocalGpuRecipeRegistry:
    """Verify LOCAL_GPU_RECIPES definitions."""

    def test_known_recipes_exist(self):
        expected = {"qwen3", "qwen3_5"}
        assert set(LOCAL_GPU_RECIPES.keys()) == expected

    def test_all_recipes_have_requirements(self):
        for name, recipe in LOCAL_GPU_RECIPES.items():
            assert "requirements" in recipe, f"Recipe '{name}' missing 'requirements'"
            assert recipe["requirements"], f"Recipe '{name}' has empty requirements"

    def test_qwen3_has_trl_vllm(self):
        reqs = LOCAL_GPU_RECIPES["qwen3"]["requirements"]
        assert any("trl[vllm]" in r for r in reqs)

    def test_qwen3_5_has_separate_vllm(self):
        reqs = LOCAL_GPU_RECIPES["qwen3_5"]["requirements"]
        assert any("vllm" in r and "trl" not in r for r in reqs)

    def test_no_overlap_with_container_recipes(self):
        """LOCAL_GPU_RECIPES is a separate dict, not sharing objects."""
        for name in LOCAL_GPU_RECIPES:
            if name in CONTAINER_RECIPES:
                assert (
                    LOCAL_GPU_RECIPES[name] is not CONTAINER_RECIPES[name]
                ), f"Recipe '{name}' shares object between registries"


class TestGenerateLocalGpuRequirements:
    """Tests for generate_local_gpu_requirements()."""

    def test_writes_file(self, tmp_path):
        out = tmp_path / "reqs.txt"
        result = generate_local_gpu_requirements("qwen3_5", str(out))
        assert result == out
        assert out.exists()

    def test_file_contains_all_requirements(self, tmp_path):
        out = tmp_path / "reqs.txt"
        generate_local_gpu_requirements("qwen3_5", str(out))
        content = out.read_text()
        for req in LOCAL_GPU_RECIPES["qwen3_5"]["requirements"]:
            assert req in content

    def test_file_has_header_comment(self, tmp_path):
        out = tmp_path / "reqs.txt"
        generate_local_gpu_requirements("qwen3", str(out))
        first_line = out.read_text().splitlines()[0]
        assert first_line.startswith("#")
        assert "qwen3" in first_line

    def test_unknown_recipe_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown local-GPU recipe"):
            generate_local_gpu_requirements("nonexistent", str(tmp_path / "x.txt"))

    def test_default_output_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = generate_local_gpu_requirements("qwen3")
        assert result.name == "local-requirements.txt"
        assert result.exists()


class TestResolveGpuFamily:
    """Tests for resolve_gpu_family() / resolve_flash_attn_wheel()."""

    @pytest.mark.parametrize(
        "instance_type,expected_family",
        [
            ("ml.p4d.24xlarge", "ampere"),
            ("ml.p4de.24xlarge", "ampere"),
            ("ml.p5.48xlarge", "hopper"),
            ("ml.p5e.48xlarge", "hopper"),
            ("ml.p5en.48xlarge", "hopper"),
            ("ml.p6.48xlarge", "blackwell"),
            ("ml.p6e.48xlarge", "blackwell"),
            # Real SageMaker/EC2 family tokens are hyphenated, not bare "p6"
            # — confirmed in docs/checking-spot-capacity-scores.md.
            ("ml.p6-b200.48xlarge", "blackwell"),
            ("ml.p6-b300.48xlarge", "blackwell"),
        ],
    )
    def test_known_instance_types_resolve_to_expected_family(
        self, instance_type, expected_family
    ):
        assert resolve_gpu_family(instance_type)["name"] == expected_family

    def test_p6_b200_resolves_correctly_despite_hyphen_in_family_token(self):
        """The real instance type name — not the bracketed shorthand "p6[-b200]"

        used in this module's own comments — must resolve without being
        mistaken for the unrelated bare "p6" entry.
        """
        assert resolve_gpu_family("ml.p6-b200.48xlarge")["gpu"] == "B200"
        assert resolve_gpu_family("ml.p6-b300.48xlarge")["gpu"] == "B300"

    def test_p5_does_not_swallow_p5en_or_p5e(self):
        """Exact family-token matching, not prefix substring matching —
        'p5' must not accidentally match instance types for 'p5e'/'p5en'.
        """
        assert resolve_gpu_family("ml.p5.48xlarge")["gpu"] == "H100"
        assert resolve_gpu_family("ml.p5en.48xlarge")["gpu"] == "H200"

    def test_unknown_instance_type_raises(self):
        with pytest.raises(ValueError, match="Unrecognized GPU family"):
            resolve_gpu_family("ml.g5.12xlarge")

    def test_every_gpu_family_has_a_flash_attn_wheel(self):
        family_names = {meta["name"] for meta in GPU_FAMILIES.values()}
        assert family_names == set(FLASH_ATTN_WHEELS.keys())

    def test_resolve_flash_attn_wheel_returns_a_url(self):
        wheel = resolve_flash_attn_wheel("ml.p4de.24xlarge")
        assert wheel.startswith("https://")
        assert wheel.endswith(".whl")

    def test_resolve_flash_attn_wheel_unknown_instance_raises(self):
        with pytest.raises(ValueError, match="Unrecognized GPU family"):
            resolve_flash_attn_wheel("ml.g5.12xlarge")
