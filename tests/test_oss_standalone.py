"""Standalone OSS smoke test — the core works with nothing Intuit around it.

Every other test module may assume the monorepo layout. This one deliberately
does not. It is the executable version of the question "if we delete
``intuit/`` and publish the wheel, does anything still work?", and it must hold
in all three worlds the package ships into:

* the monorepo with the extension installed (normal dev),
* the ``oss`` tox env, which keeps ``intuit/`` off ``sys.path``,
* a published wheel with no ``intuit/`` directory in existence.

To stay honest about that, **nothing here skips**. There are no
``requires_intuit`` marks and no ``importorskip`` guards; if a test in this
module cannot run somewhere, the OSS story is broken and it should fail loudly.

Preconditions are deliberately nil: no network, no AWS credentials, no S3
objects, no dataset files, no prompt YAMLs, no GPU. Anything requiring those is
out of scope by design — this asserts the package *boots, configures and
validates* standalone, not that it can complete a training run.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Iterator

import pytest
import yaml

import foresightrlcore
from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.config.evaluate_validation import validate_evaluate_config
from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
from foresightrlcore.config.optimizer_validation import validate_optimizer_config
from foresightrlcore.config.perf_schema import PerfTestJobConfig
from foresightrlcore.config.perf_validation import validate_perf_config
from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.config.validation import validate_config

#: The OSS task used throughout. It needs no credentials and no task config,
#: which is exactly why it is the right one for a zero-precondition example.
OSS_TASK = "email_outcome"

CORE_DIR = Path(foresightrlcore.__file__).resolve().parent


# ---------------------------------------------------------------------------
# Import + CLI surface
# ---------------------------------------------------------------------------


class TestCoreImportsStandalone:
    """The package and its CLI come up without the extension present."""

    def test_package_imports(self):
        assert foresightrlcore.__name__ == "foresightrlcore"

    def test_cli_main_is_callable(self):
        from foresightrlcore.cli import main

        assert callable(main)

    def test_cli_reports_usage_without_a_command(self, capsys):
        """``rl-train`` with no args prints help and exits cleanly."""
        from foresightrlcore.cli import main

        assert main([]) == 0
        assert "usage" in capsys.readouterr().out.lower()


class TestPluginLoadingWithoutExtension:
    """Plugin discovery tolerates having nothing to discover."""

    def test_load_plugins_succeeds_and_is_idempotent(self):
        from foresightrlcore.plugins import load_plugins

        load_plugins()
        load_plugins()

    def test_load_optimizer_plugins_succeeds_and_is_idempotent(self):
        from foresightrlcore.plugins import load_optimizer_plugins

        load_optimizer_plugins()
        load_optimizer_plugins()

    def test_load_evaluate_plugins_succeeds_and_is_idempotent(self):
        from foresightrlcore.plugins import load_evaluate_plugins

        load_evaluate_plugins()
        load_evaluate_plugins()

    def test_no_extra_packages_env_is_not_an_error(self, monkeypatch):
        """A container with no staged extensions must still boot."""
        from foresightrlcore.plugins import EXTRA_PACKAGES_ENV, load_extra_packages

        monkeypatch.delenv(EXTRA_PACKAGES_ENV, raising=False)
        load_extra_packages()


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


class TestOssRegistriesAreSelfSufficient:
    """The OSS core ships usable built-ins on every pluggable axis.

    A registry that is empty without the extension would mean the axis is
    Intuit-only in practice, whatever the plugin architecture claims.
    """

    def _registries(self):
        from foresightrlcore.optimizer.registry import (
            DATASETS,
            JUDGES,
            MODES,
            PROMPT_SETS,
        )

        return {
            "judges": JUDGES,
            "datasets": DATASETS,
            "modes": MODES,
            "prompt_sets": PROMPT_SETS,
        }

    @pytest.mark.parametrize(
        "axis,expected",
        [
            ("judges", {"correctness", "hallucination"}),
            ("datasets", {"parquet"}),
            ("modes", {"single_call"}),
            ("prompt_sets", {"single_v1"}),
        ],
    )
    def test_axis_ships_oss_builtins(self, axis, expected):
        from foresightrlcore.plugins import load_optimizer_plugins

        load_optimizer_plugins()
        assert expected <= set(self._registries()[axis].list())

    def test_every_registered_entry_resolves(self):
        """No dangling names: everything listed can actually be fetched."""
        from foresightrlcore.plugins import load_optimizer_plugins

        load_optimizer_plugins()
        for name, registry in self._registries().items():
            for entry in registry.list():
                assert registry.get(entry) is not None, f"{name}:{entry} is dangling"

    def test_builtin_prompt_set_ships_inside_the_package(self):
        """Its YAML must travel with the wheel, not sit under examples/.

        Anything outside the package is absent from a staged job source dir,
        which is why the extension's set requires an S3 ``prompts_path``.
        """
        from foresightrlcore.optimizer.prompt_sets import SINGLE_V1_PATH

        assert SINGLE_V1_PATH.is_file()
        assert CORE_DIR in SINGLE_V1_PATH.parents

    def test_builtin_prompt_set_renders_the_single_call_contract(self):
        """``single_call`` renders section 'single', key 'prompt'."""
        from foresightrlcore.optimizer.prompts import make_prompt_set

        prompt_set = make_prompt_set("single_v1")
        rendered = prompt_set.render("single", "prompt", input="What is 2 + 2?")
        assert "What is 2 + 2?" in rendered
        assert "{" not in rendered, f"unsubstituted placeholder left in: {rendered!r}"


class TestOssEvaluateIsSelfSufficient:
    """Evaluate's four axes and its catalog ship usable OSS built-ins.

    The evaluate operation exists to run benchmarks that live *outside* this
    repo, so the risk is the opposite of the optimizer's: not that the core is
    empty, but that the parts making an external benchmark runnable turn out to
    be Intuit-only. These assertions pin the generic ones.
    """

    def _registries(self):
        from foresightrlcore.evaluate.registry import (
            BENCHMARK_TYPES,
            DATASET_LOADERS,
            METRICS,
            REPORTERS,
        )

        return {
            "benchmark_types": BENCHMARK_TYPES,
            "dataset_loaders": DATASET_LOADERS,
            "metrics": METRICS,
            "reporters": REPORTERS,
        }

    @pytest.mark.parametrize(
        "axis,expected",
        [
            # Both tiers: record-driven types and harness-driven ones. Without
            # external_process/plugin, a third-party benchmark has no way in.
            (
                "benchmark_types",
                {"mcq", "generative", "external_process", "plugin", "scenario_suite"},
            ),
            ("dataset_loaders", {"csv", "jsonl", "parquet", "scenario_dir", "inline"}),
            ("metrics", {"exact_match", "substring_match", "choice_match"}),
            ("reporters", {"table", "json"}),
        ],
    )
    def test_axis_ships_oss_builtins(self, axis, expected):
        from foresightrlcore.plugins import load_evaluate_plugins

        load_evaluate_plugins()
        assert expected <= set(self._registries()[axis].list())

    def test_every_registered_entry_resolves(self):
        from foresightrlcore.plugins import load_evaluate_plugins

        load_evaluate_plugins()
        for name, registry in self._registries().items():
            for entry in registry.list():
                assert registry.get(entry) is not None, f"{name}:{entry} is dangling"

    def test_the_builtin_catalog_ships_inside_the_package(self):
        """Like the prompt set, the catalog must travel with the wheel."""
        from foresightrlcore.evaluate.catalog import BUILTIN_CATALOG

        assert BUILTIN_CATALOG.is_file()
        assert CORE_DIR in BUILTIN_CATALOG.parents

    def test_the_builtin_catalog_runs_with_no_preconditions(self):
        """A first run must need no S3 object, no dataset file, no credentials.

        Reads the built-in file directly rather than the merged catalog: an
        installed extension is entitled to contribute S3-backed benchmarks, and
        the guarantee under test is about what ships in the core.
        """
        from foresightrlcore.evaluate.catalog import BUILTIN_CATALOG, read_catalog_file

        catalog = read_catalog_file(BUILTIN_CATALOG)
        assert catalog, "the OSS catalog is empty: there is nothing to demo"
        for name, spec in catalog.items():
            assert spec.dataset is not None, f"{name} has no dataset"
            assert spec.dataset.loader == "inline", (
                f"catalog benchmark {name!r} reads {spec.dataset.loader!r} data; "
                "the built-in catalog must be runnable offline"
            )

    def test_an_endpoint_can_be_served_without_any_provider_credentials(self):
        """The endpoint contract is what external benchmarks bind to."""
        from foresightrlcore.inference.backends import EchoChatBackend
        from foresightrlcore.inference.client import ChatClient
        from foresightrlcore.inference.proxy import OpenAICompatProxy

        with OpenAICompatProxy(EchoChatBackend("m"), model_id="m") as endpoint:
            result = ChatClient(endpoint).chat([{"role": "user", "content": "ping"}])

        assert result.text == "[echo:m] ping"
        assert endpoint.env()["OPENAI_BASE_URL"] == endpoint.base_url


class TestOssTaskResolution:
    """The built-in task loads; unknown ones fail with an actionable message."""

    def test_oss_task_loads(self):
        from foresightrlcore.tasks import load_task_module

        module = load_task_module(OSS_TASK)
        for attr in ("get_reward_function", "get_callbacks", "get_dataset_loader"):
            assert hasattr(module, attr), f"{OSS_TASK} is missing {attr}"

    def test_unknown_task_raises_value_error_not_import_error(self):
        """A missing task is a config problem, so it must not surface as a crash."""
        from foresightrlcore.tasks import load_task_module

        with pytest.raises(ValueError, match="Unknown task type"):
            load_task_module("no_such_task_anywhere")


# ---------------------------------------------------------------------------
# Configs that use only OSS constructs
# ---------------------------------------------------------------------------


def _training_config_dict() -> dict:
    """A complete training config referencing nothing Intuit.

    OSS task, Bedrock judge (IAM-role auth, so no ``secrets:`` block at all),
    and no ``container.extra_packages``.
    """
    return {
        "job": {"name": "oss-standalone-smoke"},
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "method": {"type": "grpo"},
        "training": {
            "epochs": 1,
            "per_device_batch_size": 1,
            "grpo": {"num_generations": 2, "max_completion_length": 256},
        },
        "data": {"train_path": "s3://example-bucket/train.jsonl"},
        "output": {"s3_path": "s3://example-bucket/output/"},
        "execution": {"type": "aws_training_job"},
        "container": {"recipe": "qwen3_5"},
        "task": {
            "type": OSS_TASK,
            "config": {
                "llm_clients": {
                    "default": {
                        "provider": "bedrock",
                        "bedrock": {
                            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                            "region": "us-west-2",
                        },
                    }
                }
            },
        },
    }


def _sft_verl_training_config_dict() -> dict:
    """verl's FSDP SFT trainer (method.type: sft_verl) needs no task/

    secrets/LLM-judge block at all — unlike GRPO, this is the entire
    config surface. Uses the same small public model as the rest of this
    file's OSS convention; no model.profile needed since Qwen3-0.6B has
    none of the LoRA-targeting/attn_implementation quirks a profile
    exists to paper over.
    """
    return {
        "job": {"name": "oss-standalone-sft-verl-smoke"},
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "method": {"type": "sft_verl"},
        "infra": {"vllm_mode": "none", "instance_type": "ml.p4de.24xlarge"},
        "training": {
            "sft_verl": {"data_format": "messages", "use_remove_padding": False},
        },
        "data": {"train_path": "s3://example-bucket/train.jsonl"},
        "output": {
            "s3_path": "s3://example-bucket/output/",
            "checkpoint_s3_path": "s3://example-bucket/checkpoints/",
        },
        "execution": {"type": "aws_training_job"},
        "container": {"recipe": "verl_sft"},
    }


def _optimizer_config_dict() -> dict:
    """Optimizer config on the generic parquet dataset + Bedrock candidate."""
    return {
        "job": {"name": "oss-optimizer-smoke"},
        "optimizer": {
            "datasets": [{"name": "parquet", "path": "s3://example-bucket/d.parquet"}],
            "prompt_set": "oss_demo",
            "prompts_path": "s3://example-bucket/prompts/oss_demo.yaml",
            "models": [
                {
                    "name": "anthropic.claude-3-haiku-20240307-v1:0",
                    "provider": "bedrock",
                    "region": "us-west-2",
                }
            ],
            "modes": ["single_call"],
            "judges": [{"name": "correctness", "weight": 1, "client": "default"}],
            "judge_clients": {
                "default": {
                    "provider": "bedrock",
                    "bedrock": {
                        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                        "region": "us-west-2",
                    },
                }
            },
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://example-bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }


def _evaluate_config_dict() -> dict:
    """Evaluate config on a catalog benchmark + a Bedrock candidate."""
    return {
        "job": {"name": "oss-evaluate-smoke"},
        "evaluate": {
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [
                {
                    "name": "anthropic.claude-3-haiku-20240307-v1:0",
                    "provider": "bedrock",
                    "region": "us-west-2",
                }
            ],
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://example-bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }


def _perf_config_dict() -> dict:
    return {
        "job": {"name": "oss-perf-smoke"},
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "perf_test": {
            "name": "run1",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://example-bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://example-bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }


class TestOssConfigsValidate:
    """Each operation can be configured end-to-end with OSS-only constructs."""

    def test_training_config_validates_clean(self):
        config = TrainingJobConfig.model_validate(_training_config_dict())
        assert validate_config(config) == []

    def test_sft_verl_training_config_validates_clean(self):
        config = TrainingJobConfig.model_validate(_sft_verl_training_config_dict())
        assert validate_config(config) == []

    def test_optimizer_config_validates_clean(self):
        config = OptimizerJobConfig.model_validate(_optimizer_config_dict())
        assert validate_optimizer_config(config) == []

    def test_perf_config_validates_clean(self):
        config = PerfTestJobConfig.model_validate(_perf_config_dict())
        assert validate_perf_config(config) == []

    def test_evaluate_config_validates_clean(self):
        config = EvaluateJobConfig.model_validate(_evaluate_config_dict())
        assert validate_evaluate_config(config) == []

    def test_training_config_needs_no_secrets_block(self):
        """Bedrock auth is the SageMaker execution role, so ``secrets:`` is optional.

        If this ever regresses, OSS users would be forced through IDPS, which
        they cannot reach.
        """
        config = TrainingJobConfig.model_validate(_training_config_dict())
        assert config.secrets is None

    def test_training_config_stages_no_extension(self):
        config = TrainingJobConfig.model_validate(_training_config_dict())
        assert config.container.extra_packages == []


class TestShippedOssExample:
    """The one example config an OSS user can actually run must stay valid.

    Every other template under ``examples/configs/`` targets the ``advice``
    task and Genos/IDPS credentials, so this file is the entire open-source
    on-ramp. Validating it here means a schema change cannot quietly break it.
    """

    EXAMPLE = CORE_DIR.parent / "examples" / "configs" / "oss_quickstart.yaml"

    def _skip_without_source_tree(self):
        if not self.EXAMPLE.is_file():
            pytest.skip("examples/ is not shipped in the wheel")

    def test_example_exists_and_validates(self):
        self._skip_without_source_tree()
        config = TrainingJobConfig.from_yaml(self.EXAMPLE)
        assert validate_config(config) == []

    def test_example_uses_no_intuit_constructs(self):
        """Guards against someone 'fixing' the example by pasting advice config."""
        self._skip_without_source_tree()
        raw = yaml.safe_load(self.EXAMPLE.read_text())

        assert raw["task"]["type"] == OSS_TASK
        assert "secrets" not in raw, "the OSS example must not need a secrets block"
        assert raw.get("container", {}).get("extra_packages", []) == []

        clients = raw["task"]["config"]["llm_clients"]
        for name, client in clients.items():
            assert client["provider"] == "bedrock", (
                f"llm_client {name!r} uses provider {client['provider']!r}; "
                "the OSS example must authenticate with the execution role"
            )


class TestShippedLocalExample:
    """A GPU machine can train from this file without an AWS account.

    The dataset ships in the repo. Validation stays offline: it checks the
    config and that the JSONL path exists, and it does not launch training.
    """

    EXAMPLE = CORE_DIR.parent / "examples" / "configs" / "local_example.yaml"

    def _skip_without_source_tree(self):
        if not self.EXAMPLE.is_file():
            pytest.skip("examples/ is not shipped in the wheel")

    def test_example_exists_and_validates(self):
        self._skip_without_source_tree()
        config = TrainingJobConfig.from_yaml(self.EXAMPLE)
        assert validate_config(config) == []
        assert config.execution.type == "local"
        assert config.secrets is None
        assert config.infra.vllm_mode.value == "none"
        assert config.task is not None
        assert config.task.type == OSS_TASK

        train_path = str(config.data.train_path)
        output_path = str(config.output.s3_path)
        assert not train_path.startswith("s3://")
        assert not output_path.startswith("s3://")

        dataset = self.EXAMPLE.parents[2] / train_path
        assert dataset.is_file(), f"sample dataset missing: {dataset}"
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) > config.data.eval_split_size
        for row in rows:
            assert "prompt" in row
            assert row["outcome_category"] in {
                "purchased",
                "added_to_cart",
                "browsed_after",
                "clicked",
                "opened",
                "ignored",
                "unsubscribed",
            }

    def test_example_needs_no_cloud_or_secrets(self):
        self._skip_without_source_tree()
        raw = yaml.safe_load(self.EXAMPLE.read_text())
        assert "secrets" not in raw
        assert "llm_clients" not in raw["task"].get("config", {})
        assert raw.get("container", {}).get("extra_packages", []) == []
        text = self.EXAMPLE.read_text(encoding="utf-8")
        assert "s3://" not in text
        assert "your-bucket" not in text


class TestCliValidatesOssConfigs:
    """The published entry point validates OSS configs off disk."""

    @pytest.mark.parametrize(
        "name,builder,command",
        [
            ("train.yaml", _training_config_dict, "validate"),
            ("optimizer.yaml", _optimizer_config_dict, "validate-optimizer"),
            ("perf.yaml", _perf_config_dict, "validate-perf"),
            ("evaluate.yaml", _evaluate_config_dict, "validate-evaluate"),
        ],
    )
    def test_cli_validate_exits_zero(self, tmp_path, name, builder, command):
        from foresightrlcore.cli import main

        path = tmp_path / name
        path.write_text(yaml.dump(builder()))
        assert main([command, "--config", str(path)]) == 0

    def test_cli_lists_benchmarks_without_the_extension(self):
        from foresightrlcore.cli import main

        assert main(["list-benchmarks"]) == 0


# ---------------------------------------------------------------------------
# Bedrock is the OSS auth story
# ---------------------------------------------------------------------------


class TestBedrockPathNeedsNoExtension:
    """Bedrock must resolve without the extension and without Genos env vars.

    This is the OSS replacement for the coverage in ``test_llm_resolver.py``,
    which skips when the extension is absent.
    """

    GENOS_VARS = (
        "GENOS_APP_SECRET",
        "GENOS_APP_ID",
        "GENOS_ENV",
        "GENOS_EXPERIENCE_ID",
        "GENOS_OFFLINE_JOB_ID",
        "GENOS_INTUIT_ORIGINATING_ASSETALIAS",
    )

    def test_resolves_bedrock_manager_with_no_genos_env(self, monkeypatch):
        from foresightrlcore.clients import BedrockClientManager, resolve_llm_manager
        from foresightrlcore.config.schema import LLMClientConfig

        for var in self.GENOS_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

        config = LLMClientConfig.model_validate(
            {"provider": "bedrock", "bedrock": {"model_id": "m", "region": "us-west-2"}}
        )
        assert isinstance(resolve_llm_manager(config), BedrockClientManager)
        for var in self.GENOS_VARS:
            assert os.environ.get(var) in (None, "")


# ---------------------------------------------------------------------------
# Static guarantees about the shipped source
# ---------------------------------------------------------------------------


def _core_python_files() -> Iterator[Path]:
    for path in sorted(CORE_DIR.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _eager_imports(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Imports executed at module import time.

    Top-level statements plus those nested in module-level ``if`` / ``try``
    blocks (``if TYPE_CHECKING:``, optional-dependency shims). Imports inside a
    function or method body are lazy and therefore fine — that is how the core
    reaches optional extensions.
    """
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    yield child


class TestFingerprintCheckScansShippingTrees:
    """The fingerprint guard must scan every directory the public tree ships.

    ``tests/`` and ``docs/`` used to be excluded on the claim that they do not
    ship. They do, and the exclusion made the check report clean while account
    ids, internal hosts, and production names were still in those trees.
    """

    SCRIPT = CORE_DIR.parent / "scripts" / "check_intuit_fingerprints.sh"

    def test_shipping_trees_are_not_hard_excluded(self):
        if not self.SCRIPT.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        text = self.SCRIPT.read_text(encoding="utf-8")
        start = text.index("EXCLUDED_PATHS=(")
        end = text.index(")", start)
        block = text[start:end]
        for skipped in ("tests/*", "docs/*", "examples/*", "docker/*"):
            assert skipped not in block, (
                f"{skipped} is excluded from the fingerprint check, but that "
                f"path ships in the public tree"
            )

    def test_pattern_covers_internal_pypi_host(self):
        """`artifact.intuit.com` does not match the artifactory host."""
        if not self.SCRIPT.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        text = self.SCRIPT.read_text(encoding="utf-8")
        pattern_line = next(
            line for line in text.splitlines() if line.startswith("PATTERN=")
        )
        assert "artifactory" in pattern_line
        assert "a\\.intuit\\.com" in pattern_line

    def test_fingerprint_script_passes(self):
        if not self.SCRIPT.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        import subprocess

        result = subprocess.run(
            ["bash", str(self.SCRIPT)],
            cwd=self.SCRIPT.parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestOssPublishStripList:
    """The publish strip list must stay accurate.

    Deleting ``intuit/`` is not sufficient to produce the public repo: four
    Intuit build files have to remain at the repo root because Jenkins and the
    ``auto`` CLI locate them there and cannot be pointed elsewhere. They are
    removed at publish time instead, which makes the list load-bearing -- a
    renamed path would silently ship. These tests fail in that case.
    """

    STRIP_LIST = CORE_DIR.parent / "scripts" / "oss_publish_striplist.txt"
    BUILD_SCRIPT = CORE_DIR.parent / "scripts" / "make_oss_tree.sh"

    def _entries(self) -> list[str]:
        if not self.STRIP_LIST.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        entries = []
        for raw in self.STRIP_LIST.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                entries.append(line)
        return entries

    def test_every_entry_exists(self):
        repo_root = CORE_DIR.parent
        missing = [e for e in self._entries() if not (repo_root / e).exists()]
        assert not missing, (
            f"strip-list entries no longer exist: {missing}. If a path was "
            f"renamed, the new one is not being stripped and would be published."
        )

    @pytest.mark.parametrize(
        "required",
        [
            "intuit",
            "Jenkinsfile",
            "msaas-config.yaml",
            "KubernetesPods.yaml",
            ".autorc",
        ],
    )
    def test_known_intuit_paths_are_listed(self, required):
        """Guards the non-obvious root files, not just ``intuit/``."""
        assert required in self._entries()

    def test_build_script_is_executable(self):
        if not self.BUILD_SCRIPT.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        assert os.access(self.BUILD_SCRIPT, os.X_OK)


class TestOssPublishTransform:
    """Identity edits happen in the output tree, not in this repository."""

    def _transform(self):
        script = CORE_DIR.parent / "scripts" / "oss_publish_transform.py"
        if not script.is_file():
            pytest.skip("scripts/ is not shipped in the wheel")
        import importlib.util

        spec = importlib.util.spec_from_file_location("oss_publish_transform", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.transform

    def test_rewrites_identity_and_drops_internal_hosts(self, tmp_path: Path):
        root = CORE_DIR.parent
        pyproject = root / "pyproject.toml"
        changelog = root / "CHANGELOG.md"
        codeowners = root / ".github" / "CODEOWNERS"
        if not pyproject.is_file():
            pytest.skip("no pyproject.toml next to the installed package")

        version = ""
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith('version = "'):
                version = line.split('"', 2)[1]
                break
        assert version

        (tmp_path / "pyproject.toml").write_text(
            pyproject.read_text(encoding="utf-8"), encoding="utf-8"
        )
        if changelog.is_file():
            (tmp_path / "CHANGELOG.md").write_text(
                changelog.read_text(encoding="utf-8"), encoding="utf-8"
            )
        (tmp_path / ".github").mkdir()
        (tmp_path / ".github" / "CODEOWNERS").write_text(
            (
                codeowners.read_text(encoding="utf-8")
                if codeowners.is_file()
                else "* @global-owner1 @global-owner2\n"
            ),
            encoding="utf-8",
        )

        preserved = self._transform()(tmp_path)

        rewritten = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
        assert preserved == version
        assert 'name = "foresightrlcore"' in rewritten
        assert f'version = "{version}"' in rewritten
        assert "intlgntsys-mlservices.foresightrlcore.foresightrlcore" not in rewritten
        assert "[[tool.uv.index]]" not in rewritten

        pieces = [rewritten, (tmp_path / ".github" / "CODEOWNERS").read_text()]
        if (tmp_path / "CHANGELOG.md").is_file():
            pieces.append((tmp_path / "CHANGELOG.md").read_text(encoding="utf-8"))
        combined = "\n".join(pieces)
        assert "artifact.intuit.com" not in combined
        assert "jira.intuit.com" not in combined
        assert "github.intuit.com" not in combined
        assert "global-owner" not in pieces[1]
        assert "@gazmany" in pieces[1]
        assert "@ofir408" in pieces[1]
        for internal in ("y" + "gazman", "oben" + "shoham"):
            assert f"@{internal}" not in pieces[1]
            if len(pieces) > 2:
                assert f"@{internal}" not in pieces[2]

        # The source tree is the development repo and keeps its package name.
        assert (
            'name = "intlgntsys-mlservices.foresightrlcore.foresightrlcore"'
            in pyproject.read_text(encoding="utf-8")
        )


class TestCitation:
    """Research users are asked to cite the framework, with named authors."""

    AUTHORS = ("Ofir Ben Shoham", "Yakov Gazman")

    def test_citation_file_names_authors(self):
        path = CORE_DIR.parent / "CITATION.cff"
        if not path.is_file():
            pytest.skip("CITATION.cff is not shipped in the wheel")
        text = path.read_text(encoding="utf-8")
        assert "cff-version:" in text
        assert "title: ForesightRLCore" in text
        assert "given-names: Ofir Ben" in text
        assert "family-names: Shoham" in text
        assert "given-names: Yakov" in text
        assert "family-names: Gazman" in text
        assert "artifact.intuit.com" not in text
        assert "jira.intuit.com" not in text
        assert "github.intuit.com" not in text

        pyproject = (CORE_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
        for author in self.AUTHORS:
            assert f'{{ name = "{author}" }}' in pyproject

    def test_readme_asks_users_to_cite(self):
        readme = CORE_DIR.parent / "README.md"
        if not readme.is_file():
            pytest.skip("README.md is not next to the installed package")
        text = readme.read_text(encoding="utf-8")
        assert (
            "If you use foresight-rl-core or find this repository useful "
            "for your research or work, please cite us using the following citation:"
        ) in text
        assert "title={ForesightRLCore}" in text
        assert "author={Shoham, Ofir Ben and Gazman, Yakov}" in text


class TestNoEagerExtensionDependency:
    """The core may reach for the extension lazily, never at import time.

    An eager import would make every one of these tests fail at collection in
    an OSS checkout, which is precisely the breakage this module exists to
    prevent. Catching it as a static assertion names the offending file
    instead of producing a wall of ``ModuleNotFoundError``.
    """

    def test_no_module_level_import_of_the_extension(self):
        offenders = []
        for path in _core_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in _eager_imports(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                else:
                    module = " ".join(alias.name for alias in node.names)
                if "foresightrl_intuit" in module:
                    rel = path.relative_to(CORE_DIR)
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "foresightrlcore imports the Intuit extension at module scope: "
            + ", ".join(offenders)
            + ". Move it inside the function that needs it and raise a clear "
            "error when it is absent."
        )

    def test_extension_is_not_a_declared_dependency(self):
        """The OSS wheel must not require the extension to install.

        Parsed rather than grepped: the file mentions the extension in prose
        comments, and only the actual requirement lists matter here.
        """
        import tomllib

        pyproject = CORE_DIR.parent / "pyproject.toml"
        if not pyproject.is_file():  # installed wheel, no source tree alongside
            pytest.skip("no pyproject.toml next to the installed package")

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        requirement_lists = [project.get("dependencies", [])]
        requirement_lists.extend(project.get("optional-dependencies", {}).values())
        requirement_lists.extend(data.get("dependency-groups", {}).values())

        for requirements in requirement_lists:
            for requirement in requirements:
                assert "foresightrl" not in str(requirement).replace("-", "_").lower()
