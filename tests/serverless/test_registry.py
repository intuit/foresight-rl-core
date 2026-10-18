"""Tests for :mod:`foresightrlcore.serverless.registry`."""

from __future__ import annotations

import pytest

from foresightrlcore.serverless import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot + restore the module-level registries between tests."""
    snap_lambdas = dict(registry.LAMBDAS)
    snap_datasets = dict(registry.DATASETS)
    snap_converters = dict(registry.CONVERTERS)
    try:
        yield
    finally:
        registry.LAMBDAS.clear()
        registry.LAMBDAS.update(snap_lambdas)
        registry.DATASETS.clear()
        registry.DATASETS.update(snap_datasets)
        registry.CONVERTERS.clear()
        registry.CONVERTERS.update(snap_converters)


class TestRegisterLambda:
    def test_register_then_lookup(self) -> None:
        spec = registry.register_lambda(
            name="test-reward",
            main_module="foresightrl_intuit.tasks.advice.lambda_reward",
            extra_modules=["foresightrl_intuit.tasks.advice.judge_logic"],
            env_defaults={"FOO": "bar"},
            description="unit-test entry",
        )
        assert spec.name == "test-reward"
        assert registry.LAMBDAS["test-reward"] is spec
        assert spec.env_defaults["FOO"] == "bar"

    def test_overwrite_logs_warning(self, caplog) -> None:
        registry.register_lambda(
            name="dup", main_module="foresightrl_intuit.tasks.advice.lambda_reward"
        )
        with caplog.at_level("WARNING"):
            registry.register_lambda(
                name="dup",
                main_module="foresightrl_intuit.tasks.advice.lambda_reward",
                extra_modules=["foresightrl_intuit.tasks.advice.judge_logic"],
            )
        assert any("overwriting" in rec.message for rec in caplog.records)

    def test_default_runtime_timeout_memory_match_constants(self) -> None:
        """A bare ``register_lambda`` call must yield the Bedrock-friendly
        package-wide defaults — not AWS's brittle global defaults (3s /
        128MB / python3.9), which silently break any non-trivial reward
        function. See ``DEFAULT_LAMBDA_*`` constants for the rationale."""
        spec = registry.register_lambda(
            name="defaults-only",
            main_module="foresightrl_intuit.tasks.advice.lambda_reward",
        )
        assert spec.runtime == registry.DEFAULT_LAMBDA_RUNTIME == "python3.12"
        assert spec.timeout_s == registry.DEFAULT_LAMBDA_TIMEOUT_S == 900
        assert spec.memory_mb == registry.DEFAULT_LAMBDA_MEMORY_MB == 1024

    def test_explicit_runtime_timeout_memory_override_defaults(self) -> None:
        spec = registry.register_lambda(
            name="custom-config",
            main_module="foresightrl_intuit.tasks.advice.lambda_reward",
            runtime="python3.13",
            timeout_s=60,
            memory_mb=512,
        )
        assert spec.runtime == "python3.13"
        assert spec.timeout_s == 60
        assert spec.memory_mb == 512


class TestTruncateLambdaFunctionName:
    """``truncate_lambda_function_name`` enforces AWS's 64-char cap on
    ``CreateFunction`` while preserving the ``SageMaker-evaluator-``
    prefix and the trailing ``-YYYYMMDD_HHMMSS`` timestamp shape."""

    def test_short_name_unchanged(self) -> None:
        name = "SageMaker-evaluator-example-reward-fn1-20260519_134046"
        assert len(name) <= registry.MAX_LAMBDA_FUNCTION_NAME_LEN
        assert registry.truncate_lambda_function_name(name) == name

    def test_boundary_64_char_name_unchanged(self) -> None:
        name = "a" * registry.MAX_LAMBDA_FUNCTION_NAME_LEN
        assert registry.truncate_lambda_function_name(name) == name

    def test_long_name_is_truncated_to_cap(self) -> None:
        name = "SageMaker-evaluator-example-reward-fn1-smoke-source-" "20260519_133526"
        assert len(name) > registry.MAX_LAMBDA_FUNCTION_NAME_LEN
        out = registry.truncate_lambda_function_name(name)
        assert len(out) == registry.MAX_LAMBDA_FUNCTION_NAME_LEN

    def test_long_name_preserves_prefix_and_suffix(self) -> None:
        name = (
            "SageMaker-evaluator-some-very-long-evaluator-name-here-" "20260519_133526"
        )
        out = registry.truncate_lambda_function_name(name)
        assert out.startswith("SageMaker-evaluator-")
        assert out.endswith("20260519_133526")

    def test_truncation_logs_warning(self, caplog) -> None:
        name = "x" * (registry.MAX_LAMBDA_FUNCTION_NAME_LEN + 10)
        with caplog.at_level("WARNING"):
            registry.truncate_lambda_function_name(name)
        assert any(
            "exceeds AWS's" in rec.message and "char limit" in rec.message
            for rec in caplog.records
        )


class TestResolveLambda:
    @pytest.mark.requires_intuit
    def test_resolves_main_and_extras_to_existing_files(self) -> None:
        registry.register_lambda(
            name="resolve-me",
            main_module="foresightrl_intuit.tasks.advice.lambda_reward",
            extra_modules=["foresightrl_intuit.tasks.advice.judge_logic"],
        )
        main_path, extras = registry.resolve_lambda("resolve-me")
        assert main_path.endswith("lambda_reward.py")
        assert any(p.endswith("judge_logic.py") for p in extras)

    def test_unknown_name_raises_keyerror_with_helpful_message(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            registry.resolve_lambda("missing-lambda")


class TestRegisterConverter:
    def test_register_with_default_schema(self) -> None:
        spec = registry.register_converter(
            name="adv",
            mapper="foresightrl_intuit.tasks.advice.convert:convert_record",
        )
        assert spec.name == "adv"
        assert "rlvr" in spec.schemas

    def test_register_with_explicit_schemas(self) -> None:
        spec = registry.register_converter(
            name="multi",
            mapper="foresightrl_intuit.tasks.advice.convert:convert_record",
            schemas={"rlvr", "sft"},
        )
        assert spec.schemas == frozenset({"rlvr", "sft"})


@pytest.mark.requires_intuit
class TestResolveConverter:
    def test_resolve_by_registry_name(self) -> None:
        registry.register_converter(
            name="advctest",
            mapper="foresightrl_intuit.tasks.advice.convert:convert_record",
        )
        fn, schemas = registry.resolve_converter("advctest")
        assert callable(fn)
        assert "rlvr" in schemas

    def test_resolve_by_dotted_path(self) -> None:
        fn, schemas = registry.resolve_converter(
            "foresightrl_intuit.tasks.advice.convert:convert_record"
        )
        assert callable(fn)
        # Dotted-path imports default to {"rlvr"}.
        assert "rlvr" in schemas

    def test_unknown_registry_name_raises(self) -> None:
        with pytest.raises(KeyError, match="No registered converter"):
            registry.resolve_converter("does-not-exist")

    def test_bad_dotted_path_raises(self) -> None:
        with pytest.raises(AttributeError):
            registry.resolve_converter(
                "foresightrl_intuit.tasks.advice.convert:no_such_function"
            )


class TestDiscoverTasks:
    def test_idempotent(self) -> None:
        registry.discover_tasks()
        size_before = len(registry.LAMBDAS)
        registry.discover_tasks()
        assert len(registry.LAMBDAS) == size_before

    @pytest.mark.requires_intuit
    def test_advice_lambda_is_registered_after_discovery(self) -> None:
        from foresightrlcore.tasks import load_task_module

        registry.discover_tasks()
        load_task_module("advice")
        assert any(
            spec.main_module.endswith(".lambda_reward")
            for spec in registry.LAMBDAS.values()
        ), "the extension registered no reward lambda"
