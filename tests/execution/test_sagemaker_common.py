"""Tests for shared SageMaker CreateTrainingJob request helpers."""

from __future__ import annotations

import os
import sys
import textwrap
import types
from typing import Any

import pytest

from foresightrlcore.execution.sagemaker_common import (
    EXTRA_PACKAGES_ENV,
    apply_inter_container_encryption,
    extra_packages_env,
    serverless_inter_container_encryption,
    stage_extra_packages,
)


def _install_fake_training_job(monkeypatch, supports_encryption: bool = True) -> Any:
    """Stub out ``sagemaker.core.resources.TrainingJob`` for the CM tests."""

    if supports_encryption:

        class FakeTrainingJob:
            calls: list[dict] = []

            @classmethod
            def create(
                cls,
                training_job_name: str,
                enable_inter_container_traffic_encryption: bool | None = None,
                **kwargs: Any,
            ) -> str:
                cls.calls.append(
                    {
                        "training_job_name": training_job_name,
                        "enable_inter_container_traffic_encryption": (
                            enable_inter_container_traffic_encryption
                        ),
                    }
                )
                return training_job_name

    else:

        class FakeTrainingJob:  # type: ignore[no-redef]
            calls: list[dict] = []

            @classmethod
            def create(cls, training_job_name: str, **kwargs: Any) -> str:
                cls.calls.append({"training_job_name": training_job_name})
                return training_job_name

    FakeTrainingJob.calls = []
    module = types.ModuleType("sagemaker.core.resources")
    module.TrainingJob = FakeTrainingJob  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sagemaker.core.resources", module)
    return FakeTrainingJob


class TestApplyInterContainerEncryption:
    def test_sets_true(self):
        request: dict = {}
        apply_inter_container_encryption(request, True)
        assert request["EnableInterContainerTrafficEncryption"] is True

    def test_sets_false(self):
        request: dict = {}
        apply_inter_container_encryption(request, False)
        assert request["EnableInterContainerTrafficEncryption"] is False

    def test_mutates_in_place_preserving_other_keys(self):
        request: dict = {"TrainingJobName": "job", "ResourceConfig": {"x": 1}}
        apply_inter_container_encryption(request, True)
        assert request["TrainingJobName"] == "job"
        assert request["ResourceConfig"] == {"x": 1}
        assert request["EnableInterContainerTrafficEncryption"] is True


@pytest.fixture
def fake_extension(tmp_path, monkeypatch):
    """An importable throwaway package the stager can resolve."""
    pkg_root = tmp_path / "site"
    pkg = pkg_root / "fake_ext"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(textwrap.dedent("""
            REGISTERED = []

            def _register():
                REGISTERED.append("yes")
            """))
    (pkg / "inner.py").write_text("VALUE = 1\n")
    java = pkg / "perf_test" / "gatling_java" / "com" / "example"
    java.mkdir(parents=True)
    (java / "MySim.java").write_text("package com.example;\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "junk.pyc").write_text("junk")

    monkeypatch.syspath_prepend(str(pkg_root))
    sys.modules.pop("fake_ext", None)
    yield "fake_ext"
    sys.modules.pop("fake_ext", None)


class TestStageExtraPackages:
    def test_copies_package_source(self, tmp_path, fake_extension):
        dest = tmp_path / "src"
        dest.mkdir()

        staged = stage_extra_packages(str(dest), [fake_extension])

        assert staged == [fake_extension]
        assert (dest / "fake_ext" / "__init__.py").is_file()
        assert (dest / "fake_ext" / "inner.py").is_file()

    def test_copies_non_python_assets(self, tmp_path, fake_extension):
        """Extensions ship Gatling .java sources the perf runner compiles.

        Narrowing the copy to *.py would drop them, and the job would only fail
        later with a Java class-not-found from Gradle.
        """
        dest = tmp_path / "src"
        dest.mkdir()

        stage_extra_packages(str(dest), [fake_extension])

        staged_java = (
            dest
            / "fake_ext"
            / "perf_test"
            / "gatling_java"
            / "com"
            / "example"
            / "MySim.java"
        )
        assert staged_java.is_file()

    def test_skips_bytecode_artifacts(self, tmp_path, fake_extension):
        dest = tmp_path / "src"
        dest.mkdir()

        stage_extra_packages(str(dest), [fake_extension])

        assert not (dest / "fake_ext" / "__pycache__").exists()

    def test_empty_list_is_a_noop(self, tmp_path):
        dest = tmp_path / "src"
        dest.mkdir()

        assert stage_extra_packages(str(dest), []) == []
        assert list(dest.iterdir()) == []

    def test_unimportable_package_raises(self, tmp_path):
        dest = tmp_path / "src"
        dest.mkdir()

        with pytest.raises(ValueError, match="not importable"):
            stage_extra_packages(str(dest), ["definitely_not_a_real_package_xyz"])


class TestExtraPackagesEnv:
    def test_joins_names(self):
        assert extra_packages_env(["a", "b"]) == {EXTRA_PACKAGES_ENV: "a,b"}

    def test_empty_yields_no_env(self):
        assert extra_packages_env([]) == {}


class TestLoadExtraPackages:
    def test_imports_and_calls_register_hook(self, monkeypatch, fake_extension):
        import foresightrlcore.plugins as plugins

        monkeypatch.setattr(plugins, "_EXTRA_PACKAGES_LOADED", set())
        monkeypatch.setenv(EXTRA_PACKAGES_ENV, fake_extension)

        plugins.load_extra_packages()

        assert sys.modules[fake_extension].REGISTERED == ["yes"]

    def test_each_package_loads_once(self, monkeypatch, fake_extension):
        import foresightrlcore.plugins as plugins

        monkeypatch.setattr(plugins, "_EXTRA_PACKAGES_LOADED", set())
        monkeypatch.setenv(EXTRA_PACKAGES_ENV, fake_extension)

        plugins.load_extra_packages()
        plugins.load_extra_packages()

        assert sys.modules[fake_extension].REGISTERED == ["yes"]

    def test_unset_env_is_a_noop(self, monkeypatch):
        import foresightrlcore.plugins as plugins

        monkeypatch.setattr(plugins, "_EXTRA_PACKAGES_LOADED", set())
        monkeypatch.delenv(EXTRA_PACKAGES_ENV, raising=False)

        plugins.load_extra_packages()

        assert plugins._EXTRA_PACKAGES_LOADED == set()

    def test_broken_package_does_not_raise(self, monkeypatch):
        import foresightrlcore.plugins as plugins

        monkeypatch.setattr(plugins, "_EXTRA_PACKAGES_LOADED", set())
        monkeypatch.setenv(EXTRA_PACKAGES_ENV, "definitely_not_a_real_package_xyz")

        plugins.load_extra_packages()

        assert os.environ[EXTRA_PACKAGES_ENV] == "definitely_not_a_real_package_xyz"


class TestServerlessInterContainerEncryption:
    def test_injects_flag_into_create(self, monkeypatch):
        job = _install_fake_training_job(monkeypatch)

        with serverless_inter_container_encryption(True):
            job.create(training_job_name="rlvr-job")

        assert job.calls == [
            {
                "training_job_name": "rlvr-job",
                "enable_inter_container_traffic_encryption": True,
            }
        ]

    def test_restores_create_on_exit(self, monkeypatch):
        job = _install_fake_training_job(monkeypatch)
        original = job.create

        with serverless_inter_container_encryption(True):
            assert job.create is not original
        assert job.create == original

        job.create(training_job_name="unpatched")
        assert job.calls[-1]["enable_inter_container_traffic_encryption"] is None

    def test_restores_create_when_submit_raises(self, monkeypatch):
        job = _install_fake_training_job(monkeypatch)
        original = job.create

        with pytest.raises(ValueError):
            with serverless_inter_container_encryption(True):
                raise ValueError("submit failed")

        assert job.create == original

    def test_disabled_does_not_patch(self, monkeypatch):
        job = _install_fake_training_job(monkeypatch)
        original = job.create

        with serverless_inter_container_encryption(False):
            assert job.create == original
            job.create(training_job_name="opted-out")

        assert job.calls[-1]["enable_inter_container_traffic_encryption"] is None

    def test_raises_when_sdk_drops_the_parameter(self, monkeypatch):
        _install_fake_training_job(monkeypatch, supports_encryption=False)

        with pytest.raises(RuntimeError, match="no longer accepts"):
            with serverless_inter_container_encryption(True):
                pass  # pragma: no cover - context manager raises on entry
