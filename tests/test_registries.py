"""Tests for method, execution, and operation registries."""

from __future__ import annotations

import pytest


class TestMethodRegistry:
    """foresightrlcore.methods registry."""

    def test_register_and_get(self):
        from foresightrlcore.methods import get_method, list_methods, register_method

        class FakeMethod:
            method_type = "fake"

        register_method("fake", FakeMethod)
        assert get_method("fake") is FakeMethod
        assert "fake" in list_methods()

    def test_unknown_method_raises(self):
        from foresightrlcore.methods import get_method

        with pytest.raises(KeyError, match="Unknown method"):
            get_method("nonexistent_method_xyz")

    def test_grpo_auto_registered(self):
        import foresightrlcore.methods.grpo  # noqa: F401 — triggers registration
        from foresightrlcore.methods import get_method

        cls = get_method("grpo")
        assert cls.method_type == "grpo"  # type: ignore[attr-defined]


class TestExecutionRegistry:
    """foresightrlcore.execution registry."""

    def test_register_and_get(self):
        from foresightrlcore.execution import (
            get_backend,
            list_backends,
            register_backend,
        )

        class FakeBackend:
            backend_type = "fake"

        register_backend("fake", FakeBackend)
        assert get_backend("fake") is FakeBackend
        assert "fake" in list_backends()

    def test_unknown_backend_raises(self):
        from foresightrlcore.execution import get_backend

        with pytest.raises(KeyError, match="Unknown execution backend"):
            get_backend("nonexistent_backend_xyz")

    def test_local_auto_registered(self):
        import foresightrlcore.execution.local  # noqa: F401
        from foresightrlcore.execution import get_backend

        cls = get_backend("local")
        assert cls.backend_type == "local"  # type: ignore[attr-defined]

    def test_aws_auto_registered(self):
        import foresightrlcore.execution.aws_training_job  # noqa: F401
        from foresightrlcore.execution import get_backend

        cls = get_backend("aws_training_job")
        assert cls.backend_type == "aws_training_job"  # type: ignore[attr-defined]


class TestOperationRegistry:
    """foresightrlcore.operations registry."""

    def test_register_and_get(self):
        from foresightrlcore.operations import (
            get_operation,
            list_operations,
            register_operation,
        )

        class FakeOp:
            name = "fake"

        register_operation("fake", FakeOp)
        assert get_operation("fake") is FakeOp
        assert "fake" in list_operations()

    def test_unknown_op_raises(self):
        from foresightrlcore.operations import get_operation

        with pytest.raises(KeyError, match="Unknown operation"):
            get_operation("nonexistent_op_xyz")

    def test_train_auto_registered(self):
        import foresightrlcore.operations.train  # noqa: F401
        from foresightrlcore.operations import get_operation

        cls = get_operation("train")
        assert cls.name == "train"  # type: ignore[attr-defined]

    def test_perf_test_auto_registered(self):
        import foresightrlcore.operations.perf_test  # noqa: F401
        from foresightrlcore.operations import get_operation

        cls = get_operation("perf_test")
        assert cls.name == "perf_test"  # type: ignore[attr-defined]
