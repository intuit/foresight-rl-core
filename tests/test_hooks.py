"""Tests for foresightrlcore.hooks — training hooks loader and resolver."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.schema import HooksConfig, TrainingJobConfig
from foresightrlcore.hooks import (
    import_dotted_path,
    resolve_callbacks,
    resolve_pre_post_hooks,
    resolve_trainer_class,
)

# ── import_dotted_path ─────────────────────────────────────────────────


class TestImportDottedPath:
    """Tests for the low-level dotted-path importer."""

    def test_import_builtin_class(self):
        cls = import_dotted_path("collections.OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_import_function(self):
        fn = import_dotted_path("os.path.join")
        import os.path

        assert fn is os.path.join

    def test_import_nested(self):
        cls = import_dotted_path("email.mime.text.MIMEText")
        from email.mime.text import MIMEText

        assert cls is MIMEText

    def test_no_dot_raises_import_error(self):
        with pytest.raises(ImportError, match="must contain at least one dot"):
            import_dotted_path("nodots")

    def test_bad_module_raises_import_error(self):
        with pytest.raises(ImportError, match="Could not import module"):
            import_dotted_path("nonexistent_pkg_xyz.SomeClass")

    def test_bad_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            import_dotted_path("os.path.nonexistent_attr_xyz")


# ── resolve_trainer_class ──────────────────────────────────────────────


class TestResolveTrainerClass:
    """Tests for trainer class resolution from hooks config."""

    def test_returns_default_when_hooks_is_none(self):
        default = type("DefaultTrainer", (), {})
        assert resolve_trainer_class(None, default) is default

    def test_returns_default_when_trainer_class_not_set(self):
        hooks = HooksConfig()
        default = type("DefaultTrainer", (), {})
        assert resolve_trainer_class(hooks, default) is default

    def test_imports_custom_class(self):
        hooks = HooksConfig(trainer_class="collections.OrderedDict")
        from collections import OrderedDict

        result = resolve_trainer_class(hooks, dict)
        assert result is OrderedDict

    def test_works_with_dict_config(self):
        hooks_dict = {"trainer_class": "collections.OrderedDict"}
        from collections import OrderedDict

        result = resolve_trainer_class(hooks_dict, dict)
        assert result is OrderedDict

    def test_dict_without_trainer_class_returns_default(self):
        hooks_dict: dict = {}
        default = type("D", (), {})
        assert resolve_trainer_class(hooks_dict, default) is default

    def test_bad_path_raises(self):
        hooks = HooksConfig(trainer_class="nonexistent.Trainer")
        with pytest.raises(ImportError):
            resolve_trainer_class(hooks, dict)


# ── resolve_callbacks ──────────────────────────────────────────────────


class TestResolveCallbacks:
    """Tests for callback resolution from hooks config."""

    def test_returns_empty_when_hooks_is_none(self):
        assert resolve_callbacks(None) == []

    def test_returns_empty_when_no_callbacks(self):
        hooks = HooksConfig()
        assert resolve_callbacks(hooks) == []

    def test_returns_empty_for_empty_list(self):
        hooks = HooksConfig(callbacks=[])
        assert resolve_callbacks(hooks) == []

    def test_instantiates_callback_classes(self):
        hooks = HooksConfig(callbacks=["collections.OrderedDict"])
        result = resolve_callbacks(hooks)
        assert len(result) == 1
        from collections import OrderedDict

        assert isinstance(result[0], OrderedDict)

    def test_multiple_callbacks(self):
        hooks = HooksConfig(
            callbacks=["collections.OrderedDict", "collections.defaultdict"]
        )
        result = resolve_callbacks(hooks)
        assert len(result) == 2

    def test_bad_callback_path_raises(self):
        hooks = HooksConfig(callbacks=["nonexistent.Callback"])
        with pytest.raises(ImportError):
            resolve_callbacks(hooks)

    def test_works_with_dict_config(self):
        hooks_dict = {"callbacks": ["collections.OrderedDict"]}
        result = resolve_callbacks(hooks_dict)
        assert len(result) == 1

    def test_dict_without_callbacks_returns_empty(self):
        hooks_dict: dict = {}
        assert resolve_callbacks(hooks_dict) == []


# ── resolve_pre_post_hooks ─────────────────────────────────────────────


class TestResolvePrePostHooks:
    """Tests for pre/post training hook resolution."""

    def test_returns_none_none_when_hooks_is_none(self):
        pre, post = resolve_pre_post_hooks(None)
        assert pre is None
        assert post is None

    def test_returns_none_none_when_not_set(self):
        hooks = HooksConfig()
        pre, post = resolve_pre_post_hooks(hooks)
        assert pre is None
        assert post is None

    def test_loads_pre_train_hook(self):
        hooks = HooksConfig(pre_train="os.path.join")
        pre, post = resolve_pre_post_hooks(hooks)
        assert pre is not None
        assert callable(pre)
        assert post is None

    def test_loads_post_train_hook(self):
        hooks = HooksConfig(post_train="os.path.join")
        pre, post = resolve_pre_post_hooks(hooks)
        assert pre is None
        assert post is not None
        assert callable(post)

    def test_loads_both_hooks(self):
        hooks = HooksConfig(
            pre_train="os.path.join",
            post_train="os.path.exists",
        )
        pre, post = resolve_pre_post_hooks(hooks)
        assert pre is not None
        assert post is not None
        assert callable(pre)
        assert callable(post)

    def test_bad_pre_train_raises(self):
        hooks = HooksConfig(pre_train="nonexistent.hook")
        with pytest.raises(ImportError):
            resolve_pre_post_hooks(hooks)

    def test_works_with_dict_config(self):
        hooks_dict = {"pre_train": "os.path.join"}
        pre, post = resolve_pre_post_hooks(hooks_dict)
        assert pre is not None
        assert callable(pre)
        assert post is None


# ── HooksConfig schema ─────────────────────────────────────────────────


class TestHooksConfigSchema:
    """Tests for HooksConfig Pydantic model parsing."""

    def test_all_fields_default_to_none(self):
        hooks = HooksConfig()
        assert hooks.trainer_class is None
        assert hooks.pre_train is None
        assert hooks.post_train is None
        assert hooks.callbacks is None

    def test_all_fields_set(self):
        hooks = HooksConfig(
            trainer_class="pkg.Trainer",
            pre_train="pkg.pre",
            post_train="pkg.post",
            callbacks=["pkg.Cb1", "pkg.Cb2"],
        )
        assert hooks.trainer_class == "pkg.Trainer"
        assert hooks.pre_train == "pkg.pre"
        assert hooks.post_train == "pkg.post"
        assert hooks.callbacks == ["pkg.Cb1", "pkg.Cb2"]

    def test_from_dict(self):
        data = {
            "trainer_class": "pkg.T",
            "callbacks": ["pkg.C"],
        }
        hooks = HooksConfig.model_validate(data)
        assert hooks.trainer_class == "pkg.T"
        assert hooks.callbacks == ["pkg.C"]
        assert hooks.pre_train is None


# ── HooksConfig in TrainingJobConfig ───────────────────────────────────


class TestHooksInTrainingJobConfig:
    """Test hooks integration in the full config schema."""

    def test_config_without_hooks_has_none(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.training.hooks is None

    def test_config_with_hooks_parses(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "hooks": {
                        "trainer_class": "my_pkg.CustomTrainer",
                        "pre_train": "my_pkg.freeze_layers",
                        "post_train": "my_pkg.upload_metrics",
                        "callbacks": ["my_pkg.EarlyStopping", "my_pkg.Logger"],
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        hooks = config.training.hooks
        assert hooks is not None
        assert hooks.trainer_class == "my_pkg.CustomTrainer"
        assert hooks.pre_train == "my_pkg.freeze_layers"
        assert hooks.post_train == "my_pkg.upload_metrics"
        assert hooks.callbacks == ["my_pkg.EarlyStopping", "my_pkg.Logger"]

    def test_hooks_roundtrip_yaml(self, tmp_yaml, tmp_path):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "hooks": {
                        "trainer_class": "my_pkg.T",
                        "callbacks": ["my_pkg.C"],
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        out = tmp_path / "rt.yaml"
        config.to_yaml(out)
        config2 = TrainingJobConfig.from_yaml(out)
        assert config2.training.hooks is not None
        assert config2.training.hooks.trainer_class == "my_pkg.T"
        assert config2.training.hooks.callbacks == ["my_pkg.C"]

    def test_hooks_partial_fields(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "hooks": {
                        "callbacks": ["my_pkg.Cb"],
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        hooks = config.training.hooks
        assert hooks is not None
        assert hooks.trainer_class is None
        assert hooks.callbacks == ["my_pkg.Cb"]


# ── Backward compatibility ─────────────────────────────────────────────


class TestHooksBackwardCompat:
    """Existing configs without hooks continue to work."""

    def test_full_config_without_hooks(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        assert config.training.hooks is None
        assert config.job.name == "full-test-job"
        assert config.training.grpo.loss_type == "dr_grpo"

    def test_grpo_handler_without_hooks(self, full_config_yaml):
        from foresightrlcore.methods.grpo import GrpoMethodHandler

        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        errors = handler.validate(config)
        assert errors == []
        args = handler.build_trainer_args(config)
        assert "num_generations" in args
