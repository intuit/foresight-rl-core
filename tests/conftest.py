"""Shared test fixtures for rl-train tests."""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

#: Set by the ``oss`` tox env to keep ``intuit/`` off ``sys.path`` even though
#: the directory is present in the monorepo.
OSS_ONLY = bool(os.environ.get("FORESIGHT_OSS_ONLY"))

# Stub heavy GPU dependencies before loading Intuit task packages.
_TrainerCallback = type("TrainerCallback", (), {"on_log": lambda self, *a, **kw: None})
_transformers = MagicMock()
_transformers.TrainerCallback = _TrainerCallback
_transformers.AutoTokenizer = MagicMock()
for _mod_name, _stub in [
    ("transformers", _transformers),
    ("langchain_openai", MagicMock()),
    ("datasets", MagicMock()),
    ("peft", MagicMock()),
]:
    sys.modules.setdefault(_mod_name, _stub)

# Load Intuit extension when present (monorepo dev / full test matrix).
_intuit_pkg = Path(__file__).resolve().parents[1] / "intuit"
if _intuit_pkg.is_dir() and not OSS_ONLY:
    _intuit_str = str(_intuit_pkg)
    if _intuit_str not in sys.path:
        sys.path.insert(0, _intuit_str)
    try:
        from foresightrl_intuit import _register

        _register()
        from foresightrlcore.plugins import load_plugins

        load_plugins()
    except ImportError:
        pass

#: Whether the optional Intuit extension can actually be imported. This is the
#: condition the suite gates on -- not ``OSS_ONLY`` -- so the three ways the
#: extension can be missing all behave identically: the ``oss`` tox env (which
#: keeps ``intuit/`` off ``sys.path``), an OSS checkout where ``intuit/`` has
#: been deleted, and a plain ``pip install foresightrlcore`` with no monorepo
#: around it. Gating on the env var alone made the last two hard-fail at
#: collection instead of skipping.
INTUIT_AVAILABLE = importlib.util.find_spec("foresightrl_intuit") is not None

# Stub heavy GPU dependencies that are not installed in the test environment.
# Must happen before any test module imports the advice/runtime modules.
# (Duplicates above are harmless — setdefault keeps the first stub.)
for _mod_name in ("transformers", "langchain_openai", "datasets", "peft"):
    sys.modules.setdefault(
        _mod_name, _transformers if _mod_name == "transformers" else MagicMock()
    )


def pytest_collection_modifyitems(config, items):
    """Skip extension-dependent tests whenever the extension is not importable."""
    if INTUIT_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="requires the foresightrl-intuit extension")
    for item in items:
        if "requires_intuit" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_yaml(tmp_path: Path):
    """Factory fixture that writes a YAML string to a temp file and returns the path."""

    def _write(content: str, filename: str = "config.yaml") -> Path:
        p = tmp_path / filename
        p.write_text(textwrap.dedent(content))
        return p

    return _write


@pytest.fixture
def minimal_config_dict() -> dict:
    """Minimal valid config dict matching the TrainingJobConfig schema."""
    return {
        "job": {"name": "test-job"},
        "model": {"name": "Qwen/Qwen3-4B"},
        "data": {"train_path": "s3://bucket/train.jsonl"},
        "output": {"s3_path": "s3://bucket/output"},
        "task": {"type": "advice"},
    }


@pytest.fixture
def minimal_config_yaml(tmp_yaml, minimal_config_dict) -> Path:
    """Write a minimal valid YAML config and return the path."""
    content = yaml.dump(minimal_config_dict, default_flow_style=False)
    return tmp_yaml(content)


@pytest.fixture
def full_config_dict() -> dict:
    """Full config dict with all new Phase 1 fields (method, execution)."""
    return {
        "version": 1,
        "job": {"name": "full-test-job", "max_runtime_hours": 6},
        "model": {
            "name": "Qwen/Qwen3-14B",
            "enable_thinking": True,
            "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
        },
        "method": {"type": "grpo"},
        "training": {
            "epochs": 1,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "grpo": {
                "num_generations": 8,
                "max_completion_length": 8192,
                "beta": 0.001,
                "loss_type": "dr_grpo",
            },
        },
        "data": {"train_path": "s3://my-bucket/train"},
        "output": {"s3_path": "s3://my-bucket/output"},
        "execution": {"type": "aws_training_job"},
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
    }


@pytest.fixture
def full_config_yaml(tmp_yaml, full_config_dict) -> Path:
    """Write a full config YAML and return the path."""
    content = yaml.dump(full_config_dict, default_flow_style=False)
    return tmp_yaml(content)


@pytest.fixture
def local_config_dict() -> dict:
    """Config dict for local GPU execution."""
    return {
        "job": {"name": "local-test"},
        "model": {"name": "Qwen/Qwen3-4B"},
        "method": {"type": "grpo"},
        "training": {
            "grpo": {"num_generations": 4},
        },
        "data": {"train_path": "/data/train"},
        "output": {"s3_path": "/output/model"},
        "execution": {
            "type": "local",
            "local": {
                "gpus": "0,1,2,3,4,5,6,7",
                "output_dir": "/tmp/training_output",
            },
        },
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
    }


@pytest.fixture
def local_config_yaml(tmp_yaml, local_config_dict) -> Path:
    """Write a local config YAML and return the path."""
    content = yaml.dump(local_config_dict, default_flow_style=False)
    return tmp_yaml(content)
