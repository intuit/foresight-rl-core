"""Tests for SageMaker InstanceGroup host discovery.

Fixture shape is the SM_TRAINING_ENV dump from failed mixed-instance job
grpo-advice-qwen3-06b-dvllm-mixed-br-20260914084643 (algo-1 = training).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from foresightrlcore.runtime.scripts.instance_group_hosts import (
    dump_instance_group_debug,
    main,
    resolve_instance_group_hosts,
)

# Captured from CloudWatch on the failed mixed smoke. extra_framework
# parameters were empty; instance_groups is a name list; hosts live in
# instance_groups_dict and resource_config.instance_groups (a list).
REAL_SM_TRAINING_ENV: dict = {
    "additional_framework_parameters": {},
    "current_host": "algo-1",
    "current_instance_group": "training",
    "current_instance_group_hosts": ["algo-1"],
    "hosts": ["algo-1", "algo-2"],
    "instance_groups": ["vllm", "training"],
    "instance_groups_dict": {
        "vllm": {
            "instance_group_name": "vllm",
            "instance_type": "ml.g6.12xlarge",
            "hosts": ["algo-2"],
        },
        "training": {
            "instance_group_name": "training",
            "instance_type": "ml.g6.16xlarge",
            "hosts": ["algo-1"],
        },
    },
    "is_hetero": True,
    "resource_config": {
        "current_host": "algo-1",
        "current_instance_type": "ml.g6.16xlarge",
        "current_group_name": "training",
        "hosts": ["algo-1", "algo-2"],
        "instance_groups": [
            {
                "instance_group_name": "vllm",
                "instance_type": "ml.g6.12xlarge",
                "hosts": ["algo-2"],
            },
            {
                "instance_group_name": "training",
                "instance_type": "ml.g6.16xlarge",
                "hosts": ["algo-1"],
            },
        ],
        "network_interface_name": "eth0",
    },
}

RESOURCECONFIG_LIST = {
    "current_host": "algo-1",
    "current_instance_group": "training",
    "hosts": ["algo-1", "algo-2"],
    "instance_groups": [
        {
            "instance_group_name": "vllm",
            "instance_type": "ml.g6.12xlarge",
            "hosts": ["algo-2"],
        },
        {
            "instance_group_name": "training",
            "instance_type": "ml.g6.16xlarge",
            "hosts": ["algo-1"],
        },
    ],
}


class TestResolveFromRealJobEnv:
    def test_vllm_hosts_are_algo2_not_algo1(self) -> None:
        hosts = resolve_instance_group_hosts("vllm", training_env=REAL_SM_TRAINING_ENV)
        assert hosts == ["algo-2"]

    def test_training_hosts_are_algo1(self) -> None:
        hosts = resolve_instance_group_hosts(
            "training", training_env=REAL_SM_TRAINING_ENV
        )
        assert hosts == ["algo-1"]

    def test_unknown_group_empty(self) -> None:
        hosts = resolve_instance_group_hosts("cpu", training_env=REAL_SM_TRAINING_ENV)
        assert hosts == []

    def test_legacy_additional_framework_parameters_are_empty_and_ignored(
        self,
    ) -> None:
        """The first parser only read this key — it is {} on real jobs."""
        extra = REAL_SM_TRAINING_ENV["additional_framework_parameters"]
        assert extra == {}
        hosts = resolve_instance_group_hosts(
            "vllm",
            training_env={
                "additional_framework_parameters": {},
                "instance_groups_dict": REAL_SM_TRAINING_ENV["instance_groups_dict"],
            },
        )
        assert hosts == ["algo-2"]


class TestFallbackShapes:
    def test_resource_config_list_when_dict_missing(self) -> None:
        env = {
            "resource_config": {
                "instance_groups": RESOURCECONFIG_LIST["instance_groups"]
            }
        }
        assert resolve_instance_group_hosts("vllm", training_env=env) == ["algo-2"]
        assert resolve_instance_group_hosts("training", training_env=env) == ["algo-1"]

    def test_resourceconfig_file_list_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "resourceconfig.json"
        path.write_text(json.dumps(RESOURCECONFIG_LIST))
        hosts = resolve_instance_group_hosts(
            "vllm",
            training_env={},
            resource_config_path=path,
        )
        assert hosts == ["algo-2"]

    def test_sm_resource_config_env(self) -> None:
        environ = {"SM_RESOURCE_CONFIG": json.dumps(RESOURCECONFIG_LIST)}
        hosts = resolve_instance_group_hosts(
            "training",
            training_env={},
            environ=environ,
            resource_config_path="/no/such/file.json",
        )
        assert hosts == ["algo-1"]

    def test_dict_keyed_instance_groups(self) -> None:
        rc = {
            "instance_groups": {
                "vllm": {"hosts": ["algo-9"]},
                "training": {"hosts": ["algo-8"]},
            }
        }
        assert resolve_instance_group_hosts(
            "vllm", training_env={}, resource_config=rc
        ) == ["algo-9"]

    def test_pascal_case_create_training_job_shape(self) -> None:
        rc = {
            "InstanceGroups": [
                {"InstanceGroupName": "vllm", "Hosts": ["algo-7"]},
                {"Name": "training", "hosts": ["algo-6"]},
            ]
        }
        assert resolve_instance_group_hosts(
            "vllm", training_env={}, resource_config=rc
        ) == ["algo-7"]
        assert resolve_instance_group_hosts(
            "training", training_env={}, resource_config=rc
        ) == ["algo-6"]

    def test_legacy_additional_framework_parameters(self) -> None:
        env = {
            "additional_framework_parameters": {
                "sagemaker_instance_groups": {
                    "vllm": {"hosts": ["algo-5"]},
                }
            }
        }
        assert resolve_instance_group_hosts("vllm", training_env=env) == ["algo-5"]

    def test_invalid_sm_training_env_json(self) -> None:
        hosts = resolve_instance_group_hosts(
            "vllm",
            environ={"SM_TRAINING_ENV": "not-json"},
            resource_config_path="/no/such/file.json",
        )
        assert hosts == []


class TestCli:
    def test_prints_hosts(self, capsys: pytest.CaptureFixture[str]) -> None:
        environ = {"SM_TRAINING_ENV": json.dumps(REAL_SM_TRAINING_ENV)}
        with patch.dict("os.environ", environ, clear=False):
            rc = main(["vllm"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "algo-2"

    def test_missing_group_exits_1(self) -> None:
        with patch.dict("os.environ", {"SM_TRAINING_ENV": "{}"}, clear=False):
            assert main(["vllm"]) == 1

    def test_dump_includes_instance_groups_dict(self) -> None:
        environ = {
            "SM_TRAINING_ENV": json.dumps(REAL_SM_TRAINING_ENV),
            "SM_CURRENT_HOST": "algo-1",
            "SM_CURRENT_INSTANCE_GROUP": "training",
        }
        text = dump_instance_group_debug(
            environ=environ, resource_config_path="/no/such/file.json"
        )
        payload = json.loads(text)
        assert payload["training_env.instance_groups_dict"]["vllm"]["hosts"] == [
            "algo-2"
        ]
        assert payload["SM_CURRENT_INSTANCE_GROUP"] == "training"
