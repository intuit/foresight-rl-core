"""Resolve SageMaker heterogeneous InstanceGroup host lists.

SageMaker writes cluster topology to ``SM_TRAINING_ENV`` and
``/opt/ml/input/config/resourceconfig.json``. The container schema is
snake_case and list-shaped, not the PascalCase ``CreateTrainingJob`` API.

Confirmed on mixed dedicated_vllm (job
``grpo-advice-qwen3-06b-dvllm-mixed-br-20260914084643``):

* ``SM_TRAINING_ENV["instance_groups_dict"][group]["hosts"]``
* ``SM_TRAINING_ENV["resource_config"]["instance_groups"]`` is a **list** of
  ``{instance_group_name, instance_type, hosts}``
* ``additional_framework_parameters`` is ``{}`` (do not rely on it)
* ``algo-1`` is not necessarily the vLLM node

Called from ``entrypoint.sh`` as::

    python3 instance_group_hosts.py <group>
    python3 instance_group_hosts.py --dump
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_RESOURCE_CONFIG_PATH = "/opt/ml/input/config/resourceconfig.json"


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _hosts_from_entry(entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return []
    hosts = entry.get("hosts") or entry.get("Hosts") or []
    if isinstance(hosts, str):
        return [hosts] if hosts else []
    if isinstance(hosts, list):
        return [str(h) for h in hosts if h]
    return []


def _group_name(entry: Mapping[str, Any]) -> str | None:
    name = (
        entry.get("instance_group_name")
        or entry.get("InstanceGroupName")
        or entry.get("Name")
        or entry.get("name")
    )
    return str(name) if name else None


def _hosts_from_groups_list(groups: object, group: str) -> list[str]:
    if not isinstance(groups, list):
        return []
    for entry in groups:
        if isinstance(entry, dict) and _group_name(entry) == group:
            hosts = _hosts_from_entry(entry)
            if hosts:
                return hosts
    return []


def _hosts_from_groups_dict(groups: object, group: str) -> list[str]:
    if not isinstance(groups, dict):
        return []
    return _hosts_from_entry(groups.get(group))


def _load_resource_config_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        value = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _hosts_from_resource_config(rc: Mapping[str, Any], group: str) -> list[str]:
    hosts = _hosts_from_groups_list(rc.get("instance_groups"), group)
    if hosts:
        return hosts
    hosts = _hosts_from_groups_dict(rc.get("instance_groups"), group)
    if hosts:
        return hosts
    return _hosts_from_groups_list(rc.get("InstanceGroups"), group)


def resolve_instance_group_hosts(
    group: str,
    *,
    training_env: Mapping[str, Any] | None = None,
    resource_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    resource_config_path: str | Path = DEFAULT_RESOURCE_CONFIG_PATH,
) -> list[str]:
    """Return hostnames for ``group`` (e.g. ``vllm`` or ``training``).

    Lookup order matches what SageMaker actually publishes, then older
    shapes kept as last-ditch fallbacks.
    """
    env_map: Mapping[str, str] = os.environ if environ is None else environ
    env: Mapping[str, Any]
    if training_env is None:
        env = _parse_json_object(env_map.get("SM_TRAINING_ENV", "") or "")
    else:
        env = training_env

    hosts = _hosts_from_groups_dict(env.get("instance_groups_dict"), group)
    if hosts:
        return hosts

    rc_from_env = env.get("resource_config")
    if isinstance(rc_from_env, Mapping):
        hosts = _hosts_from_resource_config(rc_from_env, group)
        if hosts:
            return hosts

    extra = env.get("additional_framework_parameters")
    if isinstance(extra, Mapping):
        hosts = _hosts_from_groups_dict(extra.get("sagemaker_instance_groups"), group)
        if hosts:
            return hosts

    rc: Mapping[str, Any]
    if resource_config is None:
        parsed = _parse_json_object(env_map.get("SM_RESOURCE_CONFIG", "") or "")
        rc = parsed if parsed else _load_resource_config_file(resource_config_path)
    else:
        rc = resource_config
    if rc:
        hosts = _hosts_from_resource_config(rc, group)
        if hosts:
            return hosts
    return []


def dump_instance_group_debug(
    *,
    environ: Mapping[str, str] | None = None,
    resource_config_path: str | Path = DEFAULT_RESOURCE_CONFIG_PATH,
) -> str:
    """Compact topology dump for CloudWatch when host lookup fails."""
    env_map: Mapping[str, str] = os.environ if environ is None else environ
    env = _parse_json_object(env_map.get("SM_TRAINING_ENV", "") or "")
    rc_file = _load_resource_config_file(resource_config_path)
    payload = {
        "SM_CURRENT_HOST": env_map.get("SM_CURRENT_HOST"),
        "SM_CURRENT_INSTANCE_GROUP": env_map.get("SM_CURRENT_INSTANCE_GROUP"),
        "SM_HOSTS": env_map.get("SM_HOSTS"),
        "training_env.instance_groups": env.get("instance_groups"),
        "training_env.instance_groups_dict": env.get("instance_groups_dict"),
        "training_env.resource_config": env.get("resource_config"),
        "training_env.additional_framework_parameters": env.get(
            "additional_framework_parameters"
        ),
        "SM_RESOURCE_CONFIG": _parse_json_object(
            env_map.get("SM_RESOURCE_CONFIG", "") or ""
        )
        or None,
        "resourceconfig.json": rc_file or None,
        "resourceconfig.json_path": str(resource_config_path),
    }
    return json.dumps(payload, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--dump", "-d"):
        print(dump_instance_group_debug())
        return 0
    if not args:
        print(
            "usage: instance_group_hosts.py <group>|--dump",
            file=sys.stderr,
        )
        return 2
    hosts = resolve_instance_group_hosts(args[0])
    if not hosts:
        return 1
    print("\n".join(hosts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
