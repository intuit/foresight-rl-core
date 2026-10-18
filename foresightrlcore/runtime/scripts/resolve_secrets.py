#!/usr/bin/env python3
"""Resolve secrets at container runtime.

``provider: local`` is handled here. ``provider: idps`` is delegated to the
foresightrl-intuit extension, which is staged with ``container.extra_packages``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("resolve_secrets")

ENV_FILE_DEFAULT = "/tmp/resolved_secrets.env"


def _resolve_local(secrets_cfg: dict, env_file: str) -> int:
    local_cfg = secrets_cfg.get("local", {})
    env_path = local_cfg["env_file"]
    data: dict[str, str] = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key.strip():
                data[key.strip()] = value.strip()

    resolved = 0
    with open(env_file, "w") as f:
        for entry in secrets_cfg.get("entries", []):
            env_var = entry["env_var"]
            secret_path = entry["secret_path"]
            if secret_path not in data:
                raise KeyError(f"Key '{secret_path}' not found in {env_path}")
            value = data[secret_path]
            f.write(f"export {env_var}={shlex.quote(value)}\n")
            os.environ[env_var] = value
            resolved += 1
    return resolved


def _resolve_idps(secrets_cfg: dict, env_file: str) -> int:
    """Delegate to the Intuit extension. The core does not import idps-sdk."""
    try:
        from foresightrl_intuit.auth.container_secrets import (
            resolve_idps_to_env_file,
        )
    except ImportError as exc:
        raise ImportError(
            "secrets.provider=idps requires the foresightrl-intuit extension. "
            "Set container.extra_packages: [foresightrl_intuit] on the job, "
            "or install it with: uv pip install -e intuit/"
        ) from exc
    return resolve_idps_to_env_file(secrets_cfg, env_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve secrets at runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-file", default=ENV_FILE_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        return 1

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)

        secrets_cfg = config.get("secrets")
        if not secrets_cfg or not secrets_cfg.get("entries"):
            logger.info("No secrets configured — skipping resolution.")
            return 0

        provider = secrets_cfg.get("provider", "local")
        if provider == "local":
            count = _resolve_local(secrets_cfg, args.env_file)
        elif provider == "idps":
            count = _resolve_idps(secrets_cfg, args.env_file)
        else:
            logger.error("Unsupported secrets provider: %s", provider)
            return 1

        logger.info("Secret resolution complete (%d secrets).", count)
        return 0
    except Exception:
        logger.error("Secret resolution failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
