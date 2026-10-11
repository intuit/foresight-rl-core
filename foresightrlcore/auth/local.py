"""Local secrets provider — reads secrets from a .env file.

Intended for local GPU runs where IDPS is not available.
The .env file uses KEY=VALUE format (one per line), with ``#`` comments
and blank lines ignored.
"""

from __future__ import annotations

import logging

from foresightrlcore.auth import register_provider

logger = logging.getLogger(__name__)


class LocalFileProvider:
    """Reads a .env file (KEY=VALUE per line) and resolves secret paths as keys."""

    def __init__(self, env_file: str) -> None:
        self._data: dict[str, str] = {}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key:
                    self._data[key] = value
        logger.info(
            "Loaded %d key(s) from local env file: %s", len(self._data), env_file
        )

    def resolve(self, secret_path: str) -> str:
        if secret_path not in self._data:
            raise KeyError(
                f"Key '{secret_path}' not found in env file. "
                f"Available keys: {list(self._data.keys())}"
            )
        return self._data[secret_path]


register_provider("local", LocalFileProvider)
