"""foresightrlcore.auth — runtime secret resolution."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SecretsProvider(Protocol):
    """Interface that all secret providers must implement."""

    def resolve(self, secret_path: str) -> str:
        """Resolve a single secret path to its plaintext value."""
        ...


_PROVIDERS: Dict[str, type] = {}
_PROVIDER_FACTORIES: Dict[str, Callable[[Dict[str, Any]], "SecretsProvider"]] = {}


def register_provider(name: str, provider_cls: type) -> None:
    _PROVIDERS[name] = provider_cls


def register_provider_factory(
    name: str, factory: Callable[[Dict[str, Any]], "SecretsProvider"]
) -> None:
    """Register a callable ``(secrets_config) -> SecretsProvider``."""
    _PROVIDER_FACTORIES[name] = factory


def get_provider(name: str) -> type:
    if name not in _PROVIDERS:
        available = sorted(set(_PROVIDERS) | set(_PROVIDER_FACTORIES))
        raise KeyError(f"Unknown secrets provider '{name}'. Available: {available}")
    return _PROVIDERS[name]


def create_provider(secrets_config: Dict[str, Any]) -> SecretsProvider:
    """Instantiate a SecretsProvider from a parsed secrets config dict."""
    from foresightrlcore.plugins import load_plugins

    load_plugins()

    provider_name = secrets_config.get("provider", "local")
    if provider_name in _PROVIDER_FACTORIES:
        factory = _PROVIDER_FACTORIES[provider_name]
        return factory(secrets_config)

    provider_cls = get_provider(provider_name)

    if provider_name == "local":
        local_cfg = secrets_config.get("local", {})
        return provider_cls(env_file=local_cfg["env_file"])

    raise ValueError(
        f"Unsupported secrets provider: {provider_name!r}. "
        f"Install the foresightrl-intuit extension for IDPS support."
    )


def resolve_secrets(config_path: str) -> int:
    """Read secrets config from a YAML file and resolve all entries as env vars."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    secrets_cfg = config.get("secrets")
    if not secrets_cfg or not secrets_cfg.get("entries"):
        logger.info("No secrets configured — skipping resolution.")
        return 0

    provider = create_provider(secrets_cfg)
    resolved = 0

    for entry in secrets_cfg["entries"]:
        env_var = entry["env_var"]
        secret_path = entry["secret_path"]

        try:
            value = provider.resolve(secret_path)
            os.environ[env_var] = value
            logger.info("Resolved secret -> %s (length: %d)", env_var, len(value))
            resolved += 1
        except Exception:
            logger.error(
                "Failed to resolve secret '%s' for env var '%s'",
                secret_path,
                env_var,
                exc_info=True,
            )
            raise

    logger.info(
        "Resolved %d secret(s) from %s provider.", resolved, secrets_cfg["provider"]
    )
    return resolved


# Auto-register built-in local provider (OSS core).
import foresightrlcore.auth.local  # noqa: E402, F401
