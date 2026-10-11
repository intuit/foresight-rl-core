"""Prompt sets: load YAML templates and substitute placeholders.

Substitution uses ``str.replace`` (NOT ``str.format``), matching common
orchestrator patterns which do ``prompt.replace("{business_summary}", ...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from foresightrlcore.optimizer.registry import PROMPT_SETS


class YamlPromptSet:
    """A :class:`~foresightrlcore.optimizer.interfaces.PromptSet` backed by a YAML file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: Optional[Dict[str, Any]] = None

    @property
    def path(self) -> Path:
        """Where the YAML is read from.

        Exposed so validation can tell a prompt set that ships inside the wheel
        from one living in a repo ``examples/`` directory: only the latter has
        to be uploaded to S3 for a remote job, since the launcher stages the
        package but not the repo.
        """
        return self._path

    def _load(self) -> Dict[str, Any]:
        if self._data is None:
            if not self._path.exists():
                raise FileNotFoundError(f"prompts file not found: {self._path}")
            with self._path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"prompts file is not a mapping: {self._path}")
            self._data = loaded
        return self._data

    def template(self, section: str, key: str) -> str:
        data = self._load()
        sec = data.get(section)
        if not isinstance(sec, dict):
            raise KeyError(
                f"prompt section {section!r} not found; "
                f"available: {sorted(k for k in data)}"
            )
        if key not in sec or not isinstance(sec[key], str):
            raise KeyError(
                f"prompt key {section!r}.{key!r} not found; "
                f"available keys: {sorted(sec)}"
            )
        return sec[key]

    def render(self, section: str, key: str, **variables: Any) -> str:
        text = self.template(section, key)
        for name, value in variables.items():
            text = text.replace("{" + name + "}", "" if value is None else str(value))
        return text


def make_prompt_set(name: str, prompts_path: Optional[str] = None) -> YamlPromptSet:
    """Resolve a prompt set by registry name, honoring a path override."""
    if prompts_path:
        return YamlPromptSet(prompts_path)
    factory = PROMPT_SETS.get(name)
    result = factory()
    assert isinstance(result, YamlPromptSet)
    return result


__all__ = ["YamlPromptSet", "make_prompt_set"]
