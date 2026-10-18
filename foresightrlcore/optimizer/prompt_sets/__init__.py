"""Built-in prompt sets registered on the ``PROMPT_SETS`` axis.

The YAML lives inside the package rather than in a repo ``examples/`` directory
so it travels with the wheel and into a job's staged source directory. Prompt
sets that keep their YAML outside a package must be given an
``optimizer.prompts_path`` pointing at S3 when running on ``aws_training_job``,
because the launcher ships packages, not the repo.
"""

from __future__ import annotations

from pathlib import Path

from foresightrlcore.optimizer.prompts import YamlPromptSet
from foresightrlcore.optimizer.registry import PROMPT_SETS

_PROMPT_SET_DIR = Path(__file__).resolve().parent

#: Pass-through single-prompt template. See the YAML for the contract.
SINGLE_V1_PATH = _PROMPT_SET_DIR / "single_v1.yaml"


@PROMPT_SETS.register("single_v1")
def _make_single_v1() -> YamlPromptSet:
    return YamlPromptSet(SINGLE_V1_PATH)


__all__ = ["SINGLE_V1_PATH"]
