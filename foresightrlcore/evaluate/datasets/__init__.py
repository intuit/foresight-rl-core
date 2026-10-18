"""Dataset loaders. Importing this package registers the built-in loaders.

Four formats cover everything the core needs:

* ``csv`` / ``jsonl`` / ``parquet`` -- tabular records, one row per question
* ``scenario_dir`` -- one *directory* per record, for agentic benchmarks that
  grade filesystem artifacts rather than text
* ``inline`` -- records written directly in the YAML, for smoke tests
"""

from foresightrlcore.evaluate.datasets import (  # noqa: F401
    inline_loader,
    scenario_dir_loader,
    tabular_loader,
)

__all__ = ["inline_loader", "scenario_dir_loader", "tabular_loader"]
