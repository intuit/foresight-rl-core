"""Reporters. Importing this package registers the built-in ones.

Registered names: ``table`` (parquet + leaderboard CSV + markdown) and ``json``
(the self-contained manifest), both run by default, plus ``confusion``
(precision / recall with slice breakdowns, opt-in).

A reporter is one class with ``report(result, output_dir)`` plus
``@REPORTERS.register("name")``. Sinks that talk to an internal service belong
in an extension package, not here.
"""

from foresightrlcore.evaluate.reporters import (  # noqa: F401
    confusion,
    json_reporter,
    table,
)

__all__ = ["confusion", "json_reporter", "table"]
