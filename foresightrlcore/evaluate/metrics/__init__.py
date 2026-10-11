"""Metrics. Importing this package registers the built-in metrics.

Registered names: ``exact_match``, ``substring_match``, ``choice_match``,
``extraction_match``, ``llm_judge``. Adding one is a class with a ``score``
method plus ``@METRICS.register("name")``.
"""

from foresightrlcore.evaluate.metrics import (  # noqa: F401
    aggregate,
    choice_match,
    extraction,
    judge,
    text_match,
)

__all__ = ["aggregate", "choice_match", "extraction", "judge", "text_match"]
