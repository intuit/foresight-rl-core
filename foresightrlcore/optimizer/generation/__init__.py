"""Backwards-compatible alias for :mod:`foresightrlcore.generation`.

The generation clients were promoted to the package level so ``evaluate`` can
use them without importing ``optimizer``. Only the public factory is re-exported
here; import the clients themselves from :mod:`foresightrlcore.generation`.
"""

from foresightrlcore.generation import make_generation_client

__all__ = ["make_generation_client"]
