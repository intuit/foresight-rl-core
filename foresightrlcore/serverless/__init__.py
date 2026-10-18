"""Generic helpers for SageMaker Serverless reward Lambdas + dataset converters.

Peer of :mod:`foresightrlcore.tasks`. Holds **only framework-grade**
code shared across tasks — no task-specific prompts, criteria, or
field mappings. Per-task serverless code lives under
``foresightrlcore.tasks.<task>.lambda_reward`` and
``foresightrlcore.tasks.<task>.convert``.

Submodules
----------

* :mod:`foresightrlcore.serverless.registry` — three flat dicts
  (``LAMBDAS``, ``DATASETS``, ``CONVERTERS``) populated at task-package
  import time. Lookup helpers used by the backend, the bundler, and the
  generic CLI.
* :mod:`foresightrlcore.serverless.spec` — pydantic output schemas
  (today: :class:`RLVRRecord`; reserved keys for SFT/DPO/KTO) plus the
  :data:`Mapper` typedef.
* :mod:`foresightrlcore.serverless.bundler` — generalized AST bundler
  for the SageMaker Studio single-file paste path.
* :mod:`foresightrlcore.serverless.convert` — generic JSONL converter
  loop powering ``rl-train serverless convert``.
* :mod:`foresightrlcore.serverless.helpers` — reusable
  :class:`BedrockJudge` + sample-IO primitives.
"""

from foresightrlcore.serverless import bundler, convert, helpers, registry, spec

__all__ = ["bundler", "convert", "helpers", "registry", "spec"]
