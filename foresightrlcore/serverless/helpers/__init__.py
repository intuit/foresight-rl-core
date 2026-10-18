"""Generic reusable helpers for serverless reward Lambdas.

Modules in this package are **stdlib + boto3 only** — no
training-stack imports (no transformers, torch, etc.). They are
intentionally importable from inside a Lambda zip without pulling the
rest of the framework.

* :mod:`foresightrlcore.serverless.helpers.bedrock_judge` — generic
  ``BedrockJudge`` wrapper around ``boto3.bedrock-runtime.converse``
  with retry policy and transient-error detection.
* :mod:`foresightrlcore.serverless.helpers.sample_io` — generic Lambda
  event parsing (extract messages / response / ground truth).
"""

from foresightrlcore.serverless.helpers.bedrock_judge import (  # noqa: F401
    DEFAULT_TRANSIENT_BEDROCK_ERRORS,
    BedrockJudge,
    JudgeError,
)
from foresightrlcore.serverless.helpers.sample_io import (  # noqa: F401
    coerce_batch,
    extract_assistant_content,
    extract_response,
    extract_state,
)

__all__ = [
    "BedrockJudge",
    "DEFAULT_TRANSIENT_BEDROCK_ERRORS",
    "JudgeError",
    "coerce_batch",
    "extract_assistant_content",
    "extract_response",
    "extract_state",
]
