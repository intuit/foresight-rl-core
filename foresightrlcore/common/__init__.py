"""Operation-agnostic plumbing shared across operations.

* :mod:`foresightrlcore.common.io` -- S3 transfer + JSONL checkpoint read/write.
* :mod:`foresightrlcore.common.retry` -- backoff + auth-refresh call wrapper.
* :mod:`foresightrlcore.common.sampling` -- deterministic stratified sampling.

These were promoted out of ``optimizer`` so ``evaluate`` can reuse them without
importing another operation.
"""
