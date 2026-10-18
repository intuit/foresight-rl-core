"""Dataset providers. Importing this package registers the built-in datasets."""

from foresightrlcore.optimizer.datasets import parquet_dataset  # noqa: F401

__all__ = ["parquet_dataset"]
