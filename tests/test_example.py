from foresightrlcore import train


def test_train_function_exists():
    """Verify the public train() API is importable."""
    assert callable(train)
