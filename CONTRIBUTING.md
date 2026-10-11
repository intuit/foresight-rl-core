# Contributing

Contributions are welcome. This is a Python project. Issues and pull requests
are tracked on GitHub.

## Reporting bugs

Open a GitHub issue. Include the Python version, the command you ran, and the
traceback.

## What we expect in a change

- Tests cover new behavior. Run the suite with `uv run tox`.
- Coverage stays at or above 80%.
- Format with `black` and order imports with `isort`. `tox -e lint` also runs
  `mypy`.
- Explain why the change is needed in the pull request. Link the GitHub issue
  when there is one.

## Local setup

```shell
uv sync --all-extras --all-groups
uv run tox -e lint
uv run tox
```

## Pull requests

Fork the repository, branch from the default branch, and open a pull request.
Reviewers look for a clear description, passing checks, and tests for the
behavior you changed.
