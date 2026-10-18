---
name: code-quality-workflow
description: >-
  Run the full code quality and test suite using uv run tox.
  Validates lint (isort, black, mypy) and tests (pytest with coverage
  across py311, py312, py313). Iteratively fixes failures with minimal
  changes until all checks pass. Use when the user asks to run tests,
  run tox, check code quality, validate the service, run the full
  test suite, or after making any code change.
---

# Code Quality Workflow

Run `uv run tox` and iteratively fix failures until all checks pass.

## Execution

```bash
cd /Users/ygazman/dev/foresight-rl-core && source .venv/bin/activate && uv run tox
```

**What tox runs:**
- **lint env**: isort (imports) → black (formatting) → mypy (types)
- **py311/py312/py313 envs**: pytest with coverage (361+ tests)

**Exit code 0** = all passed. Non-zero = failures to fix.

## Quick Lint-Only Check

```bash
cd /Users/ygazman/dev/foresight-rl-core && source .venv/bin/activate && uv run tox -e lint
```

## Error Fixing Workflow

When tox fails:

1. **Identify** which environment failed (lint or pyXXX) and which tool
2. **Apply minimal fixes only:**
   - isort fails → auto-fix: `source .venv/bin/activate && uv run isort .`
   - black fails → auto-fix: `source .venv/bin/activate && uv run black .`
   - mypy fails → fix type annotations in reported files
   - pytest fails → fix failing tests or the code causing failures
3. **Rerun `uv run tox` completely** after every fix
4. **Repeat** until exit code 0

### Fix Rules

- Fix ONLY what's broken
- Make minimal, surgical changes
- No new features, refactoring, optimizations, or docs during fixing
- Preserve existing functionality
- Fix one error category at a time (isort → black → mypy → pytest)

## Success Criteria

- `uv run tox` exits with code 0
- All lint checks pass (isort, black, mypy)
- All pytest tests pass across py311, py312, py313
- Coverage reports generated

## Reporting

After completion, provide:
- Total iterations required
- Errors fixed (file names and error types)
- Final tox output status
- Test count and coverage percentage
- Confirmation no unnecessary changes were made

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `black --check --diff` reformats a file | Line length or formatting mismatch | Run `uv run black .` to auto-fix |
| `isort --check-only` shows diff | Import order wrong | Run `uv run isort .` to auto-fix |
| mypy "incompatible type" | Wrong type annotation or missing stub | Fix the annotation or add `# type: ignore[code]` with justification |
| pytest collection error | Import error or missing fixture | Check imports and conftest.py |
