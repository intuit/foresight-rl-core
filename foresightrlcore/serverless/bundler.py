"""Single-file Lambda bundler for the SageMaker Studio paste path.

WHY THIS EXISTS
~~~~~~~~~~~~~~~

A serverless reward Lambda is typically a small package of files
(``lambda_function.py`` + helper modules + a stdlib-only shared core).
In production we ship those as a multi-file zip via
:meth:`AwsServerlessBackend._create_lambda_with_extras`, which calls
``boto3.client("lambda").create_function`` directly. That requires
``lambda:CreateFunction`` permission on the executor; some account
contexts (Studio role) have it, some local SSO sessions do not.

When a user must register through Studio's UI, that UI takes only a
**single Python file** as the Lambda code. This module generalizes the
former ``serverless/advice/bundle_lambda.py`` to take any
:class:`LambdaSpec` and produce one self-contained ``.py`` file that
can be pasted in.

HOW IT WORKS
~~~~~~~~~~~~

For each module in the spec (``extra_modules`` first, then
``main_module``):

1. Parse the file with ``ast`` to locate the boundary between the
   module's top-of-file imports and its body.
2. Drop everything between ``# BUNDLER_STRIP_BEGIN`` and
   ``# BUNDLER_STRIP_END`` markers (the cross-module
   ``try/except ImportError`` blocks — unnecessary in the bundled
   output because all modules' top-level names live in one namespace).
3. Concatenate remaining bodies in dependency order.
4. Deduplicate module-level imports and emit them once at the top.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from foresightrlcore.serverless.registry import LambdaSpec, resolve_lambda

STRIP_BEGIN = "# BUNDLER_STRIP_BEGIN"
STRIP_END = "# BUNDLER_STRIP_END"


def build_bundle_from_name(name: str) -> str:
    """Build a single-file Lambda bundle for the registered lambda ``name``."""
    main_path, extra_paths = resolve_lambda(name)
    return build_bundle(main_path=main_path, extra_paths=extra_paths)


def build_bundle_from_spec(spec: LambdaSpec) -> str:
    """Build a single-file bundle for a :class:`LambdaSpec`."""
    return build_bundle_from_name(spec.name)


def build_bundle(main_path: str, extra_paths: List[str]) -> str:
    """Build a single-file bundle from explicit file paths.

    The bundler concatenates ``extra_paths`` first (in given order),
    then ``main_path``. This matches dependency order when each extra is
    a helper imported by the main file.
    """
    import_blocks: List[List[str]] = []
    bodies: List[Tuple[str, str]] = []

    sources: List[Tuple[str, Path]] = [
        (Path(p).stem, Path(p)) for p in (*extra_paths, main_path)
    ]
    for name, path in sources:
        text = path.read_text()
        text = _strip_marker_blocks(text)
        imports, body = _split_imports_and_body(text)
        body = _strip_module_docstring(body)
        import_blocks.append(imports)
        bodies.append((name, body))

    header = _make_header(sources)
    deduped_imports = _dedupe_imports(import_blocks)

    parts = [header, "\n", deduped_imports, "\n"]
    for name, body in bodies:
        parts.append("\n# " + "=" * 75 + "\n")
        parts.append(f"# {name}\n")
        parts.append("# " + "=" * 75 + "\n\n")
        parts.append(body.lstrip("\n"))
        if not body.endswith("\n"):
            parts.append("\n")
    return "".join(parts)


def _strip_marker_blocks(source: str) -> str:
    """Remove every ``# BUNDLER_STRIP_BEGIN ... # BUNDLER_STRIP_END`` block."""
    pattern = re.compile(
        rf"^[ \t]*{re.escape(STRIP_BEGIN)}.*?^[ \t]*{re.escape(STRIP_END)}[^\n]*\n?",
        flags=re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", source)


def _split_imports_and_body(source: str) -> Tuple[List[str], str]:
    """Return ``(top_level_import_lines, body_text)`` for one module."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    import_line_ranges: List[Tuple[int, int]] = []
    other_top_level: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            import_line_ranges.append((start, end))
        else:
            other_top_level.append(node)

    import_line_set = {
        line for start, end in import_line_ranges for line in range(start, end)
    }
    import_lines = [lines[i] for i in sorted(import_line_set)]
    body_lines = [lines[i] for i in range(len(lines)) if i not in import_line_set]
    return import_lines, "".join(body_lines)


def _normalise_import(line: str) -> str:
    """Canonical form for dedupe — collapse whitespace, strip comments."""
    no_comment = re.sub(r"\s*#.*$", "", line.rstrip())
    return re.sub(r"\s+", " ", no_comment).strip()


def _dedupe_imports(import_blocks: Iterable[List[str]]) -> str:
    """Deduplicate import lines across all source files, preserving order."""
    seen: set[str] = set()
    out_lines: List[str] = []
    for block in import_blocks:
        for line in block:
            key = _normalise_import(line)
            if not key or key in seen:
                continue
            seen.add(key)
            out_lines.append(line.rstrip("\n") + "\n")
    return "".join(out_lines)


def _strip_module_docstring(body: str) -> str:
    """Remove the module-level docstring (a leading triple-quoted string)."""
    tree = ast.parse(body)
    if not tree.body:
        return body
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        lines = body.splitlines(keepends=True)
        end = first.end_lineno or first.lineno
        return "".join(lines[end:])
    return body


def _make_header(sources: List[Tuple[str, Path]]) -> str:
    listing = "\n".join(f"#   - {path.name}" for _, path in sources)
    return f'''"""GENERATED FILE — DO NOT EDIT BY HAND.

Single-file bundle of a serverless reward Lambda for pasting into the
SageMaker Studio evaluator code editor. Regenerate with::

    rl-train serverless bundle --name <registered-lambda-name> [--output PATH]

Sources are concatenated in dependency order, cross-module
``try/except ImportError`` shims (delimited by the bundler's strip
markers in the original sources) are removed, and module-level
imports are deduplicated:

{listing}
"""
'''


__all__ = [
    "STRIP_BEGIN",
    "STRIP_END",
    "build_bundle",
    "build_bundle_from_name",
    "build_bundle_from_spec",
]
