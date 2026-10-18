"""``rl-train serverless`` subcommand implementations.

Four subcommands:

* ``register`` — get-or-create an AI Registry Evaluator for a registered
  reward Lambda (or an explicit ``--source`` path).
* ``bundle``   — produce a single-file Lambda bundle (the SageMaker
  Studio paste path).
* ``convert``  — generic JSONL converter; ``--schema`` is pluggable
  (today: ``rlvr``; reserved: ``sft``/``dpo``/``kto``).
* ``list``     — print registered lambdas, datasets, converters, and
  schemas.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("foresightrlcore.serverless.cli")


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``serverless`` subcommand group to an existing CLI."""
    parser = subparsers.add_parser(
        "serverless",
        help="Manage SageMaker Serverless reward Lambdas, datasets, converters",
    )
    sub = parser.add_subparsers(dest="serverless_command", help="Subcommands")

    # ---- register --------------------------------------------------------
    reg = sub.add_parser(
        "register",
        help="Get-or-create an AI Registry Evaluator for a reward Lambda",
    )
    reg.add_argument(
        "--name",
        required=True,
        help="Evaluator name = registry key (e.g. 'my-task-rlvr-judge')",
    )
    reg.add_argument(
        "--role",
        required=True,
        help="IAM role ARN used as the Lambda execution role",
    )
    reg.add_argument(
        "--source",
        help="Explicit path/S3 URI to a Lambda .py file. Overrides the "
        "registry lookup for one-off Lambdas not in the package.",
    )
    reg.add_argument(
        "--extra-file",
        action="append",
        default=[],
        dest="extra_files",
        help="Additional file to bundle into the Lambda zip (repeatable). "
        "Only valid alongside --source.",
    )

    # ---- bundle ----------------------------------------------------------
    bun = sub.add_parser(
        "bundle",
        help="Build a single-file Lambda for SageMaker Studio paste",
    )
    bun.add_argument(
        "--name",
        required=True,
        help="Registered Lambda name to bundle",
    )
    bun.add_argument(
        "--output",
        help="Output path for the bundle (default: stdout)",
    )

    # ---- convert ---------------------------------------------------------
    conv = sub.add_parser(
        "convert",
        help="Convert a JSONL dataset to a Serverless customization schema",
    )
    conv.add_argument("--input", required=True, help="Input JSONL file")
    conv.add_argument("--output", required=True, help="Output JSONL file")
    conv.add_argument(
        "--schema",
        default="rlvr",
        help="Output schema (default: rlvr; reserved: sft|dpo|kto)",
    )
    conv_source = conv.add_mutually_exclusive_group(required=True)
    conv_source.add_argument(
        "--converter",
        help="Registered converter name OR dotted path 'pkg.module:fn'",
    )
    conv_source.add_argument(
        "--mapping",
        help="Path to a YAML mapping spec (no-code path; see docs)",
    )

    # ---- list ------------------------------------------------------------
    lst = sub.add_parser(
        "list",
        help="List registered lambdas / datasets / converters / schemas",
    )
    lst.add_argument(
        "kind",
        nargs="?",
        choices=["lambdas", "datasets", "converters", "schemas", "all"],
        default="all",
        help="Which registry to list (default: all)",
    )

    return parser


def dispatch(args: argparse.Namespace) -> int:
    """Route a parsed ``serverless ...`` argv to the matching subcommand."""
    cmd = getattr(args, "serverless_command", None)
    if cmd is None:
        logger.error("Usage: rl-train serverless {register|bundle|convert|list} ...")
        return 1
    if cmd == "register":
        return _cmd_register(args)
    if cmd == "bundle":
        return _cmd_bundle(args)
    if cmd == "convert":
        return _cmd_convert(args)
    if cmd == "list":
        return _cmd_list(args)
    logger.error("Unknown serverless subcommand: %r", cmd)
    return 1


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def _cmd_register(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend
        from foresightrlcore.serverless import registry
    except ImportError as exc:
        logger.error("Failed to import serverless backend: %s", exc)
        return 1

    registry.discover_tasks()
    name = args.name

    # Build a tiny fake SL-config-like object that AwsServerlessBackend
    # already knows how to consume.
    sl = _FakeServerlessConfig(
        custom_reward_function_name=name,
        custom_reward_function_role=args.role,
        custom_reward_function_source=args.source,
        custom_reward_function_extra_files=list(args.extra_files or []),
    )

    if not args.source:
        if name not in registry.LAMBDAS:
            logger.error(
                "Lambda %r not registered and no --source given. " "Registered: %s",
                name,
                sorted(registry.LAMBDAS),
            )
            return 1

    try:
        from sagemaker import Session

        session = Session()
    except ImportError as exc:
        logger.error("Failed to create SageMaker session: %s", exc)
        return 1

    evaluator = AwsServerlessBackend._resolve_evaluator(sl, session)
    if evaluator is None:
        logger.error("Failed to resolve evaluator")
        return 1
    arn = getattr(evaluator, "arn", None) or getattr(evaluator, "evaluator_arn", None)
    logger.info("Evaluator ready: %s (ARN: %s)", name, arn or "<unknown>")
    return 0


class _FakeServerlessConfig:
    """Minimal shim for :meth:`AwsServerlessBackend._resolve_evaluator`."""

    def __init__(
        self,
        custom_reward_function_name: str,
        custom_reward_function_role: str,
        custom_reward_function_source: Optional[str],
        custom_reward_function_extra_files: list[str],
    ) -> None:
        self.custom_reward_function_name = custom_reward_function_name
        self.custom_reward_function_role = custom_reward_function_role
        self.custom_reward_function_source = custom_reward_function_source
        self.custom_reward_function_extra_files = custom_reward_function_extra_files


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


def _cmd_bundle(args: argparse.Namespace) -> int:
    from foresightrlcore.serverless import bundler, registry

    registry.discover_tasks()
    if args.name not in registry.LAMBDAS:
        logger.error(
            "Lambda %r not registered. Registered: %s",
            args.name,
            sorted(registry.LAMBDAS),
        )
        return 1
    text = bundler.build_bundle_from_name(args.name)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        logger.info("Wrote bundle: %s (%d bytes)", args.output, len(text))
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def _cmd_convert(args: argparse.Namespace) -> int:
    from foresightrlcore.serverless import convert, registry

    registry.discover_tasks()
    try:
        mapper = convert.resolve_mapper(
            converter=args.converter, mapping=args.mapping, schema=args.schema
        )
    except (ValueError, KeyError, NotImplementedError) as exc:
        logger.error("Failed to resolve converter: %s", exc)
        return 1

    summary = convert.convert_file(
        input_path=args.input,
        output_path=args.output,
        mapper=mapper,
        schema=args.schema,
    )
    if summary.total_out == 0 and summary.total_in > 0:
        logger.error(
            "All %d input records were skipped — see log for per-line reasons",
            summary.total_in,
        )
        return 2
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    from foresightrlcore.serverless import registry
    from foresightrlcore.serverless.spec import (
        available_schemas,
        reserved_schemas,
    )

    registry.discover_tasks()
    kinds = (
        ["lambdas", "datasets", "converters", "schemas"]
        if args.kind == "all"
        else [args.kind]
    )
    for kind in kinds:
        print(f"\n[{kind}]")
        if kind == "lambdas":
            for name, lambda_spec in sorted(registry.LAMBDAS.items()):
                print(f"  {name}")
                print(f"      module : {lambda_spec.main_module}")
                if lambda_spec.extra_modules:
                    print(f"      extras : {', '.join(lambda_spec.extra_modules)}")
                if lambda_spec.env_defaults:
                    keys = ", ".join(sorted(lambda_spec.env_defaults))
                    print(f"      env    : {keys}")
                if lambda_spec.description:
                    print(f"      desc   : {lambda_spec.description}")
        elif kind == "datasets":
            for name, dataset_spec in sorted(registry.DATASETS.items()):
                print(f"  {name}")
                if dataset_spec.description:
                    print(f"      desc   : {dataset_spec.description}")
        elif kind == "converters":
            for name, converter_spec in sorted(registry.CONVERTERS.items()):
                print(f"  {name}")
                print(f"      mapper : {converter_spec.mapper}")
                print(f"      schemas: {sorted(converter_spec.schemas)}")
                if converter_spec.description:
                    print(f"      desc   : {converter_spec.description}")
        elif kind == "schemas":
            for name in available_schemas():
                print(f"  {name}  (implemented)")
            for name in reserved_schemas():
                print(f"  {name}  (reserved)")
    return 0


__all__ = ["build_parser", "dispatch"]
