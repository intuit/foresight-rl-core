"""``rl-train evaluate`` / ``validate-evaluate`` / ``list-benchmarks``.

Kept thin: argument parsing plus delegation to :class:`EvaluateOperation` and
:func:`~foresightrlcore.config.evaluate_validation.validate_evaluate_config`.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("foresightrlcore.evaluate.cli")


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the evaluate subcommands."""
    run = subparsers.add_parser(
        "evaluate",
        help="Run benchmarks against candidate models and write a report",
    )
    run.add_argument("--config", required=True, help="Path to evaluate YAML")
    run.add_argument(
        "--wait",
        action="store_true",
        help="Block until the submitted job reaches a terminal state",
    )

    validate = subparsers.add_parser(
        "validate-evaluate",
        help="Validate an evaluate config without submitting",
    )
    validate.add_argument("--config", required=True, help="Path to evaluate YAML")

    listing = subparsers.add_parser(
        "list-benchmarks",
        help="List the benchmarks and pluggable axes available to evaluate",
    )
    listing.add_argument(
        "--catalog",
        action="append",
        default=[],
        help="Extra catalog YAML to include (repeatable).",
    )


def dispatch(args: argparse.Namespace) -> int:
    """Route a parsed evaluate argv to the matching handler."""
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "validate-evaluate":
        return _cmd_validate(args)
    if args.command == "list-benchmarks":
        return _cmd_list(args)
    logger.error("Unknown evaluate subcommand: %r", args.command)
    return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
        from foresightrlcore.config.evaluate_validation import validate_evaluate_config

        config = EvaluateJobConfig.from_yaml(args.config)
        errors = validate_evaluate_config(config)
        if errors:
            logger.error("Evaluate config validation failed: %s", args.config)
            for err in errors:
                logger.error("  - %s", err)
            return 1

        run = config.evaluate
        logger.info("Evaluate config is valid: %s", args.config)
        logger.info("  Job:        %s", config.job.name)
        logger.info("  Benchmarks: %s", ", ".join(run.benchmark_names()))
        logger.info(
            "  Models:     %s",
            ", ".join(f"{m.name}({m.provider})" for m in run.models),
        )
        logger.info("  Trials:     %d", run.trials)
        logger.info(
            "  Execution:  %s (instance=%s, gpu=%s)",
            config.execution.type,
            config.resolved_instance_type(),
            config.requires_gpu(),
        )
        return 0
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        return 1
    except Exception as exc:  # noqa: BLE001 - user-facing error
        logger.error("Evaluate config validation failed: %s", exc)
        return 1


def _cmd_list(args: argparse.Namespace) -> int:
    """Print the catalog and the four registries.

    This is the answer to "what can I run, and what can I plug in" -- including
    anything an installed extension contributed.
    """
    try:
        from foresightrlcore.evaluate.catalog import load_catalog
        from foresightrlcore.evaluate.registry import (
            BENCHMARK_TYPES,
            DATASET_LOADERS,
            METRICS,
            REPORTERS,
        )
        from foresightrlcore.plugins import load_evaluate_plugins

        load_evaluate_plugins()
        catalog = load_catalog(args.catalog)

        logger.info("Benchmarks (%d):", len(catalog))
        for name, spec in sorted(catalog.items()):
            logger.info(
                "  %-28s %-16s %s",
                name,
                spec.type or "mcq",
                spec.display_name or "",
            )
        logger.info("Benchmark types:  %s", ", ".join(BENCHMARK_TYPES.list()))
        logger.info("Dataset loaders:  %s", ", ".join(DATASET_LOADERS.list()))
        logger.info("Metrics:          %s", ", ".join(METRICS.list()))
        logger.info("Reporters:        %s", ", ".join(REPORTERS.list()))
        return 0
    except Exception as exc:  # noqa: BLE001 - user-facing error
        logger.error("Could not list benchmarks: %s", exc)
        return 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.evaluate import launch as launch_evaluate

        for job_id in launch_evaluate(args.config, wait=args.wait):
            logger.info("Job ID: %s", job_id)
        return 0
    except Exception as exc:  # noqa: BLE001 - user-facing error
        logger.error("Evaluate launch failed: %s", exc)
        return 1


__all__ = ["build_parser", "dispatch"]
