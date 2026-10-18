"""``rl-train perf-test`` / ``validate-perf`` subcommand implementations.

Kept thin: argument parsing + delegation to :mod:`PerfTestOperation` and
:mod:`foresightrlcore.config.perf_validation`.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("foresightrlcore.perf_test.cli")


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``perf-test`` and ``validate-perf`` subcommands."""
    perf = subparsers.add_parser(
        "perf-test",
        help="Launch a performance-test run",
    )
    perf.add_argument("--config", required=True, help="Path to perf-test YAML")
    perf.add_argument(
        "--wait",
        action="store_true",
        help="Block until the submitted job reaches a terminal state",
    )

    validate = subparsers.add_parser(
        "validate-perf",
        help="Validate a perf-test config without submitting",
    )
    validate.add_argument("--config", required=True, help="Path to perf-test YAML")


def dispatch(args: argparse.Namespace) -> int:
    """Route a parsed perf-test argv to the matching handler."""
    if args.command == "perf-test":
        return _cmd_perf_test(args)
    if args.command == "validate-perf":
        return _cmd_validate_perf(args)
    logger.error("Unknown perf-test subcommand: %r", args.command)
    return 1


def _cmd_validate_perf(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.config.perf_schema import PerfTestJobConfig
        from foresightrlcore.config.perf_validation import validate_perf_config

        config = PerfTestJobConfig.from_yaml(args.config)
        errors = validate_perf_config(config)
        if errors:
            logger.error("Perf-test config validation failed: %s", args.config)
            for err in errors:
                logger.error("  - %s", err)
            return 1

        run = config.perf_test
        logger.info("Perf-test config is valid: %s", args.config)
        logger.info("  Job:      %s", config.job.name)
        logger.info("  Model:    %s", config.model.name)
        logger.info(
            "  Run:      %s (instance=%s, tps_levels=%s)",
            run.name,
            run.instance_type,
            run.gatling.tps_levels,
        )
        return 0
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        return 1
    except Exception as exc:  # noqa: BLE001 -- user-facing error
        logger.error("Perf-test config validation failed: %s", exc)
        return 1


def _cmd_perf_test(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.perf_test import launch as launch_perf_test

        job_ids = launch_perf_test(args.config, wait=args.wait)
        for jid in job_ids:
            logger.info("Job ID: %s", jid)
        return 0
    except Exception as exc:  # noqa: BLE001 -- user-facing error
        logger.error("Perf-test launch failed: %s", exc)
        return 1


__all__ = ["build_parser", "dispatch"]
