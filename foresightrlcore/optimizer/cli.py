"""``rl-train optimizer`` / ``validate-optimizer`` subcommand implementations.

Kept thin: argument parsing + delegation to :class:`OptimizerOperation` and
:func:`foresightrlcore.config.optimizer_validation.validate_optimizer_config`.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("foresightrlcore.optimizer.cli")


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``optimizer`` and ``validate-optimizer`` subcommands."""
    run = subparsers.add_parser(
        "optimizer",
        help="Launch a GenAI optimizer run (quality + cost evaluation)",
    )
    run.add_argument("--config", required=True, help="Path to optimizer YAML")
    run.add_argument(
        "--wait",
        action="store_true",
        help="Block until the submitted job reaches a terminal state",
    )

    validate = subparsers.add_parser(
        "validate-optimizer",
        help="Validate an optimizer config without submitting",
    )
    validate.add_argument("--config", required=True, help="Path to optimizer YAML")


def dispatch(args: argparse.Namespace) -> int:
    """Route a parsed optimizer argv to the matching handler."""
    if args.command == "optimizer":
        return _cmd_optimizer(args)
    if args.command == "validate-optimizer":
        return _cmd_validate_optimizer(args)
    logger.error("Unknown optimizer subcommand: %r", args.command)
    return 1


def _cmd_validate_optimizer(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
        from foresightrlcore.config.optimizer_validation import (
            validate_optimizer_config,
        )

        config = OptimizerJobConfig.from_yaml(args.config)
        errors = validate_optimizer_config(config)
        if errors:
            logger.error("Optimizer config validation failed: %s", args.config)
            for err in errors:
                logger.error("  - %s", err)
            return 1

        run = config.optimizer
        logger.info("Optimizer config is valid: %s", args.config)
        logger.info("  Job:        %s", config.job.name)
        logger.info("  Prompt set: %s", run.prompt_set)
        logger.info(
            "  Models:     %s",
            ", ".join(f"{m.name}({m.provider})" for m in run.models),
        )
        logger.info("  Modes:      %s", ", ".join(run.modes))
        logger.info("  Judges:     %s", ", ".join(run.judge_names()))
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
        logger.error("Optimizer config validation failed: %s", exc)
        return 1


def _cmd_optimizer(args: argparse.Namespace) -> int:
    try:
        from foresightrlcore.optimizer import launch as launch_optimizer

        job_ids = launch_optimizer(args.config, wait=args.wait)
        for jid in job_ids:
            logger.info("Job ID: %s", jid)
        return 0
    except Exception as exc:  # noqa: BLE001 - user-facing error
        logger.error("Optimizer launch failed: %s", exc)
        return 1


__all__ = ["build_parser", "dispatch"]
