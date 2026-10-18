"""CLI entry point for rl-train.

Usage:
    rl-train validate --config path/to/config.yaml
    rl-train train --config path/to/config.yaml [--wait]
    rl-train perf-test --config path/to/perf.yaml [--wait]
    rl-train validate-perf --config path/to/perf.yaml
    rl-train optimizer --config path/to/optimizer.yaml [--wait]
    rl-train validate-optimizer --config path/to/optimizer.yaml
    rl-train evaluate --config path/to/evaluate.yaml [--wait]
    rl-train validate-evaluate --config path/to/evaluate.yaml
    rl-train list-benchmarks [--catalog PATH]
    rl-train setup-local-gpu --recipe <name> [--install] [--output PATH]
    rl-train setup-local-gpu --list
    rl-train serverless register --name <X> --role <ARN> [--source PATH] [--extra-file ...]
    rl-train serverless bundle --name <X> [--output PATH]
    rl-train serverless convert --input IN --output OUT [--schema rlvr|sft|dpo|kto]
                                {--converter NAME | --mapping YAML}
    rl-train serverless list [lambdas|datasets|converters|schemas|all]
    rl-train tokenize-pack --input DIR --tokenizer PATH --output DIR
                            [--pack-size N] [--data-format flat|messages]
                            [--enable-thinking] [--eval-split-size N]
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("foresightrlcore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rl-train",
        description="Multi-method, multi-provider training library",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a config file without launching"
    )
    validate_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )

    train_parser = subparsers.add_parser("train", help="Launch a training job")
    train_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    train_parser.add_argument(
        "--wait", action="store_true", help="Block until job completes"
    )

    setup_parser = subparsers.add_parser(
        "setup-local-gpu",
        help="Generate a pip requirements file from a local-GPU recipe",
    )
    setup_group = setup_parser.add_mutually_exclusive_group(required=True)
    setup_group.add_argument("--recipe", help="Recipe name (e.g. qwen3, qwen3_5)")
    setup_group.add_argument(
        "--list",
        action="store_true",
        dest="list_recipes",
        help="List available local-GPU recipes",
    )
    setup_parser.add_argument(
        "--install",
        action="store_true",
        help="Also run 'pip install -r' after generating the file",
    )
    setup_parser.add_argument(
        "--output",
        default="local-requirements.txt",
        help="Output file path (default: local-requirements.txt)",
    )

    from foresightrlcore.evaluate import cli as evaluate_cli
    from foresightrlcore.optimizer import cli as optimizer_cli
    from foresightrlcore.perf_test import cli as perf_test_cli
    from foresightrlcore.serverless import cli as serverless_cli
    from foresightrlcore.tokenizer import cli as tokenizer_cli

    perf_test_cli.build_parser(subparsers)
    optimizer_cli.build_parser(subparsers)
    evaluate_cli.build_parser(subparsers)
    serverless_cli.build_parser(subparsers)
    tokenizer_cli.build_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "validate":
        return _cmd_validate(args)
    elif args.command == "train":
        return _cmd_train(args)
    elif args.command == "setup-local-gpu":
        return _cmd_setup_local_gpu(args)
    elif args.command in ("perf-test", "validate-perf"):
        return perf_test_cli.dispatch(args)
    elif args.command in ("optimizer", "validate-optimizer"):
        return optimizer_cli.dispatch(args)
    elif args.command in ("evaluate", "validate-evaluate", "list-benchmarks"):
        return evaluate_cli.dispatch(args)
    elif args.command == "serverless":
        return serverless_cli.dispatch(args)
    elif args.command == "tokenize-pack":
        return tokenizer_cli.dispatch(args)

    return 0


def _cmd_validate(args) -> int:
    """Validate a config file and report errors."""
    try:
        from foresightrlcore.config.schema import TrainingJobConfig
        from foresightrlcore.config.validation import validate_config

        config = TrainingJobConfig.from_yaml(args.config)
        errors = validate_config(config)

        if errors:
            logger.error(f"Config validation failed: {args.config}")
            for err in errors:
                logger.error(f"  - {err}")
            return 1

        logger.info(f"Config is valid: {args.config}")
        logger.info(f"  Job:       {config.job.name}")
        logger.info(f"  Model:     {config.model.name}")
        logger.info(f"  Method:    {config.method.type}")
        logger.info(f"  Task:      {config.task.type if config.task else 'n/a'}")
        logger.info(f"  Execution: {config.execution.type}")
        return 0

    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return 1


def _cmd_train(args) -> int:
    """Launch a training job."""
    try:
        from foresightrlcore import train

        job_id = train(args.config, wait=args.wait)
        logger.info(f"Job ID: {job_id}")
        return 0
    except Exception as e:
        logger.error(f"Training launch failed: {e}")
        return 1


def _cmd_setup_local_gpu(args) -> int:
    """Generate (and optionally install) local-GPU requirements from a recipe."""
    import subprocess

    from foresightrlcore.config.recipes import (
        LOCAL_GPU_RECIPES,
        generate_local_gpu_requirements,
    )

    if args.list_recipes:
        logger.info("Available local-GPU recipes:")
        for name, recipe in sorted(LOCAL_GPU_RECIPES.items()):
            n_deps = len(recipe["requirements"])
            logger.info(f"  {name}  ({n_deps} dependencies)")
        return 0

    try:
        out_path = generate_local_gpu_requirements(args.recipe, args.output)
        logger.info(f"Wrote {out_path}")
    except ValueError as e:
        logger.error(str(e))
        return 1

    if args.install:
        logger.info(f"Installing from {out_path} ...")
        result = subprocess.run(
            ["pip", "install", "-r", str(out_path)],
        )
        if result.returncode != 0:
            logger.error("pip install failed")
            return 1
        logger.info("Installation complete")

    return 0
