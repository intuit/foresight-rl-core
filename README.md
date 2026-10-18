# Python Package: `foresight-rl-core`

![Supported Python versions](https://shields.io/badge/python-3.10_|_3.11_|_3.12_|_3.13-green?logo=python)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A modular library for fine-tuning language models using reinforcement learning methods (GRPO, DPO) across multiple execution backends (local GPU, AWS SageMaker, and more).

## Quick Start

```bash
# Validate a SageMaker config (no AWS calls)
rl-train validate --config examples/configs/oss_quickstart.yaml

# Validate the local GPU config. The sample dataset is in the repo.
# Training itself runs on a machine with an NVIDIA GPU:
rl-train validate --config examples/configs/local_example.yaml
rl-train train --config examples/configs/local_example.yaml
```

## Installation

### uv (recommended)

```shell
uv add foresightrlcore
```

### pip

```shell
pip install foresightrlcore
```

### Optional extras

```bash
pip install "foresightrlcore[aws]"        # + AWS SageMaker support
pip install "foresightrlcore[optimizer]"  # + optimizer datasets
```

For local GPU training, install the base package and generate GPU deps from a recipe:

```bash
pip install foresightrlcore
rl-train setup-local-gpu --recipe qwen3_5   # generates local-requirements.txt
pip install -r local-requirements.txt
```

### HyperPod EKS backend

The `aws_hyperpod_eks` execution backend requires `kubectl`:

```bash
# macOS
brew install kubernetes-cli

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

Also install the [AWS Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) for the SSM tunnel:

```bash
# macOS
brew install --cask session-manager-plugin
```

See `docs/launching-hyperpod-eks-training-job.md` for the per-job
launch reference, and `intuit/docs/hyperpod-eks-setup-notes.md` for the
one-time cluster-bootstrap setup (IAM roles, FSx, EKS add-ons, etc.).

## Configuration

All training parameters are defined in YAML config files. See `examples/configs/` for templates:

| Config | Description |
|--------|-------------|
| `oss_quickstart.yaml` | GRPO on SageMaker. Replace `s3://your-bucket` before launching |
| `local_example.yaml` | GRPO on one local GPU. Sample JSONL is in the repo; no AWS account |
| `sagemaker_serverless_example.yaml` | Serverless model customization (DPO/RLVR/RLAIF) |
| `perf_test_example.yaml` | Gatling load test against `vllm serve` (TPS sweep, optional speculative decoding) |
| `optimizer_example.yaml` | GenAI optimizer: quality + cost evaluation across candidate LLMs (`genos`/`bedrock`/`local`) |
| `evaluate_example.yaml` | Benchmark suites × candidate models → leaderboard (catalog entries, trials, confidence intervals) |

### Container Recipes (SageMaker)

Recipes are named presets that resolve the DLC image, framework version, and pip dependencies for a model family. Set `container.recipe` in your config instead of managing versions manually:

```yaml
container:
  recipe: qwen3_5              # resolves framework_version, image_uri, and pip deps
  # recipe: qwen3              # alternative for Qwen3-14B era stack
```

Override any recipe default by setting fields explicitly — `image_uri` overrides the image, `requirements` overrides pip deps:

```yaml
container:
  recipe: qwen3_5
  image_uri: "your-custom-ecr-image:latest"   # takes precedence over recipe image
  requirements:                                 # takes precedence over recipe deps
    - "trl>=1.3.0"
```

Available container recipes: `qwen3`, `qwen3_5`. See [Container Recipes](docs/launching-sagemaker-training-job.md#container-recipes) for details.

### Local GPU Setup

For local GPU machines, generate a pinned requirements file from a recipe:

```bash
# List available recipes
rl-train setup-local-gpu --list

# Generate requirements file
rl-train setup-local-gpu --recipe qwen3_5
# → writes local-requirements.txt

# Install
pip install -r local-requirements.txt
```

## Usage

### CLI

```bash
# Validate a config before launching
rl-train validate --config examples/configs/oss_quickstart.yaml

# Launch a SageMaker training job
rl-train train --config examples/configs/oss_quickstart.yaml

# Local GPU, from the repo root. Dataset is examples/datasets/email_outcome/train.jsonl
rl-train train --config examples/configs/local_example.yaml

# Launch and block until the job completes
rl-train train --config examples/configs/oss_quickstart.yaml --wait

# Generate local GPU requirements from a recipe
rl-train setup-local-gpu --recipe qwen3_5
```

### Performance Testing

```bash
# Validate a perf-test config (no GPU needed)
rl-train validate-perf --config examples/configs/perf_test_example.yaml

# Launch a Gatling load test (vllm serve + TPS sweep in a SageMaker container)
AWS_DEFAULT_REGION=us-west-2 rl-train perf-test --config examples/configs/perf_test_example.yaml
```

### Optimizer (GenAI quality + cost evaluation)

```bash
# Validate an optimizer config (no GPU needed)
rl-train validate-optimizer --config examples/configs/optimizer_example.yaml

# Launch a batch (dataset × model × mode) generate→judge→cost evaluation
AWS_DEFAULT_REGION=us-west-2 rl-train optimizer --config examples/configs/optimizer_example.yaml
```

### Evaluate (benchmarks → leaderboard)

```bash
# What benchmarks, metrics, loaders and reporters are registered?
rl-train list-benchmarks

# Validate an evaluate config (no GPU needed)
rl-train validate-evaluate --config examples/configs/evaluate_example.yaml

# Launch a (benchmark × model) cross-product and write a leaderboard
AWS_DEFAULT_REGION=us-west-2 rl-train evaluate --config examples/configs/evaluate_example.yaml
```

Every candidate is served as an OpenAI-compatible endpoint, so a benchmark
living in its own repository runs unmodified — see
[the extension contract](docs/evaluate-extension-contract.md).

### Python API

```python
from foresightrlcore import train

job_id = train(config_path="path/to/config.yaml")
```

## Architecture

```
foresightrlcore/
  cli.py              # CLI entry point (train, validate, perf-test, optimizer, evaluate, list-benchmarks, ...)
  config/             # Pydantic schema, profiles, validation
  config/recipes.py   # Container and local-GPU recipe registries
  operations/         # Operation protocol (train, perf_test, optimizer, evaluate; future: test, deploy)
  methods/            # Method handlers (grpo, future: sft, dpo)
  execution/          # Execution backends (aws_training_job, aws_serverless, local)
  tasks/              # Task plugins (reward functions, callbacks)
  inference/          # Model-as-endpoint: any provider behind one OpenAI-compatible URL
  perf_test/          # Perf-test operation: vllm serve + Gatling load tests
  optimizer/          # Optimizer operation: GenAI quality + cost evaluation
  evaluate/           # Evaluate operation: benchmarks × models → leaderboard
  runtime/scripts/    # Container-side training, perf-test, optimizer and evaluate drivers
```

### Extensibility

- **New method**: Subclass `TrlMethodHandler` in `foresightrlcore/methods/<name>.py`, add method-specific validation and trainer args, call `register_method()`. The base class handles the full training pipeline (task loading, datasets, LoRA, hooks, train/save).
- **New backend**: Add `foresightrlcore/execution/<name>.py` implementing `ExecutionBackend`
- **New task**: Add `foresightrlcore/tasks/<name>.py` with `get_reward_function`, `get_callbacks`, `get_dataset_loader`
- **New operation**: Add `foresightrlcore/operations/<name>.py` implementing `Operation`
- **New recipe**: Add an entry to `CONTAINER_RECIPES` or `LOCAL_GPU_RECIPES` in `foresightrlcore/config/recipes.py`, add a test
- **Custom training behavior**: Use `training.hooks` for callbacks, pre/post hooks, or custom trainer subclasses
- **Custom container image**: Set `container.image_uri` to a fully-qualified ECR URI to bypass DLC image resolution

## Documentation

| Guide | Description |
|-------|-------------|
| [SageMaker Training Jobs](docs/launching-sagemaker-training-job.md) | Config reference for SageMaker training jobs (recipes, custom images, topology) |
| [Serverless Jobs](docs/launching-serverless-job.md) | Config reference for serverless model customization (DPO/RLVR/RLAIF) |
| [Local GPU Training](docs/launching-local-job.md) | Config reference for local GPU execution (setup, recipes, torchrun) |
| [Performance Testing](docs/launching-performance-test.md) | Gatling load tests against `vllm serve` — TPS sweep, speculative decoding, JSONL input |
| [Optimizer](docs/launching-optimizer.md) | GenAI quality (gates + LLM-as-judge) + cost evaluation across candidate LLMs; pluggable datasets/prompts/modes/judges |
| [Evaluate](docs/launching-evaluate.md) | Benchmark suites against candidate models; two benchmark tiers, a benchmark catalog, trials with confidence intervals |
| [Evaluate Extension Contract](docs/evaluate-extension-contract.md) | What a benchmark authored in another repository can rely on: endpoint contract, both benchmark protocols, result shape |
| [Executive Summary](docs/rl-train-summary.md) | High-level overview of the framework |
| [Architecture Design](docs/library-architecture-design.md) | Full technical design |
| [Creating a Task](docs/tasks/creating-a-task.md) | Step-by-step guide to adding a new task |
| [Multi-Reward Functions](docs/tasks/multi-reward-functions.md) | Per-item reward dispatch and multi-objective patterns |
| [Training Hooks](docs/tasks/training-hooks.md) | Callbacks, pre/post hooks, and custom trainer classes |
| [Speculative Decoding](docs/speculative-decoding.md) | Speed up vLLM rollouts with EAGLE/MTP draft models |
| [Reasoning + CUDA-graph Control](docs/reasoning-and-cuda-graph.md) | `infra.vllm.reasoning_parser` / `compilation_config` for thinking models; TRL patch script |

## Local Development

### uv

This library uses [uv to manage Python dependencies](https://docs.astral.sh/uv/getting-started/features/#projects).  
`brew` is The easiest way to install on macOS:

```shell
brew install uv
```

For additional installation options (e.g. setting the PATH, installing a specific version, etc),
see the installation docs:  
https://docs.astral.sh/uv/getting-started/installation/

## Python Versions Supported

### Increase Minimum Supported Version

At present, the library is using Python 3.13 for packaging purposes.

To modify which Python versions are supported and tested by this library:
- Update "envlist" in "tox" section of [tox.ini](/tox.ini)
- Update "Supported Python versions" badge in [README.md](/README.md)
- Update "project.requires-python" in [pyproject.toml](/pyproject.toml) (if needed)
- Update "tool.black.target-version" in [pyproject.toml](/pyproject.toml) (optional)

### Virtual Environment

Create by running:

```shell
uv sync --all-extras
```

Run a command from the virtual environment, like code formatting:

```shell
uv run black .
```

To activate the virtual environment:

```shell
source .venv/bin/activate
```

For more information, refer to [uv's documentation](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment).

### Type checking (mypy)

This library supports Python type [annotation](https://peps.python.org/pep-0484/).
Types will be checked as part of the test suite (see below).
For more information, see the [mypy documentation](https://mypy.readthedocs.io/en/stable/getting_started.html#dynamic-vs-static-typing).

### Testing

To run the test suite locally:
```shell
uv run tox
```

## Citation

If you use foresight-rl-core or find this repository useful for your research or work, please cite us using the following citation:

```bibtex
@software{shoham2026foresightrlcore,
    title={ForesightRLCore},
    author={Shoham, Ofir Ben and Gazman, Yakov},
    license={Apache-2.0},
    url={https://github.com/intuit/foresight-rl-core},
    year={2026}
}
```

The same author list is in [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

See [Contribution Guidelines](./CONTRIBUTING.md).

## Support

Open a GitHub issue on this repository.
