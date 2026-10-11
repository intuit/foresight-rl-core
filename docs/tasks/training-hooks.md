# Training Hooks

Training hooks let you customize TRL trainer behavior — adding callbacks, running setup/teardown logic, or swapping the trainer class entirely — all from the YAML config, without modifying framework code.

## Quick Reference

| Extension Level | Config Key | Use Case | Complexity |
|-----------------|------------|----------|------------|
| **Callbacks** | `training.hooks.callbacks` | Runtime adjustments: dynamic params, logging, early stopping | Low |
| **Pre/post hooks** | `training.hooks.pre_train` / `post_train` | One-time setup/teardown: freeze layers, upload metrics | Medium |
| **Custom trainer** | `training.hooks.trainer_class` | Override `training_step`, `compute_loss`, or generation logic | High |

## Config Shape

All hooks are optional. Add them under the `training.hooks` section:

```yaml
training:
  epochs: 1
  learning_rate: 5e-5
  # ... other training params ...
  hooks:
    trainer_class: "my_pkg.trainers.IterativeGRPOTrainer"
    pre_train: "my_pkg.hooks.freeze_embeddings"
    post_train: "my_pkg.hooks.upload_metrics"
    callbacks:
      - "my_pkg.callbacks.DynamicCompletionLength"
      - "my_pkg.callbacks.RewardPlateuStopping"
```

All values are **dotted Python import paths** in the format `"package.module.ClassName"`. The framework imports and instantiates them at training time.

## Where Hook Code Lives

The framework only stores the dotted path in YAML — it does **not** ship your implementation. Your hook module has to be importable inside whichever process actually runs training.

| Backend | What gets shipped to the worker | Where to put hook code |
|---|---|---|
| `aws_training_job` | The whole `foresightrlcore/` package is bundled into `sourcedir.tar.gz` and extracted to `/opt/ml/code` (see `execution/aws_sagemaker_launcher.py`). The `runtime/` subpackage is excluded. | Put the module **inside `foresightrlcore/`** (e.g. `foresightrlcore/examples/smoke_hooks.py`). It will be uploaded automatically — no `requirements.txt` change needed. External packages must be added to `container.requirements` or the recipe so pip installs them in the container. |
| `aws_serverless` | Trainer runs on AWS-managed infrastructure with a fixed image. | Custom hook code is not supported on this backend today. |
| `local` | Your local Python environment is used as-is. | Any importable module works (installed package, editable install, or anything on `PYTHONPATH`). |

A working reference lives in the package: see `foresightrlcore/examples/smoke_hooks.py` (the four hooks wired up in the smoke test below) and the `__getattr__`-based lazy-import pattern it uses to stay importable without TRL/`transformers`.

## Execution Order

When hooks are configured, the training loop executes in this order:

1. Load task components (reward function, task callbacks, dataset loader)
2. Resolve trainer class (`hooks.trainer_class` or default TRL trainer)
3. Instantiate hook callbacks from `hooks.callbacks`
4. Merge callbacks: task callbacks first, then hook callbacks
5. Build and instantiate the trainer
6. **Run `hooks.pre_train(trainer, config)`** if configured
7. `trainer.train()`
8. **Run `hooks.post_train(trainer, config)`** if configured
9. `trainer.save_model()`

## Callbacks (Simplest)

Callbacks are `transformers.TrainerCallback` subclasses that hook into the training loop at specific points. TRL/Transformers calls them automatically during training.

### Available Callback Events

| Event | When It Fires |
|-------|---------------|
| `on_train_begin` | Before the first training step |
| `on_step_begin` | Before each optimizer step |
| `on_step_end` | After each optimizer step |
| `on_log` | When metrics are logged |
| `on_evaluate` | After evaluation |
| `on_save` | When a checkpoint is saved |
| `on_train_end` | After training completes |

### Example: Dynamic Completion Length

Gradually increase `max_completion_length` as training progresses:

```python
# my_pkg/callbacks.py
from transformers import TrainerCallback


class DynamicCompletionLengthCallback(TrainerCallback):
    """Increase max_completion_length every N steps."""

    def __init__(self, increase_every: int = 100, increase_by: int = 512):
        self.increase_every = increase_every
        self.increase_by = increase_by

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.increase_every == 0:
            old = args.max_completion_length
            args.max_completion_length = min(old + self.increase_by, 16384)
```

```yaml
training:
  hooks:
    callbacks:
      - "my_pkg.callbacks.DynamicCompletionLengthCallback"
```

### Example: Early Stopping on Reward Plateau

Stop training when the reward hasn't improved for a set number of logging steps:

```python
# my_pkg/callbacks.py
from transformers import TrainerCallback


class EarlyStoppingOnRewardPlateau(TrainerCallback):
    """Stop training when reward stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float("-inf")
        self.wait = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        reward = logs.get("reward", logs.get("eval_reward"))
        if reward is None:
            return
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True
```

### Example: Backprop Skip on Low-Reward Batches

Skip gradient updates for batches where the mean reward is below a threshold, reducing wasted compute:

```python
# my_pkg/callbacks.py
from transformers import TrainerCallback


class BackpropSkipCallback(TrainerCallback):
    """Skip gradient updates for low-reward batches."""

    def __init__(self, reward_threshold: float = 0.1):
        self.reward_threshold = reward_threshold

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None and hasattr(state, "_last_rewards"):
            mean_reward = sum(state._last_rewards) / len(state._last_rewards)
            if mean_reward < self.reward_threshold:
                control.should_log = False
```

### How Hook Callbacks Merge with Task Callbacks

Task callbacks (from `get_callbacks()` in your task module) are loaded first. Hook callbacks are appended after them. Both sets receive all trainer events.

## Pre/Post Training Hooks

Pre and post hooks are plain callables invoked once — before or after `trainer.train()`. They receive the trainer instance and the full config, so they can modify trainer state or perform side effects.

### Signature

```python
def my_hook(trainer, config) -> None:
    """
    Args:
        trainer: The TRL trainer instance (e.g. GRPOTrainer), fully constructed.
        config: The full training config (TrainingJobConfig or raw dict).
    """
    ...
```

### Example: Freeze Embedding Layers

Freeze the embedding layer before training to prevent catastrophic forgetting:

```python
# my_pkg/hooks.py
def freeze_embeddings(trainer, config):
    """Freeze token embedding weights before training starts."""
    model = trainer.model
    if hasattr(model, "base_model"):
        model = model.base_model
    for name, param in model.named_parameters():
        if "embed" in name.lower():
            param.requires_grad = False

    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    print(f"Froze {frozen} embedding parameters")
```

```yaml
training:
  hooks:
    pre_train: "my_pkg.hooks.freeze_embeddings"
```

### Example: Upload Final Metrics

Upload training metrics to S3 after training completes:

```python
# my_pkg/hooks.py
import json

def upload_metrics_to_s3(trainer, config):
    """Save final training metrics to a JSON file."""
    metrics = trainer.state.log_history
    output_path = config.output.s3_path if hasattr(config, "output") else "/tmp"
    with open(f"{output_path}/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
```

```yaml
training:
  hooks:
    post_train: "my_pkg.hooks.upload_metrics_to_s3"
```

## Custom Trainer Class (Most Powerful)

For deep modifications to training behavior — changing how losses are computed, how generations are produced, or how gradient updates work — provide a custom subclass of the TRL trainer.

### When to Use

- Override `training_step` to change gradient computation
- Override `compute_loss` to modify the loss function
- Override `generate_completions` to change how rollouts are produced
- Implement iterative GRPO with dynamic generation rounds

### Example: Iterative GRPO Trainer

An iterative GRPO trainer that runs multiple generation-and-update rounds per batch:

```python
# my_pkg/trainers.py
from trl import GRPOTrainer


class IterativeGRPOTrainer(GRPOTrainer):
    """GRPO with multiple generation rounds per training step."""

    def __init__(self, *args, num_inner_iterations: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_inner_iterations = num_inner_iterations

    def training_step(self, model, inputs, num_items_in_batch=None):
        total_loss = 0.0
        for i in range(self.num_inner_iterations):
            loss = super().training_step(model, inputs, num_items_in_batch)
            total_loss += loss.item() if hasattr(loss, "item") else loss
        return total_loss / self.num_inner_iterations
```

```yaml
training:
  hooks:
    trainer_class: "my_pkg.trainers.IterativeGRPOTrainer"
```

### Example: Reduced Backprop Trainer

Skip backpropagation on batches where all rewards are below a threshold:

```python
# my_pkg/trainers.py
import torch
from trl import GRPOTrainer


class ReducedBackpropGRPOTrainer(GRPOTrainer):
    """Skip gradient updates when batch rewards are too low."""

    def __init__(self, *args, reward_threshold: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_threshold = reward_threshold
        self.skipped_steps = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        rewards = inputs.get("rewards")
        if rewards is not None and rewards.mean().item() < self.reward_threshold:
            self.skipped_steps += 1
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        return super().training_step(model, inputs, num_items_in_batch)
```

### Keep Heavy Imports Lazy

`transformers`, `trl`, `torch`, `peft`, and `vllm` are **not** declared as runtime dependencies of `foresightrlcore` — they are container-side dependencies installed from the recipe's requirements. A hook module that does `from trl import GRPOTrainer` (or `from transformers import TrainerCallback`) at the top will fail to import on the dev/CI machine and will break `uv run tox`.

The pattern used by `foresightrlcore/examples/smoke_hooks.py` is to build the heavy class inside a factory and expose it through a module-level `__getattr__`:

```python
def _make_smoke_grpo_trainer() -> type:
    from trl import GRPOTrainer  # imported only when the class is needed

    class SmokeGRPOTrainer(GRPOTrainer):
        ...

    return SmokeGRPOTrainer


def __getattr__(name: str):
    if name == "SmokeGRPOTrainer":
        cls = _make_smoke_grpo_trainer()
        globals()[name] = cls  # cache so subsequent lookups skip the factory
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

`foresightrlcore.hooks.import_dotted_path` does `getattr(module, attr_name)` after importing the module, so the lazy attribute is built on first dotted-path resolution — exactly when the container actually has TRL installed.

If you also want the imports type-checked, add the package to `mypy.ini`:

```ini
[mypy-transformers.*]
ignore_missing_imports = True

[mypy-trl.*]
ignore_missing_imports = True
```

## Combining All Three

You can use callbacks, pre/post hooks, and a custom trainer together:

```yaml
training:
  learning_rate: 5e-5
  grpo:
    num_generations: 8
    max_completion_length: 8192
  hooks:
    trainer_class: "my_pkg.trainers.IterativeGRPOTrainer"
    pre_train: "my_pkg.hooks.freeze_embeddings"
    post_train: "my_pkg.hooks.upload_metrics_to_s3"
    callbacks:
      - "my_pkg.callbacks.EarlyStoppingOnRewardPlateau"
      - "my_pkg.callbacks.DynamicCompletionLengthCallback"
```

Execution order:

1. Task callbacks loaded via `get_callbacks()`
2. `EarlyStoppingOnRewardPlateau` and `DynamicCompletionLengthCallback` instantiated and appended
3. `IterativeGRPOTrainer` instantiated (instead of stock `GRPOTrainer`)
4. `freeze_embeddings(trainer, config)` called
5. `trainer.train()` runs (with all callbacks active)
6. `upload_metrics_to_s3(trainer, config)` called
7. `trainer.save_model()`

## Testing Hooks

Hook bugs that only surface in the SageMaker container are expensive to find. Test in two layers.

### 1. Unit tests on the dev machine

Hooks are pure Python — test them without a GPU by mocking the trainer:

#### Test a Callback

```python
from unittest.mock import MagicMock
from my_pkg.callbacks import EarlyStoppingOnRewardPlateau


def test_early_stopping_triggers():
    cb = EarlyStoppingOnRewardPlateau(patience=3, min_delta=0.01)
    control = MagicMock()
    control.should_training_stop = False
    state = MagicMock()

    for _ in range(5):
        cb.on_log(None, state, control, logs={"reward": 0.5})

    assert control.should_training_stop is True
```

#### Test a Pre-Train Hook

```python
from unittest.mock import MagicMock
from my_pkg.hooks import freeze_embeddings


def test_freeze_embeddings():
    model = MagicMock()
    param_embed = MagicMock()
    param_other = MagicMock()
    model.named_parameters.return_value = [
        ("embed_tokens.weight", param_embed),
        ("lm_head.weight", param_other),
    ]
    trainer = MagicMock()
    trainer.model = model

    freeze_embeddings(trainer, {})

    param_embed.requires_grad = False  # would be set by the hook
```

#### Test Hook Loading

```python
from foresightrlcore.hooks import import_dotted_path


def test_import_custom_callback():
    cls = import_dotted_path("my_pkg.callbacks.EarlyStoppingOnRewardPlateau")
    assert callable(cls)
```

A worked example covering all four hook slots — including the `fake_transformers` fixture pattern for testing a `TrainerCallback` subclass without `transformers` installed — is in `tests/test_examples_smoke_hooks.py`.

### 2. End-to-end smoke test on SageMaker

Unit tests confirm your hook does the right thing in isolation. They do **not** confirm that the container can import it, that the dotted path resolves at runtime, or that the hook actually fires inside the trainer's lifecycle. The `examples/configs/single_node_smoke_test_with_hooks.yaml` config wires the four bundled `foresightrlcore.examples.smoke_hooks` reference hooks (callback, pre/post, custom trainer) onto a 5-step Qwen3-0.6B run on `ml.g5.12xlarge`. Use it as a copy-paste starting point — it costs ~15 minutes of g5 time:

```bash
AWS_DEFAULT_REGION=us-west-2 rl-train train \
    --config examples/configs/single_node_smoke_test_with_hooks.yaml
```

After the job finishes, verify each hook fired:

| Hook | Verification |
|---|---|
| `trainer_class` | CloudWatch log line `[hooks-trace] SmokeGRPOTrainer.__init__ — confirms custom trainer_class hook is active` |
| `pre_train` | CloudWatch log line `[hooks-trace] pre_train_smoke_hook — …` **and** sentinel file `hooks_pre_train.txt` in the output S3 path |
| `post_train` | CloudWatch log line `[hooks-trace] post_train_smoke_hook — …` **and** sentinel file `hooks_post_train.txt` in the output S3 path (recording final `global_step` and `log_history_len`) |
| `callbacks` | CloudWatch log lines `[hooks-trace] on_train_begin/on_step_begin/on_step_end/on_log/on_train_end` |

Quick check for the sentinels:

```bash
AWS_DEFAULT_REGION=us-west-2 aws s3 ls \
    s3://<output-bucket>/<output-prefix>/<job-id>/output/model/ \
    | grep hooks_
```

When porting the pattern to your own hooks, drop a similar sentinel file from `pre_train` / `post_train` into `trainer.args.output_dir` (which is `/opt/ml/model` in the SageMaker container). Anything written there is captured into `model.tar.gz` and pushed to your output S3 path — making it visible without needing to scrape CloudWatch.

## Troubleshooting

### ImportError: Could not import module

The dotted path must point to an importable Python module **inside the worker process**. Common causes:

- For `aws_training_job`: the module lives outside `foresightrlcore/`. Move it inside the package (see [Where Hook Code Lives](#where-hook-code-lives)) or add the external package to `container.requirements`.
- The module path is misspelled.
- The package is not installed in the local Python environment (for the `local` backend).

### ImportError on the dev machine but the SageMaker job is fine

Symptom: `tox` fails with `ModuleNotFoundError: No module named 'transformers'` (or `trl`, `torch`, `peft`, `vllm`) in your hook module, even though the production job runs cleanly.

Cause: `foresightrlcore` does not declare these as runtime dependencies — they live only in the container recipe. Importing them at module top-level breaks any local tooling that imports your hook module.

Fix: use the lazy `__getattr__` pattern from [Keep Heavy Imports Lazy](#keep-heavy-imports-lazy).

### AttributeError: Module has no attribute

The class or function name (last component of the dotted path) doesn't exist in the module. Check for typos. If the module uses lazy `__getattr__`, also check that the missing name is registered inside the `__getattr__` body.

### Hooks not firing

- Verify the `training.hooks` section is properly indented in the YAML.
- Check training logs for `Loaded hook callback`, `Using custom trainer class`, `Running pre_train hook`, and `Running post_train hook` lines emitted by `foresightrlcore/hooks.py` and `foresightrlcore/methods/base.py`.
- Run `rl-train validate --config your_config.yaml` to verify the config parses correctly.
- For end-to-end proof on SageMaker, follow the [smoke-test recipe](#2-end-to-end-smoke-test-on-sagemaker).

### Custom trainer incompatible

If your custom trainer class raises errors during instantiation, verify:

- It accepts the same constructor arguments as the TRL trainer it extends.
- It's a subclass of the expected trainer (e.g. `GRPOTrainer` for `method.type: grpo`).
