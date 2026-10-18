# Creating a Task

A **task** is a pluggable module that provides domain-specific logic for training: how to score model outputs (reward function), what to log (callbacks), and how to load data (dataset loader). Tasks are the primary extension point for adapting `rl-train` to new domains.

## The TaskModule Protocol

Every task module must export three functions:

```python
def get_reward_function(task_config: dict) -> Callable:
    """Return a reward function compatible with TRL's GRPOTrainer."""

def get_callbacks(task_config: dict) -> list:
    """Return a list of TrainerCallback instances (or empty list)."""

def get_dataset_loader(task_config: dict, model_name: str) -> Callable:
    """Return a callable: (path: str) -> datasets.Dataset."""
```

The formal protocol is defined in `foresightrlcore/tasks/__init__.py` as `TaskModule`.

## Step-by-Step Example

Let's create a `code_review` task that scores code review quality.

### 1. Create the Module

Create `foresightrlcore/tasks/code_review.py`:

```python
import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_reward_function(task_config: dict[str, Any]) -> Callable:
    reward_mode = task_config.get("reward_mode", "rule_based")

    if reward_mode == "mock":
        constant = float(task_config.get("mock_constant_reward", 0.5))

        def mock_reward(completions, **kwargs):
            return [constant] * len(completions)

        return mock_reward

    if reward_mode == "rule_based":
        return _build_rule_based_reward(task_config)

    raise ValueError(f"Unknown reward_mode: {reward_mode!r}")


def get_callbacks(task_config: dict[str, Any]) -> list:
    return []


def get_dataset_loader(task_config: dict[str, Any], model_name: str) -> Callable:
    enable_thinking = task_config.get("enable_thinking", True)

    def load(path: str):
        from datasets import Dataset
        from transformers import AutoTokenizer
        import glob

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        records = []
        for fpath in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
            with open(fpath, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    if isinstance(rec.get("prompt"), list):
                        rec["prompt"] = tokenizer.apply_chat_template(
                            rec["prompt"],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    records.append(rec)
        logger.info(f"Loaded {len(records)} records from {path}")
        return Dataset.from_list(records)

    return load


def _build_rule_based_reward(task_config: dict[str, Any]) -> Callable:
    min_length = task_config.get("min_review_length", 50)

    def reward_fn(completions, code_diff=None, **kwargs):
        rewards = []
        diffs = code_diff or [""] * len(completions)
        for i, completion in enumerate(completions):
            text = (
                completion[-1]["content"]
                if isinstance(completion, list)
                else completion
            )
            score = _score_review(text, diffs[i], min_length)
            rewards.append(score)
        return rewards

    return reward_fn


def _score_review(review: str, diff: str, min_length: int) -> float:
    score = 0.0
    if len(review) >= min_length:
        score += 0.3
    if any(keyword in review.lower() for keyword in ["bug", "issue", "fix", "error"]):
        score += 0.2
    if diff and any(line in review for line in diff.split("\n")[:5]):
        score += 0.3
    if "suggest" in review.lower() or "recommend" in review.lower():
        score += 0.2
    return min(score, 1.0)
```

### 2. Reference It in Config

```yaml
task:
  type: code_review
  config:
    reward_mode: rule_based
    min_review_length: 50
```

### 3. Prepare the Dataset

The dataset loader expects JSONL files in a directory. Each record must have a `prompt` column. Additional columns become keyword arguments to the reward function.

```jsonl
{"prompt": [{"role": "user", "content": "Review this code change:\n..."}], "code_diff": "- old\n+ new"}
{"prompt": [{"role": "user", "content": "Review this PR:\n..."}], "code_diff": "- removed\n+ added"}
```

In this example, `code_diff` is automatically passed to the reward function as a keyword argument.

## Reward Function Signature

The reward function returned by `get_reward_function` must accept:

| Parameter | Type | Description |
|-----------|------|-------------|
| `completions` | `list[list[dict]] \| list[str]` | Batch of model outputs |
| `**kwargs` | `list[...]` per kwarg | One kwarg per extra dataset column, each a list with one entry per completion |

It must return `list[float]` — one score per completion.

```python
def my_reward(completions, state=None, metadata=None, **kwargs):
    # state: list[str] — from the "state" column in the dataset
    # metadata: list[str] — from the "metadata" column
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        score = compute_score(text, state[i] if state else "")
        rewards.append(score)
    return rewards
```

## Task Resolution Order

When you set `task.type` in the YAML config, the loader resolves the module in this order:

1. **File path** — absolute path or `.py` suffix: `task.type: /opt/ml/code/my_task.py`
2. **Dotted Python path** — `task.type: my_package.tasks.custom`
3. **Built-in** — `foresightrlcore.tasks.<name>`

For most cases, placing your module in `foresightrlcore/tasks/` and using a short name (`task.type: code_review`) is the simplest approach. The launcher ships the entire `foresightrlcore` package into the SageMaker container alongside the runtime scripts, so canonical imports work the same locally and on SageMaker.

## Reward Modes

A recommended pattern is supporting multiple `reward_mode` values via `task_config`:

| Mode | Purpose |
|------|---------|
| `mock` | Pipeline testing — fixed score, no external calls |
| `rule_based` | Deterministic scoring from heuristics |
| `llm_judge` | LLM-as-judge evaluation (like the `advice` task) |

Always implement a `mock` mode for smoke tests:

```yaml
task:
  config:
    reward_mode: mock
    mock_constant_reward: 0.5
```

## Callbacks

Return `TrainerCallback` subclasses from `get_callbacks` to hook into the training loop. Common uses:

- **Logging**: Aggregate per-criterion scores and flush to TensorBoard (see `CriterionLoggingCallback` in the `advice` task)
- **Checkpointing**: Custom save logic at specific steps
- **Early stopping**: Halt training based on reward trends

Return an empty list if no callbacks are needed.

## Shared Utilities

`foresightrlcore/tasks/thinking_utils.py` provides helpers for reasoning models (Qwen3, etc.):

| Function | Purpose |
|----------|---------|
| `strip_thinking_tags(text)` | Remove `<think>...</think>` tags |
| `extract_thinking_length(text)` | Character count of the thinking section |
| `parse_json_from_response(text)` | Extract JSON from LLM output (handles code blocks) |

## Testing Locally

Reward functions are pure Python — test them without a GPU:

```python
def test_code_review_reward():
    from foresightrlcore.tasks.code_review import get_reward_function

    reward_fn = get_reward_function({"reward_mode": "rule_based", "min_review_length": 10})
    completions = [[{"content": "I found a bug: the loop doesn't handle empty lists. I suggest adding a guard."}]]
    scores = reward_fn(completions, code_diff=["- for x in items:"])
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
```

## Using the Genos LLM Client

Tasks that need to call an Intuit Genos LLM endpoint (e.g. for LLM-as-judge evaluation) should use the shared client in `foresightrlcore/clients/genos.py` instead of building their own auth + HTTP logic:

```python
from foresightrlcore.clients.genos import get_llm_client, GenosClientManager

# One-shot client (credentials dict must include app_secret, etc.)
client = get_llm_client(model_id="anthropic.claude-opus-4-5-20251101-v1-0", credentials=creds)

# Thread-safe manager with automatic token refresh (15 min TTL)
mgr = GenosClientManager(model_id="anthropic.claude-opus-4-5-20251101-v1-0", credentials=creds)
client = mgr.get()  # cached until TTL expires
```

The target Genos environment (`e2e`, `prf`, `prd`) is controlled by the `GENOS_ENV` environment variable, set via the top-level `environment` section in the YAML config. Credentials are loaded by each task's `_load_credentials()` function, which reads all `GENOS_*` env vars including `GENOS_ENV` (defaults to `e2e`).

## Reference Implementation

The `advice` task at `foresightrlcore/tasks/advice.py` is the most complete example. It demonstrates:

- LLM-as-judge with structured Pydantic output parsing via `foresightrlcore.clients.genos`
- Thinking-length penalties for reasoning models
- Thread-pooled parallel evaluation
- Token refresh for external API credentials (via `GenosClientManager`)
- Per-criterion TensorBoard logging via callbacks
- Mock mode for pipeline testing
- Prediction logging to JSONL
