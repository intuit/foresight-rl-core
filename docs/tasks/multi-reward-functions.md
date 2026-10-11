# Multi-Reward Functions

This guide covers two advanced reward patterns:

1. **Per-item dispatch** — route each dataset item to a different reward function based on a column value
2. **Multi-objective rewards** — score every item with multiple reward functions and combine the results

## Per-Item Dispatch

Use this when different items in your dataset need fundamentally different scoring logic — for example, some items need an LLM judge while others use rule-based heuristics.

### How It Works

TRL's `GRPOTrainer` passes all extra dataset columns as keyword arguments to the reward function. Add a routing column (e.g. `reward_type`) to your dataset, then dispatch inside a single reward function.

### Dataset

```jsonl
{"prompt": [{"role": "user", "content": "..."}], "reward_type": "llm_judge", "state": "..."}
{"prompt": [{"role": "user", "content": "..."}], "reward_type": "rule_based", "state": "..."}
{"prompt": [{"role": "user", "content": "..."}], "reward_type": "code_exec", "test_cases": "..."}
```

### Implementation

```python
def get_reward_function(task_config: dict) -> Callable:
    reward_registry = {
        "llm_judge": _build_llm_judge_reward(task_config),
        "rule_based": _build_rule_based_reward(task_config),
        "code_exec": _build_code_exec_reward(task_config),
    }
    default_type = task_config.get("default_reward_type", "rule_based")

    def dispatcher_reward(completions, reward_type=None, **kwargs):
        types = reward_type or [default_type] * len(completions)
        rewards = [0.0] * len(completions)

        # Group by reward type for batch efficiency
        groups: dict[str, list[int]] = {}
        for i, rt in enumerate(types):
            groups.setdefault(rt, []).append(i)

        for rt, indices in groups.items():
            fn = reward_registry.get(rt)
            if fn is None:
                raise ValueError(f"Unknown reward_type: {rt!r}")

            batch_completions = [completions[i] for i in indices]
            batch_kwargs = {k: [v[i] for i in indices] for k, v in kwargs.items()}
            batch_rewards = fn(batch_completions, **batch_kwargs)

            for idx, score in zip(indices, batch_rewards):
                rewards[idx] = score

        return rewards

    return dispatcher_reward
```

### Why Group by Type?

The naive approach calls each sub-function once per item. Grouping by type preserves batch efficiency — if a sub-function uses thread pools, API batching, or GPU inference, it processes all its items at once instead of one at a time.

### Config

```yaml
task:
  type: my_multi_task
  config:
    default_reward_type: rule_based
    min_review_length: 50

    # Single LLM client at the conventional name 'default'. Any sub-function
    # that needs an LLM judge calls resolve_llm_manager(clients.get("default")).
    # See docs/tasks/llm-client-config.md for the multi-client pattern.
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

## Multi-Objective Rewards

Use this when you want to score **every** item on multiple dimensions simultaneously and combine the scores with configurable weights.

### How TRL Handles Multiple Reward Functions

`GRPOTrainer` accepts `reward_funcs` as a **list of callables**. Each function is called for every batch, producing separate reward signals. TRL combines them using `reward_weights`.

```python
trainer = GRPOTrainer(
    model=model_name,
    args=grpo_args,
    train_dataset=train_dataset,
    reward_funcs=[format_reward, quality_reward, safety_reward],
    ...
)

# In GRPOConfig:
grpo_args.reward_weights = [0.2, 0.6, 0.2]
```

### Implementation

Return a list from `get_reward_function` and configure weights in the YAML:

```python
def get_reward_function(task_config: dict) -> list[Callable]:
    fns = []

    if task_config.get("enable_format_reward", True):
        fns.append(_build_format_reward(task_config))
    if task_config.get("enable_quality_reward", True):
        fns.append(_build_quality_reward(task_config))
    if task_config.get("enable_safety_reward", True):
        fns.append(_build_safety_reward(task_config))

    return fns


def _build_format_reward(task_config: dict) -> Callable:
    def format_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            score = 1.0 if _is_valid_json(text) else 0.0
            rewards.append(score)
        return rewards
    return format_reward


def _build_quality_reward(task_config: dict) -> Callable:
    def quality_reward(completions, **kwargs):
        # Score content quality (heuristic, LLM judge, etc.)
        ...
    return quality_reward


def _build_safety_reward(task_config: dict) -> Callable:
    def safety_reward(completions, **kwargs):
        # Binary safety gate
        ...
    return safety_reward
```

### Config

```yaml
task:
  type: my_multi_objective_task
  config:
    enable_format_reward: true
    enable_quality_reward: true
    enable_safety_reward: true

training:
  grpo:
    reward_weights: [0.2, 0.6, 0.2]  # format, quality, safety
```

The `reward_weights` field is already supported in the config schema and wired through to `GRPOConfig` in `foresightrlcore/methods/grpo.py`.

### Wiring Note

The current `get_task_components` in `foresightrlcore/tasks/__init__.py` assigns the return value of `get_reward_function` directly to `reward_funcs` in `GRPOTrainer`. TRL accepts both a single callable and a list of callables, so returning a list works without any framework changes.

## Combining Both Patterns

You can combine per-item dispatch within a multi-objective setup. Each reward function in the list can internally dispatch based on dataset columns:

```python
def get_reward_function(task_config: dict) -> list[Callable]:
    return [
        _build_format_reward(task_config),       # applied to all items
        _build_dispatched_quality_reward(task_config),  # per-item dispatch
    ]
```

## Choosing the Right Pattern

| Pattern | When to Use |
|---------|-------------|
| **Single reward** | One scoring strategy for all items (default, simplest) |
| **Per-item dispatch** | Items need fundamentally different scoring (e.g., math vs. code vs. essay) |
| **Multi-objective** | All items scored on multiple dimensions (e.g., format + quality + safety) |
| **Both combined** | Multi-dimensional scoring where some dimensions vary by item type |

## Testing

Test the dispatcher with mixed-type batches:

```python
def test_dispatcher_routes_correctly():
    from my_task import get_reward_function

    reward_fn = get_reward_function({
        "default_reward_type": "rule_based",
        "reward_mode": "mock",
    })

    completions = [
        [{"content": "advice A"}],
        [{"content": "advice B"}],
        [{"content": "code review C"}],
    ]
    scores = reward_fn(
        completions,
        reward_type=["llm_judge", "llm_judge", "rule_based"],
        state=["state1", "state2", "state3"],
    )
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
```

Test multi-objective with the full list:

```python
def test_multi_objective_returns_list():
    from my_task import get_reward_function

    fns = get_reward_function({"enable_safety_reward": True})
    assert isinstance(fns, list)
    for fn in fns:
        scores = fn([[{"content": "test"}]])
        assert len(scores) == 1
```
