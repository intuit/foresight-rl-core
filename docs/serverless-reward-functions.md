# Serverless reward functions

How to add, register, bundle, and deploy a custom reward Lambda for
SageMaker Serverless RLVR. The framework owns the boilerplate
(get-or-create AI Registry evaluator, multi-file Lambda zip, single-file
Studio bundle, generic JSONL converter); your task only provides the
reward-shape logic.

## TL;DR

1. Put your task code under `foresightrlcore/tasks/<task>/` with the
   four-file layout below.
2. Register the lambda + converter in `__init__.py`. Use the same name
   you'll write in YAML as `custom_reward_function_name`.
3. Run `rl-train serverless list lambdas` to confirm registration.
4. Submit jobs with the usual `rl-train train --config job.yaml`.

## Per-task folder layout

```
foresightrlcore/tasks/<task>/
├── __init__.py        — registers lambda, converter, dataset; re-exports
│                         the protocol functions for the task loader
├── trainer.py         — TRL-side reward function + dataset loader + callbacks
│                         (the existing per-task file content)
├── judge_logic.py     — stdlib-only shared logic (prompts, scoring math)
│                         used by both trainer.py and lambda_reward.py
├── lambda_reward.py   — AWS Lambda handler: Bedrock transport + sample IO
│                         + reward dispatch (uses helpers from
│                         foresightrlcore/serverless/helpers/)
└── convert.py         — record-level mapper from your raw JSONL to the
                          SageMaker RLVR schema (RLVRRecord)
```

The `<task>/__init__.py` registers the components and re-exports the
training protocol functions, wrapping the trainer re-export in a
`try/except ImportError` so the registry remains usable on machines that
don't have the training stack installed.

## Registry contract

Three flat registries in
[`foresightrlcore/serverless/registry.py`](../foresightrlcore/serverless/registry.py),
populated at task-package import time:

| Registry      | Key                                  | What it tracks                                          |
|---------------|--------------------------------------|---------------------------------------------------------|
| `LAMBDAS`     | AI Registry evaluator name           | `LambdaSpec(main_module, extra_modules, env_defaults)`  |
| `CONVERTERS`  | Short name (`"advice"`, `"mailchimp"`) | Dotted callable + supported schemas                     |
| `DATASETS`    | AI Registry DataSet name             | Description + optional suggested validation dataset     |

Registration call (one block per task, in the task's `__init__.py`):

```python
from foresightrlcore.serverless import registry

registry.register_lambda(
    name="example-reward-fn",         # = custom_reward_function_name in YAML
    main_module="foresightrlcore.tasks.advice.lambda_reward",
    extra_modules=[
        "foresightrlcore.tasks.advice.judge_logic",
        "foresightrlcore.tasks.thinking_utils",
        "foresightrlcore.serverless.helpers.bedrock_judge",
        "foresightrlcore.serverless.helpers.sample_io",
    ],
    env_defaults={
        "JUDGE_MODEL_ID": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "JUDGE_REGION":   "us-west-2",
    },
)
```

## `rl-train serverless` CLI

```
rl-train serverless register   --name <X> --role <ARN> [--source PATH] [--extra-file ...]
rl-train serverless bundle     --name <X> [--output PATH]
rl-train serverless convert    --input IN --output OUT [--schema rlvr|sft|dpo|kto]
                               {--converter NAME|dotted-path | --mapping YAML}
rl-train serverless list       [lambdas|datasets|converters|schemas|all]
```

* `register` — get-or-create the AI Registry evaluator. With `--source`
  the framework registers an explicit `.py` file path; without it the
  framework looks `--name` up in the registry and bundles the
  registered `main_module` + `extra_modules`.
* `bundle` — emit a single-file Lambda for SageMaker Studio's paste UI.
  Drops the `# BUNDLER_STRIP_BEGIN`/`# BUNDLER_STRIP_END` cross-module
  imports.
* `convert` — JSONL → SageMaker schema, pluggable per `--schema`.
  `--schema rlvr` is the only schema implemented; `sft`, `dpo`, `kto`
  are reserved.
* `list` — inspect what's registered.

## YAML config

`execution.serverless.custom_reward_function_name` doubles as the AI
Registry evaluator name AND the registry key. One identifier, not two:

```yaml
execution:
  type: aws_serverless
  serverless:
    custom_reward_function_name: example-reward-fn    # = registry key
    custom_reward_function_role: "arn:aws:iam::<account>:role/<role>"
    # custom_reward_function_source: ...               # OPTIONAL escape hatch
```

Resolution order at launch:

1. AI Registry has an evaluator under that name → use it.
2. `custom_reward_function_source` set → build the evaluator from that
   file path / S3 URI.
3. Look the name up in `registry.LAMBDAS` → bundle and create the
   evaluator from the registered `LambdaSpec`.

When `source` is set and the name is also registered, `source` wins
(explicit beats implicit) and a warning is logged.

## Generic JSONL converter

Two plug-in points: the output schema and the per-task mapper.

### Schemas

Implemented today: `rlvr`. Reserved keys: `sft`, `dpo`, `kto`.
Each schema is a pydantic model with `extra="forbid"` in
[`foresightrlcore/serverless/spec.py`](../foresightrlcore/serverless/spec.py).
Adding a future schema is a single-file change: add the model, set
`SCHEMAS["sft"] = SFTRecord`, write one round-trip test.

### Per-task mappers

Two ways to define a mapper:

1. **Python mapper** (full power):

   ```python
   from foresightrlcore.serverless import registry
   from foresightrlcore.serverless.spec import RewardModel, RLVRRecord

   def convert_record(record: dict) -> RLVRRecord:
       return RLVRRecord(
           data_source="advice",
           prompt=record["prompt"],
           ability="generation",
           reward_model=RewardModel(
               ground_truth=record["state"], style="model"
           ),
           extra_info=_extra_info(record),
       )

   # In your task's __init__.py:
   registry.register_converter(
       name="advice",
       mapper="foresightrlcore.tasks.advice.convert:convert_record",
       schemas={"rlvr"},
   )
   ```

2. **YAML mapping spec** (no-code, simple renames):

   ```yaml
   # convert_mailchimp.yaml
   schema: rlvr
   data_source: mailchimp
   ability: classification
   reward_style: rule
   prompt_from: prompt
   ground_truth_from: outcome_category
   extra_info_passthrough: [trace_id, task]
   ```

   ```bash
   rl-train serverless convert \
       --input raw.jsonl --output rlvr.jsonl \
       --mapping convert_mailchimp.yaml
   ```

   For fanout, derived fields, or anything more complex, write a Python
   mapper instead.

## Generic helpers

[`foresightrlcore/serverless/helpers/`](../foresightrlcore/serverless/helpers/)
holds stdlib + boto3-only utilities reusable across task Lambdas:

* `BedrockJudge(model_id, region, ...)` — thread-safe wrapper around
  `boto3.bedrock-runtime.converse` with retry on transient `ClientError`
  codes. One `.invoke(system_prompt, user_prompt)` call returns the
  assistant text or raises `JudgeError` on exhaustion.
* `coerce_batch(event)`, `extract_response(sample)`, `extract_state(sample)` —
  parse the various SageMaker RLVR Lambda invocation shapes.

Add them to your task's `LambdaSpec.extra_modules` so the bundler
includes them in the zip.

## Deferred / roadmap

The following are **out of scope** but tracked:

* Forwarding `training.serverless.extra` to the Lambda `Environment`
  variables at evaluator-creation time (so business knobs can stay in
  YAML).
* A per-user `~/.config/foresight-rl-core/defaults.yaml` for things like
  the default IAM role ARN used by `rl-train serverless register`.

See [`docs/plans/future-roadmap.md`](plans/future-roadmap.md) for the
current list.
