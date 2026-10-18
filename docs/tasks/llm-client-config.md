# LLM Client Configuration

`foresight-rl-core` exposes a single, reusable shape for configuring
LLM clients used by reward functions, judges, classifiers, or anything
else inside a task that needs to call an LLM. The block lives at
`task.config.llm_clients` and is a **mapping** of task-defined names to
provider-agnostic client configurations.

Any task that wants an LLM judge embeds the same block — there is no
task-specific schema for it.

## Reusability contract

A task using one or more LLM clients does four things:

1. **Validate the mapping** at task-module entry:

   ```python
   from foresightrlcore.config.schema import LLMClientsConfig
   clients = LLMClientsConfig.model_validate(task_config["llm_clients"])
   ```

2. **Pick the client(s) by name** — names are task-defined:

   ```python
   # advice task (single-client, conventional name 'default')
   cfg = clients.get("default")

   # hypothetical RLAIF task (multi-client)
   reward_cfg = clients.get("reward_judge")
   safety_cfg = clients.get("safety_judge")
   ```

3. **Resolve a manager** without owning provider-specific credential code:

   ```python
   from foresightrlcore.clients import resolve_llm_manager
   manager = resolve_llm_manager(cfg)
   ```

4. **Use the manager** through a provider-agnostic interface:

   ```python
   client = manager.get()
   structured = client.with_structured_output(MyPydanticSchema)
   result: MyPydanticSchema = structured.invoke(messages)
   ```

`MyPydanticSchema` is task-defined and can be **any** `pydantic.BaseModel`.
The Bedrock shim converts it into a forced `toolUse` on the fly; the
Genos client uses LangChain's `ChatOpenAI.with_structured_output`
natively.

## Single-client task

Use the conventional name `default`:

```yaml
task:
  type: advice
  config:
    reward_mode: llm_judge
    llm_clients:
      default:
        provider: bedrock          # bedrock | genos
        bedrock:
          model_id: us.anthropic.claude-sonnet-4-5-20250929-v1:0
          region: us-west-2        # optional; falls back to AWS_DEFAULT_REGION
          temperature: 0.0         # optional; default 0.0
```

To switch to Genos, replace the `bedrock:` block with `genos:`:

```yaml
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

## Multi-client task (illustrative)

A hypothetical RLAIF task that needs both a reward judge and a safety
judge declares two named entries. Providers can be mixed:

```yaml
task:
  type: rlaif
  config:
    llm_clients:
      reward_judge:
        provider: bedrock
        bedrock:
          model_id: us.anthropic.claude-opus-4-5-20251101-v1:0
          region: us-west-2
      safety_judge:
        provider: genos
        genos:
          model_id: anthropic.claude-haiku-4-5-20251001-v1-0
```

The task code retrieves each client by its name:

```python
reward_mgr = resolve_llm_manager(clients.get("reward_judge"))
safety_mgr = resolve_llm_manager(clients.get("safety_judge"))
```

No production task currently consumes more than one named client; the
example exists to document the supported shape. See
[`examples/configs/multi_client_example.yaml`](../../examples/configs/multi_client_example.yaml)
for a complete YAML.

## Provider comparison

| Concern | `bedrock` | `genos` |
|---|---|---|
| Auth | IAM role / boto3 credential chain (SageMaker execution role) | `GENOS_APP_SECRET` via `secrets:` + `GENOS_*` env vars |
| Region | `region:` field (falls back to `AWS_DEFAULT_REGION`) | N/A (Genos is region-internal) |
| Secrets section in YAML | Not required | Required (IDPS or local provider) |
| New pip dependency | None — built on `boto3`, a core dependency | None — built on `langchain_openai` already required |
| Token refresh | None (boto3 handles credential rotation) | 15-min TTL with `force_refresh()` |
| Structured output | In-house shim via Bedrock Converse `toolUse` | `ChatOpenAI.with_structured_output` natively |

### When to pick Bedrock

* Running on SageMaker and the execution role already has — or can be
  granted — `bedrock:InvokeModel`.
* Want to drop the `secrets:` block entirely (no Genos secret to manage).
* Need a model only available on Bedrock.

### When to pick Genos

* The model you need is exposed via Genos but not Bedrock.
* Existing infrastructure already provisions `GENOS_*` secrets and you
  want to keep parity with other jobs.

`environment.GENOS_ENV` selects the stack (`e2e` default). Pre-prod slugs
(`e2e`, `prf`, `qal`) are inserted into the service host. `prd` and `prod`
use the unsuffixed host; a `prd` infix does not resolve. SageMaker jobs must
use an IAM role listed on the IDPS policy for that environment.

## IAM requirement for Bedrock

The execution role used by the training job needs a policy allowing:

* `bedrock:InvokeModel`
* `bedrock:InvokeModelWithResponseStream` (only if streaming is later
  enabled)

scoped to the model ARNs in the configured region. This is a one-time
IAM update outside of code; see
[`docs/launching-sagemaker-training-job.md`](../launching-sagemaker-training-job.md).

## Migration notes

Two earlier shapes are rejected by the validator at task-load time:

* **Flat `critic_model_id`** has been removed. Use
  `llm_clients.default.{bedrock|genos}.model_id` instead.
* **Prototype `llm_judge:` key** is no longer recognized. Wrap the block
  under `llm_clients.default` (the contents are otherwise identical).

In both cases the error message includes a corrected YAML snippet.
