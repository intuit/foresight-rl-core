# Speculative Decoding

Speculative decoding can substantially speed up GRPO rollouts by having a small **draft** model propose several tokens at a time, which the **target** model then verifies in a single forward pass. This is opt-in via the `infra.vllm.speculative` block in your YAML config.

## When this applies

| `vllm_mode` | Speculative supported? |
|-------------|------------------------|
| `server`    | Yes — `--speculative-config` is forwarded through `vllm_launcher.py` to TRL's `vllm-serve` CLI verbatim |
| `colocate`  | No — TRL's `GRPOConfig` does not expose a speculative kwarg today; setting `speculative` here logs a warning and is ignored |
| `none`      | No — there is no vLLM at all |

Speculative is for **inference-time generation only**. Training math is unchanged.

## Config Shape

`infra.vllm.speculative` is forwarded **verbatim** as a JSON object to `vllm serve --speculative-config`. Anything vLLM accepts on that flag works here. The schema fixes the two required fields (`method`, `num_speculative_tokens`) and lets unknown keys pass through (`extra="allow"` on `SpeculativeConfig`).

### Custom-trained draft model (e.g. EAGLE3)

```yaml
infra:
  vllm_mode: server
  vllm:
    tensor_parallel_size: 2
    speculative:
      method: eagle3
      num_speculative_tokens: 5
      model: "s3://my-bucket/path/to/draft_checkpoint/"   # or HF ID, or local path
      draft_tensor_parallel_size: 1
```

**S3 staging.** vLLM's `--speculative-config model:` only accepts a local path or HF Hub ID, not `s3://` URIs. The launcher detects an `s3://` draft and stages it to a local directory via `aws s3 sync` before vLLM starts:

- **SageMaker**: synced to `/tmp/speculative_draft_model/` inside the container.
- **Local backend**: synced to `<output_dir>/speculative_draft_model/`.

The JSON forwarded to vLLM has `model:` rewritten to that local path. The IAM role running the job (SageMaker execution role or local AWS creds) must have read access to the S3 prefix.

Equivalent to launching vLLM directly as:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve <target_model> \
  --tensor-parallel-size 2 \
  --speculative-config '{
    "method": "eagle3",
    "num_speculative_tokens": 5,
    "model": "s3://my-bucket/path/to/draft_checkpoint/",
    "draft_tensor_parallel_size": 1
  }'
```

### Built-in MTP head (Qwen3-Next)

Some models ship with a multi-token-prediction head — no separate draft checkpoint needed. Omit `model` and `draft_tensor_parallel_size`:

```yaml
infra:
  vllm_mode: server
  vllm:
    speculative:
      method: qwen3_next_mtp
      num_speculative_tokens: 5
```

This shape only works for the model families vLLM supports it for (e.g. Qwen3-Next). Check the [vLLM speculative-decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode.html) for the current list of supported `method` values.

## Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `method` | yes | Speculative method, e.g. `eagle3`, `medusa`, `qwen3_next_mtp` |
| `num_speculative_tokens` | yes | Tokens drafted per step (typical 3–8; higher trades acceptance rate for verify cost) |
| `model` | no | Draft model path/ID. Required for custom drafts; omit for built-in MTP heads |
| `draft_tensor_parallel_size` | no | Tensor-parallel size for the draft. Defaults to vLLM's choice; set to 1 if the draft fits on a single GPU |
| *(other)* | no | Any additional vLLM speculative-config keys are passed through unchanged |

## Verifying it's actually on

After launching, confirm in the vLLM logs (CloudWatch lines prefixed with `[vllm]` for SageMaker, or `output/<job>/vllm_server.log` for the local backend):

```
INFO ... Speculative decoding enabled with method=eagle3 ...
INFO ... Loading draft model from <path> ...
```

Absence of those lines means the JSON didn't reach vLLM — re-check that `vllm_mode: server` and that the YAML block is under `infra.vllm.speculative` (not at top level).

## Choosing `num_speculative_tokens`

- Too low (1–2): little speedup over no-spec.
- Sweet spot (3–8): typical wall-clock speedup of ~1.5–2.5× depending on draft quality.
- Too high (>10): more rejected drafts, verify cost dominates.

Start with `5` and tune from there.

## Example Configs

- **`examples/configs/sagemaker_advice_v4_run_speculative.yaml`** — full advice-task run with `eagle3` + S3-hosted custom draft (exercises the staging path).
- **`examples/configs/sagemaker_advice_v4_run_speculative_mtp.yaml`** — same task with `qwen3_next_mtp` (built-in MTP head, no draft checkpoint).
- **`examples/configs/sagemaker_advice_v4_run.yaml`** — the same config without speculative (for comparison/fallback) plus commented-out speculative blocks.

## Limitations

- **Server mode only** until TRL adds a `vllm_speculative_config` field to `GRPOConfig`. Track upstream in [huggingface/trl](https://github.com/huggingface/trl); when it lands we can wire colocate via `methods/base.py:build_trainer_args`.
- **vLLM version**: speculative method names track vLLM. The `--speculative-config` JSON-object form is in vLLM 0.7+; older versions used per-flag arguments and won't accept this shape.
- **Draft/target compatibility** depends on the method:
  - **Custom-draft methods** (`eagle`, `eagle3`, `medusa`, `mlp_speculator`): drafts are trained against a specific target architecture. Mixing a draft trained for one base with a different base will either fail to load or produce poor acceptance rates.
  - **Built-in MTP methods** (`qwen3_next_mtp`, `deepseek_mtp`, etc.): the multi-token-prediction head ships with the model architecture itself, so the method works for **any checkpoint of that architecture**, including fine-tunes. `qwen3_next_mtp` works for any Qwen3-Next model (Qwen3.5, Qwen3.6, fine-tunes, etc.) without a separate draft.
