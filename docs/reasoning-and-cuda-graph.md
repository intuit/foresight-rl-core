# Reasoning + CUDA-graph Control

Thinking models (Qwen3, Qwen3.5, …) generate a `<think>…</think>` segment before their final answer. To bound the thinking budget at the vLLM engine layer and to control CUDA-graph capture on H100, the framework exposes four opt-in fields under `infra.vllm`:

| Field | What it controls |
|---|---|
| `reasoning_parser` | Reasoning parser identifier (e.g. `"qwen3"`). Tells vLLM which built-in parser to load. |
| `reasoning_config` | `{reasoning_start_str, reasoning_end_str}` delimiters that drive vLLM's `ThinkingTokenBudgetLogitsProcessor`. |
| `compilation_config` | Kwargs forwarded verbatim to vLLM's `CompilationConfig` (CUDA-graph capture sizes, warmups, mode). |
| `disable_custom_all_reduce` | When `true`, injects `disable_custom_all_reduce=True` into `LLM(...)`. Workaround for the broken vLLM 0.20.x custom all-reduce kernel at `tensor_parallel_size > 1`. |

All four are **server-mode only**. With `vllm_mode: colocate` or `none`, the validator logs a warning and the field is ignored — TRL's `GRPOConfig` has no equivalent kwarg today.

## Why a patch script?

TRL's `vllm-serve` CLI does not expose `reasoning_config`, `compilation_config`, or `disable_custom_all_reduce`, but vLLM's `LLM(...)` constructor accepts them via `**kwargs` → `EngineArgs`. Rather than forking TRL, the framework patches TRL's installed `vllm_serve.py` at job start.

`foresightrlcore/runtime/scripts/patch_trl_vllm.py`:

1. Locates `trl.scripts.vllm_serve` in site-packages.
2. Looks for the marker line `model_impl=script_args.vllm_model_impl,` inside the `LLM(...)` call.
3. Injects the requested kwargs (and the `from vllm.config import ReasoningConfig / CompilationConfig` imports) immediately after the marker.
4. Is **idempotent** — re-running on an already-patched file is a no-op.

## How the patch is delivered

The patch runs inside `foresightrlcore/runtime/scripts/vllm_launcher.py` — the single, spawn-safe entry point every backend (SageMaker `entrypoint.sh`, local `execution/local.py`, HyperPod EKS `aws_hyperpod_eks/manifests.py`) uses in place of `trl vllm-serve`. Each backend reads the opt-in YAML fields and exports three env vars that the launcher consumes at startup, **before** importing TRL:

| Env var | YAML source | Format |
|---|---|---|
| `VLLM_REASONING_CONFIG_JSON` | `infra.vllm.reasoning_config` | JSON object with `reasoning_start_str` + `reasoning_end_str` |
| `VLLM_COMPILATION_CONFIG_JSON` | `infra.vllm.compilation_config` | JSON object of `CompilationConfig` kwargs |
| `VLLM_DISABLE_CUSTOM_ALL_REDUCE` | `infra.vllm.disable_custom_all_reduce` | `"1"` to opt in |

The launcher calls `patch_trl_vllm.patch(...)` directly (not via a bash subprocess), gated three ways:

- The launcher is only used when `infra.vllm_mode == server`, AND
- At least one of the three env vars is present, AND
- The marker line is still present in TRL's source.

### Fail-fast on marker drift

If the user opts into any of the new fields but TRL upstream has rewritten `vllm_serve.py` and the marker is gone, the patch returns non-zero. `vllm_launcher.py` propagates that exit code without ever importing TRL — the job ends immediately with a clear error message rather than starting vLLM with the feature silently absent.

If the marker is still there but the file is already patched (e.g. a job rerun in the same container), `patch_trl_vllm.patch(...)` logs `"Patch already applied; skipping"` and returns zero.

## Usage

### Qwen3.5 thinking model with bounded CUDA-graph capture

```yaml
infra:
  vllm_mode: server
  topology: single_node
  vllm:
    tensor_parallel_size: 2
    reasoning_parser: qwen3
    reasoning_config:
      reasoning_start_str: "<think>"
      reasoning_end_str: "</think>"
    compilation_config:
      cudagraph_capture_sizes: [1, 2, 4, 8]
      cudagraph_num_of_warmups: 1
```

Equivalent vLLM CLI (for reference; the framework drives this via the patch script, not the CLI):

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8],"cudagraph_num_of_warmups":1}'
```

### Disabling vLLM's custom all-reduce kernel (TP > 1 workaround)

Some vLLM 0.20.0 builds ship a broken custom all-reduce kernel with `tensor_parallel_size > 1`. Set the schema field to inject `disable_custom_all_reduce=True` into the `LLM(...)` call:

```yaml
infra:
  vllm_mode: server
  vllm:
    tensor_parallel_size: 4
    disable_custom_all_reduce: true
```

The backend translates this YAML field into the `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` env var, which `vllm_launcher.py` reads and forwards to `patch_trl_vllm.patch(...)`. The env-var name is also accepted directly (`environment: { VLLM_DISABLE_CUSTOM_ALL_REDUCE: "1" }`) for ad-hoc overrides without editing the schema fields.

## Validation

`config/validation.py` emits a warning (errors stay empty) for each of these cases:

| Condition | Warning |
|---|---|
| `reasoning_parser` set with `vllm_mode != server` | `"is only honored in vllm_mode='server' ... will be ignored"` |
| `compilation_config` set with `vllm_mode != server` | same |
| `disable_custom_all_reduce` set with `vllm_mode != server` | same |
| `speculative` set with `vllm_mode != server` | same — existing pre-FILM-5307 behavior |

## Schema

`foresightrlcore/config/schema.py::VllmConfig` declares all four fields with safe defaults. Existing YAML configs validate identically — nothing in your config has to change unless you want to opt in.

```python
class VllmConfig(BaseModel):
    # ...
    reasoning_parser: Optional[str] = None
    reasoning_config: Optional[Dict[str, str]] = None
    compilation_config: Optional[Dict[str, Any]] = None
    disable_custom_all_reduce: bool = False
```

## Tests

| File | Coverage |
|---|---|
| `tests/config/test_config_schema.py::TestFilm5307PortFields` | Default-`None` and populated round-trip through `from_yaml` / `to_yaml`. |
| `tests/config/test_config_validation.py::TestFilm5307VllmFieldWarnings` | Warnings fire for non-server modes; quiet for `server`. |
| `tests/runtime/test_patch_trl_vllm.py` | No-op when nothing requested; happy-path injection for each kwarg; idempotent re-apply; **fail-fast when marker missing AND user opted in**. |
| `tests/runtime/test_vllm_launcher.py::TestApplyTrlPatchFromEnv` | Launcher reads the three env vars and forwards them to `patch_trl_vllm.patch(...)`; aborts before importing TRL on patch failure. |
| `tests/execution/test_local_backend_launch.py::TestStartVllmServer::test_patch_env_vars_set_when_yaml_opts_in` | Local backend translates YAML opt-ins into launcher env vars. |
| `tests/execution/test_aws_hyperpod_eks_backend.py::TestBuildVllmCommand::test_patch_env_vars_exported_when_yaml_opts_in` | HyperPod EKS manifest emits `export VAR=...` lines before the launcher invocation. |

## Related

- [Speculative Decoding](speculative-decoding.md) — sibling opt-in for `infra.vllm.speculative`.
- [Launching a SageMaker Training Job](launching-sagemaker-training-job.md) — full reference for `infra.vllm.*`.
