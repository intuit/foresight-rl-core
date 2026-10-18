"""Model profiles — named presets for per-model LoRA targeting + chat-template quirks.

This is shared infrastructure, not something new or verl-specific: it's
the SAME mechanism ``foresightrlcore.methods.base`` (the TRL/GRPO path)
already depends on. ``ModelConfig``'s own pydantic validator
(``config/schema.py``'s ``_apply_model_profile``) calls
``resolve_model_profile()`` unconditionally for every ``TrainingJobConfig``,
regardless of ``method.type`` — so ``model.profile: <name>`` in YAML
resolves identically whether the job runs through TRL/GRPO or verl/SFT.
``methods/base.py``'s LoRA-config-building has been reading this registry's
``exclude_modules``/``target_parameters`` since before verl support
existed, precisely so a model profile benefits both paths at once. The
only fields here that are currently verl-specific are
``attn_implementation``/``thinking_kwarg`` — not because of any
architectural split, but simply because no TRL-based recipe has yet hit a
model that needs them (see the ``gemma4_moe`` profile's
``attn_implementation`` for why Gemma 4 does).

Switching the model a config trains (Gemma 4, Qwen3-MoE, a future MoE
release, ...) touches a small set of facts that are architecture-specific
and cannot be derived automatically: which modules/parameters LoRA needs to
target, whether the router should be excluded, and what kwarg name the
tokenizer's chat template uses to toggle thinking mode. Discovering these
requires an actual inspection pass against the real checkpoint — build the
model on the meta device (no weight download) and read
``named_parameters()``/``named_modules()``, then check
``tokenizer.apply_chat_template`` with the candidate kwarg name. See the
Gemma 4 26B-A4B profile below for the exact evidence behind each field.

This registry exists so that pass only has to happen **once per model
family** — after that, switching configs is a one-line
``model.profile: <name>`` in YAML instead of hand-editing the ``lora:``
block. It does not, and cannot, eliminate the inspection pass itself.

To add a new profile:
  1. Run the meta-device inspection + chat-template check against the real
     checkpoint (see docs/tasks/ or the conversation history that produced
     the ``gemma4_moe`` profile for the exact commands).
  2. Add an entry below with whatever fields you were able to confirm —
     all fields are optional; omit what doesn't apply (e.g. a dense model
     has no ``target_parameters``).
  3. Reference it via ``model.profile: <name>`` in a config. Fields you set
     explicitly under ``model.lora`` (or ``model.thinking_kwarg`` /
     ``model.trust_remote_code``) always win over the profile — resolution
     only fills in what you left unset.
  4. Add a test in ``tests/config/test_model_profiles.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foresightrlcore.config.schema import ModelConfig

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "gemma4_moe": {
        "description": (
            "Google Gemma 4 26B-A4B (sparse MoE: 128 experts, top-8 routed "
            "+ 1 shared, 30 layers). Verified against the real "
            "google/gemma-4-26B-A4B-it checkpoint via meta-device "
            "Gemma4ForCausalLM construction + named_parameters()/"
            "named_modules() inspection, and a real tokenizer "
            "apply_chat_template() check."
        ),
        # q_proj/k_proj/v_proj/o_proj are plain nn.Linear (confirmed) -> empty
        # list lets verl/PEFT auto-detect ("all-linear").
        "target_modules": [],
        # router.proj (the per-layer expert gate) IS a plain nn.Linear
        # (shape [128, 2816]) and would be swept into "all-linear" by
        # default. Excluded here since perturbing routing during SFT is
        # usually undesirable — remove this entry in a profile override if
        # you deliberately want the router adapted too.
        "exclude_modules": ["router.proj"],
        # The 128 experts are fused 3D nn.Parameter tensors on a custom
        # Gemma4TextExperts module (experts.gate_up_proj shape
        # (128, 1408, 2816), experts.down_proj shape (128, 2816, 704)) —
        # NOT nn.Linear, so target_modules matching cannot reach them.
        # Without this, LoRA only touches attention + lm_head, missing the
        # model's dominant MoE capacity.
        "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
        # Confirmed: Gemma 4's thinking directive is a synthetic
        # system-turn insertion (`<|turn>system\n<|think|>\n<turn|>`), not
        # an assistant-prefix like Qwen3 — but the kwarg name that toggles
        # it in apply_chat_template() is the same: "enable_thinking".
        "thinking_kwarg": "enable_thinking",
        "trust_remote_code": True,
        # Without this, the first forward pass raises `RuntimeError:
        # FlashAttention forward only supports head dimension at most
        # 256`. Root-caused against the real checkpoint's config.json and
        # transformers 5.10.4's real modeling_gemma4.py: Gemma 4's hybrid
        # local/global attention gives
        # its 5 "full_attention" layers (of 30) a separate, larger
        # `global_head_dim: 512` (the other 25 "sliding_attention" layers
        # use head_dim: 256, which IS within flash-attn's limit). FlashAttn
        # 2.x's head_dim<=256 cap is universal across GPU architectures
        # (confirmed against the flash-attention v2.8.3 README, not
        # architecture-specific) and there is no version to pin around it —
        # this is a hard capability gap, not a bug. HF's Gemma4Attention
        # applies one uniform `config._attn_implementation` for every layer
        # (verl/workers/config/model.py's `override_config.get(
        # "attn_implementation", "flash_attention_2")` -> AutoConfig, single
        # value for the whole model), so there is no way to keep
        # flash_attention_2 for the 25 compatible layers only without
        # per-layer HF model surgery — out of scope for a config-driven
        # recipe. "sdpa" has no head_dim ceiling and is a generic,
        # model-agnostic backend already registered in
        # transformers.ALL_ATTENTION_FUNCTIONS, so it works for Gemma 4's
        # hybrid layers without any model-specific support needed.
        "attn_implementation": "sdpa",
    },
    "muse_glimmer": {
        "description": (
            "Meta Muse Glimmer 30B (dense, no MoE; hybrid local/global "
            "attention, 52 layers, NoPE on the global layers). Verified "
            "against the real meta-models/Muse-Glimmer-30B checkpoint via "
            "meta-device MuseGlimmerForConditionalGeneration construction + "
            "a real peft.LoraConfig build (0 vision hits, 416 correctly-"
            "targeted language_model.* modules), and the real chat_template."
        ),
        # q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj (text
        # attention + MLP) are plain nn.Linear with STANDARD names —
        # confirmed to already match the schema's own non-empty default
        # target_modules list, so no override is strictly required here.
        # Listed explicitly anyway for clarity/robustness against the
        # schema default ever changing.
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # MuseGlimmerVisionAttention ALSO uses q_proj/k_proj/v_proj (and
        # the vision adapter/projection are plain nn.Linear too), so the
        # target list above would reach into the vision tower as well.
        # MUST be a regex STRING here, not a list: PEFT's list form only
        # matches an exact trailing path segment
        # (key.endswith(f".{entry}")) and cannot express "anywhere under
        # this prefix" — confirmed empirically (a naive
        # ["vision_tower", ...] list silently excluded nothing at all).
        # The string form uses re.fullmatch instead.
        "exclude_modules": ".*(vision_tower|vision_adapter|vision_projection).*",
        "trust_remote_code": False,
        # No target_parameters: dense model, no fused-expert nn.Parameter
        # weights to reach.
        # No thinking_kwarg: the real chat_template.jinja has zero "think"
        # references — this model's reasoning/tool-calling protocol (ATEM)
        # expresses a reasoning channel via message-level fields
        # (reasoning_content), not a template kwarg toggle.
        # No attn_implementation override: head_dim=128 is well within
        # FlashAttention-2's cap (unlike Gemma 4's 512), and there are no
        # recurrent layers needing verl's special-cased monkey-patching
        # (see verl/models/transformers/monkey_patch.py — no "muse_glimmer"
        # entry exists there, and none is needed; this falls through to
        # the same generic path Gemma 4 already trains correctly under).
        #
        # The real checkpoint declares transformers_version: "5.15.0.dev0"
        # — confirmed empirically that MuseGlimmerForConditionalGeneration
        # doesn't exist below 5.15.0. verl==0.9.1's own setup.py caps
        # transformers<5.11 (a vLLM/SGLang import-crash workaround,
        # confirmed via verl's own source to be irrelevant to our vllm-free
        # sft_verl path — zero module-level vllm/sglang imports anywhere in
        # the code chain sft_verl touches). Consumed by
        # sft_verl.py::_ensure_transformers_version_compatible(), called
        # the same way/place as _ensure_flash_attn_compatible() — swaps the
        # already-installed (older, verl-constrained) transformers for
        # this version before verl's model-construction code ever runs.
        "min_transformers_version": "5.15.0",
        # verl 0.9.1's FSDPEngine._build_module reads
        # hf_config.tie_word_embeddings directly, assuming every HF config
        # exposes it at the top level — true for plain causal-LM configs,
        # but MuseGlimmerConfig (a composite VLM config) only has it on the
        # nested text_config, confirmed directly against the real
        # checkpoint (a real SageMaker run failed with exactly
        # AttributeError: 'MuseGlimmerConfig' object has no attribute
        # 'tie_word_embeddings' at that line). Passing tie_word_embeddings
        # via model.override_config does not help — the class never stores
        # it at the top level regardless of what's passed to
        # AutoConfig.from_pretrained. sft_verl.py's run() calls
        # verl_compat.patch_muse_glimmer_tie_word_embeddings() for this
        # profile specifically to route around it.
    },
}


def min_transformers_version(profile_name: "str | None") -> "str | None":
    """The minimum transformers version ``profile_name``'s architecture

    needs to even exist as importable classes (not a "recommended"
    floor — the model literally isn't defined below this version). Only
    set for profiles that needed a newer transformers than verl==0.9.1's
    own ``<5.11`` cap allows — see the ``muse_glimmer`` profile's own
    comment for the concrete verification behind this and the mechanism
    that consumes it
    (``foresightrlcore.methods.sft_verl._ensure_transformers_version_compatible``).
    ``None`` for any unset/unknown profile name — most model families need
    no override at all, since verl's own pin already covers them.
    """
    if profile_name is None:
        return None
    value = MODEL_PROFILES.get(profile_name, {}).get("min_transformers_version")
    return str(value) if value is not None else None


def resolve_model_profile(model: "ModelConfig") -> None:
    """Apply named profile defaults to a ``ModelConfig`` in-place.

    Called from ``ModelConfig``'s model_validator. Only fills fields the
    user didn't explicitly set in YAML: uses ``model.lora.model_fields_set``
    (pydantic's own "was this field actually provided" tracking) to
    distinguish "left at the LoraConfig_ default" from "explicitly
    provided" — this matters because ``target_modules`` defaults to a
    non-empty dense-model list for backward compatibility with existing
    GRPO configs, so an empty-list check alone would not reliably detect
    "not set by the user."
    """
    if model.profile is None:
        return

    if model.profile not in MODEL_PROFILES:
        available = sorted(MODEL_PROFILES.keys())
        raise ValueError(
            f"Unknown model profile: '{model.profile}'. Available: {available}"
        )

    profile = MODEL_PROFILES[model.profile]
    lora_fields_set = model.lora.model_fields_set

    if "target_modules" in profile and "target_modules" not in lora_fields_set:
        model.lora.target_modules = list(profile["target_modules"])
    if "exclude_modules" in profile and "exclude_modules" not in lora_fields_set:
        # A profile's exclude_modules may be a regex STRING (to exclude an
        # entire subtree PEFT's per-list-entry exact-suffix matching can't
        # express — see LoraConfig_.exclude_modules's docstring) rather
        # than a list — pass it through as-is, only copying actual lists.
        profile_exclude = profile["exclude_modules"]
        model.lora.exclude_modules = (
            list(profile_exclude)
            if isinstance(profile_exclude, list)
            else profile_exclude
        )
    if "target_parameters" in profile and "target_parameters" not in lora_fields_set:
        model.lora.target_parameters = list(profile["target_parameters"])
    if model.thinking_kwarg is None and "thinking_kwarg" in profile:
        model.thinking_kwarg = profile["thinking_kwarg"]
    if model.trust_remote_code is None and "trust_remote_code" in profile:
        model.trust_remote_code = profile["trust_remote_code"]
    if model.attn_implementation is None and "attn_implementation" in profile:
        model.attn_implementation = profile["attn_implementation"]
