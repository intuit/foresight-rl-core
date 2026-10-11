"""Merge a verl FSDP LoRA checkpoint into a plain HF model.

Why this exists instead of verl's own ``verl.model_merger`` tool: that tool's
``save_lora_adapter()`` (``verl/model_merger/base_model_merger.py``) infers
``adapter_config.json``'s ``target_modules`` purely from state-dict key
names. Verified directly against the real Gemma 4 architecture (meta
device, no GPU needed): for a ``target_parameters``-based MoE adapter
(``experts.gate_up_proj``/``experts.down_proj``), the real parameter key is
``...layers.N.experts.lora_A.default.weight`` — the tool extracts
``"experts"`` and writes it into ``target_modules``, but never reconstructs
``target_parameters`` at all. Reloading that config would try to wrap the
fused-parameter ``experts`` module as if it were a plain ``nn.Linear``
target, which it isn't. This is a real gap in verl 0.9.1's tooling for
``target_parameters``-style MoE LoRA, not something specific to our setup.

A full merge sidesteps the bug entirely — a merged model has no
adapter/config structure to reconstruct incorrectly. Verified the mechanism
itself works correctly for a ``target_parameters``-shaped fused-expert
module: built one with real (non-meta, randomly initialized) LoRA weights
and confirmed ``peft`` model's own ``merge_and_unload()`` changes both
fused-parameter tensors by real, non-trivial deltas, leaving zero ``lora_``
keys behind. That's genuine PEFT library code, not hand-rolled math.

Two pieces:

``reconstruct_full_state_dict`` — reuses verl's own
:class:`verl.model_merger.fsdp_model_merger.FSDPModelMerger` shard-
reconstruction machinery directly (the FSDP-shard/DTensor reassembly math,
which is generic and not the buggy part) instead of its
``save_hf_model_and_tokenizer``/``save_lora_adapter`` methods, which we
replace.

``merge_lora_and_save`` — builds the *same* ``peft.LoraConfig`` our own
training path already knows (from ``config.model.lora``, not from
ambiguous state-dict key names), loads the REAL pretrained base model
(via verl's own ``get_hf_auto_model_class`` dispatch — real weights, not
the checkpoint; not hardcoded to ``AutoModelForCausalLM``, since a
composite VLM config like Muse Glimmer's needs
``AutoModelForImageTextToText`` instead), applies the reconstructed
adapter weights on top via a non-strict ``load_state_dict``, calls peft's
own ``merge_and_unload()``, and saves a plain (non-PEFT) HF model +
tokenizer.

The checkpoint's own reconstructed state dict is deliberately NOT used as
the base-model weight source: training defaults ``save_lora_only`` to
``True`` (see ``VerlSftConfig.save_lora_only``), which saves ONLY the
adapter parameters per FSDP shard to keep checkpoints small (~a few GiB
vs ~54 GiB for a 27B model) — the checkpoint literally never contains the
frozen base weights at all in that (default) mode. Loading them from the
original pretrained model instead isn't a workaround for that case only:
for pure LoRA training the base weights are frozen and byte-identical to
the original pretrained checkpoint by construction, so it's the correct
source regardless of ``save_lora_only``.

Lazy-imports verl/torch/peft/transformers (same convention as
:mod:`foresightrlcore.methods.verl_sft_dataset`) so this module stays
importable without the GPU stack installed.
"""

from __future__ import annotations

import os
from typing import Any


def find_latest_checkpoint_step(checkpoint_root: str, step: int | None = None) -> str:
    """Resolve ``checkpoint_root/global_step_<N>``.

    ``step`` picks a specific step; ``None`` auto-detects the highest
    ``global_step_*`` directory present.
    """
    if step is not None:
        path = os.path.join(checkpoint_root, f"global_step_{step}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No checkpoint at {path!r}")
        return path

    candidates = []
    for name in os.listdir(checkpoint_root):
        if name.startswith("global_step_"):
            try:
                candidates.append((int(name[len("global_step_") :]), name))
            except ValueError:
                continue
    if not candidates:
        raise FileNotFoundError(
            f"No 'global_step_*' directories found under {checkpoint_root!r}"
        )
    _, latest_name = max(candidates)
    return os.path.join(checkpoint_root, latest_name)


def check_lora_train_meta_matches(checkpoint_dir: str, lora: Any) -> None:
    """Cross-check the merge config's ``model.lora`` against what verl's

    own checkpoint manager recorded during training
    (``lora_train_meta.json``, written unconditionally by
    ``FSDPCheckpointManager._save_lora_train_meta`` from the real trained
    ``peft_config.r``/``peft_config.lora_alpha`` — confirmed directly
    against verl 0.9.1's source, same file verl's own ``model_merger``
    reads for the identical reason via ``_load_lora_train_meta``).

    This specifically catches a copy-pasted merge config whose ``lora:``
    block has silently drifted from the SFT run's: an ``r`` mismatch would
    also surface later as a tensor-shape error inside
    ``merge_lora_and_save``, but an ``alpha`` mismatch would NOT — it only
    changes ``merge_and_unload()``'s scaling factor, with no missing or
    unexpected state-dict key and no shape error, silently producing a
    mis-scaled (not just suboptimal) merged model.

    Missing or unreadable metadata is not an error — an older verl version,
    or a training run that didn't actually use LoRA, might not have this
    file. This is a best-effort safety net on top of the example config's
    own "must exactly match" comment, not a hard requirement.
    """
    import json

    meta_path = os.path.join(checkpoint_dir, "lora_train_meta.json")
    if not os.path.exists(meta_path):
        return

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(meta, dict):
        return

    mismatches = []
    if "r" in meta and int(meta["r"]) != int(lora.r):
        mismatches.append(f"r: trained={meta['r']}, merge config={lora.r}")
    if "lora_alpha" in meta and int(meta["lora_alpha"]) != int(lora.alpha):
        mismatches.append(
            f"lora_alpha: trained={meta['lora_alpha']}, merge config={lora.alpha}"
        )

    if mismatches:
        raise ValueError(
            f"Merge config's model.lora does not match the actual training "
            f"run recorded in {meta_path!r}: {'; '.join(mismatches)}. Fix "
            "model.lora to match the SFT run's lora: block."
        )


def reconstruct_full_state_dict(
    checkpoint_dir: str, trust_remote_code: bool = True
) -> dict[str, Any]:
    """Reassemble the full (unsharded) model state dict from a verl FSDP
    checkpoint directory (``.../global_step_N``), reusing verl's own
    shard-reconstruction methods — see module docstring for why we call
    these directly instead of ``FSDPModelMerger.merge_and_save()``.
    """
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_dir,
        hf_model_config_path=os.path.join(checkpoint_dir, "huggingface"),
        trust_remote_code=trust_remote_code,
    )
    merger = FSDPModelMerger(config)

    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(
        rank_zero_state_dict, world_size
    )
    total_shards, mesh_shape = merger._calculate_shard_configuration(
        mesh, mesh_dim_names
    )
    return merger._load_and_merge_state_dicts(
        world_size, total_shards, mesh_shape, mesh_dim_names
    )


def _build_lora_config(lora: Any) -> Any:
    """Mirrors the exact target_modules/exclude_modules/target_parameters
    mapping already used for verl's own config in
    :meth:`foresightrlcore.methods.sft_verl.VerlSftMethodHandler.run` — same
    ``LoraConfig_`` input, just constructing a real ``peft.LoraConfig``
    directly instead of setting verl's Hydra config fields.
    """
    import peft

    kwargs: dict[str, Any] = {
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "target_modules": (
            list(lora.target_modules) if lora.target_modules else "all-linear"
        ),
    }
    if lora.exclude_modules:
        # A regex STRING (PEFT's re.fullmatch branch) excludes an entire
        # subtree that a plain list's exact-trailing-segment matching
        # can't express — only wrap actual lists; PEFT accepts a bare
        # string natively.
        kwargs["exclude_modules"] = (
            list(lora.exclude_modules)
            if isinstance(lora.exclude_modules, list)
            else lora.exclude_modules
        )
    if lora.target_parameters:
        kwargs["target_parameters"] = list(lora.target_parameters)
    return peft.LoraConfig(**kwargs)


def merge_lora_and_save(
    state_dict: dict[str, Any],
    base_model_path: str,
    lora: Any,
    output_dir: str,
    trust_remote_code: bool = True,
) -> None:
    """Load the REAL pretrained base model, apply the trained adapter

    weights reconstructed from the checkpoint on top, merge, and save a
    plain HF model.

    ``base_model_path`` must point at the ORIGINAL pretrained model (real
    weight files, e.g. the same S3-staged/local model.name the SFT job
    itself trained from) — not the checkpoint directory. See the module
    docstring for why: ``state_dict`` is adapter-only under the (default)
    ``save_lora_only=True`` path and never contains the frozen base
    weights at all.

    Raises if any LoRA adapter parameter the freshly built PEFT model
    expects is missing from ``state_dict``, or if ``state_dict`` contains
    a key the PEFT model doesn't recognize at all (wrong
    target_modules/target_parameters) — either would otherwise leave some
    adapter parameters at their random-init values, silently producing a
    broken (not just suboptimal) model. Missing *non*-adapter keys are
    expected and fine: those are the frozen base weights, already loaded
    for real above, that the checkpoint was never asked to save.

    Not hardcoded to ``AutoModelForCausalLM``: a composite VLM config
    (e.g. ``model.profile: muse_glimmer``'s ``MuseGlimmerConfig``) is
    registered under ``AutoModelForImageTextToText``, not
    ``AutoModelForCausalLM`` — the latter would raise
    ``ValueError: Unrecognized configuration class ...`` immediately, and
    was reproduced doing exactly that against a real merge job. Reuses
    verl's own ``get_hf_auto_model_class`` (``verl/utils/model.py``) for
    this dispatch — the *same* function
    ``VerlSftMethodHandler.run()``'s FSDP training path relies on
    (indirectly, via verl's own model construction) for the identical
    config, so the merge path can never disagree with the training path
    on which class a given checkpoint's base model needs.
    """
    import peft
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from verl.utils.model import get_hf_auto_model_class

    hf_config = AutoConfig.from_pretrained(
        base_model_path, trust_remote_code=trust_remote_code
    )
    auto_model_class = get_hf_auto_model_class(hf_config)
    base_model = auto_model_class.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )

    lora_config = _build_lora_config(lora)
    peft_model = peft.get_peft_model(base_model, lora_config)

    result = peft_model.load_state_dict(state_dict, strict=False)
    missing_adapter_keys = [k for k in result.missing_keys if "lora_" in k]
    if missing_adapter_keys or result.unexpected_keys:
        raise RuntimeError(
            "Reconstructed checkpoint state dict does not match the LoRA "
            f"config built from model.lora: missing_adapter_keys="
            f"{missing_adapter_keys}, unexpected={result.unexpected_keys}. "
            "Refusing to save a model that would have uninitialized "
            "adapter parameters."
        )

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)
