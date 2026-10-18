"""Version-specific workarounds for gaps in verl's own config surface.

Pinned to verl==0.9.1 (see ``foresightrlcore/config/requirements/verl_sft.txt``
for why that exact version). Everything here patches around a real gap in
that pinned version's own config/construction code — not method logic, so
it's kept separate from ``methods/sft_verl.py`` (which just calls into
this module) to make clear these are workarounds for an external
dependency's version, not part of this repo's own training pipeline. Some
of these gaps are closed by patching verl's own module namespace directly
(:func:`patch_verl_lora_dropout`); others by patching a *transformers*
config class instead (:func:`patch_muse_glimmer_tie_word_embeddings`) —
the common thread is that verl's own construction code makes an assumption
about its config surface that a specific model doesn't satisfy, and the
fix routes around that gap one way or the other.

Lazy-imports peft/verl (same convention as
:mod:`foresightrlcore.methods.verl_sft_dataset`) so this module stays
importable without the GPU stack installed.
"""

from __future__ import annotations

from typing import Any


def patch_verl_lora_dropout(dropout: float) -> None:
    """Forward ``model.lora.dropout`` into verl's FSDP LoRA construction.

    verl 0.9.1's ``FSDPEngine._build_lora_module``
    (``verl/workers/engine/fsdp/transformer_impl.py``) builds its
    ``peft.LoraConfig`` from a hardcoded dict of
    ``{task_type, r, lora_alpha, target_modules, target_parameters,
    exclude_modules, bias}`` — confirmed directly against the pinned
    verl==0.9.1 source — with no ``lora_dropout`` key at all, so PEFT's own
    ``LoraConfig`` default (0.0) always applies regardless of what's
    configured, unlike the TRL/GRPO path (``methods/base.py``) which passes
    ``lora_dropout`` explicitly. There is no config field on verl's
    ``HFModelConfig``/``hf_model.yaml`` to route a value through instead.

    ``LoraConfig`` is bound into ``transformer_impl``'s module namespace via
    ``from peft import LoraConfig`` and referenced there as a bare name, so
    Python resolves it from that module's globals at call time — reassigning
    the name here (to a subclass that defaults ``lora_dropout`` whenever the
    caller didn't pass one) takes effect without touching verl's own logic.
    Guarded by an identity check so an unexpected upstream restructuring
    (verl no longer importing ``LoraConfig`` that way) fails loudly instead
    of silently leaving dropout at 0.0.
    """
    from peft import LoraConfig
    from verl.workers.engine.fsdp import transformer_impl

    if transformer_impl.LoraConfig is not LoraConfig:
        raise RuntimeError(
            "Cannot patch verl's LoRA dropout: "
            "verl.workers.engine.fsdp.transformer_impl.LoraConfig is not "
            "peft.LoraConfig (verl upstream may have changed how it builds "
            "the LoRA config) — model.lora.dropout would otherwise be "
            "silently ignored."
        )

    class _LoraConfigWithDropout(LoraConfig):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("lora_dropout", dropout)
            super().__init__(*args, **kwargs)

    transformer_impl.LoraConfig = _LoraConfigWithDropout


def patch_muse_glimmer_tie_word_embeddings() -> None:
    """Expose ``tie_word_embeddings`` at the top level of ``MuseGlimmerConfig``.

    verl 0.9.1's ``FSDPEngine._build_module``
    (``verl/workers/engine/fsdp/transformer_impl.py``) reads
    ``self.model_config.hf_config.tie_word_embeddings`` directly, assuming
    every HF config exposes this field at the top level — true for plain
    causal-LM configs, but not for ``MuseGlimmerConfig``: a composite VLM
    config that nests text-generation fields (including
    ``tie_word_embeddings``) under ``text_config`` instead of mirroring
    them to the top level. Confirmed directly against the real
    ``meta-models/Muse-Glimmer-30B`` checkpoint's config — even passing
    ``tie_word_embeddings=...`` as an ``AutoConfig.from_pretrained`` kwarg
    (i.e. via ``model.override_config``) does not make it stick as a
    top-level attribute, since the class simply never stores it there.
    Reproduced directly: a real SageMaker run failed with
    ``AttributeError: 'MuseGlimmerConfig' object has no attribute
    'tie_word_embeddings'`` at exactly that line.

    ``MuseGlimmerConfig`` defines no ``__getattr__`` of its own (only
    ``__getattribute__`` overrides, for an unrelated per-layer-heterogeneity
    feature) — Python only calls ``__getattr__`` as a fallback once normal
    attribute lookup already raised ``AttributeError``, so adding one here
    cannot change behavior for any attribute that already resolves
    normally; it only fills in this one specific gap. Guarded by an
    identity check on ``__getattr__``'s absence so an unexpected upstream
    addition (transformers starts defining its own) fails loudly instead
    of silently overwriting it.
    """
    from transformers.models.muse_glimmer.configuration_muse_glimmer import (
        MuseGlimmerConfig,
    )

    if "__getattr__" in MuseGlimmerConfig.__dict__:
        raise RuntimeError(
            "Cannot patch MuseGlimmerConfig.tie_word_embeddings: "
            "MuseGlimmerConfig already defines its own __getattr__ "
            "(transformers upstream may have changed) — "
            "tie_word_embeddings would otherwise be silently shadowed."
        )

    def __getattr__(self: Any, name: str) -> Any:
        if name == "tie_word_embeddings":
            return self.text_config.tie_word_embeddings
        raise AttributeError(name)

    MuseGlimmerConfig.__getattr__ = __getattr__
