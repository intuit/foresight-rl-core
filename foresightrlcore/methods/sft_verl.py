"""verl FSDP SFT method handler.

Unlike GRPO (``foresightrlcore/methods/grpo.py``), this handler does **not**
subclass :class:`~foresightrlcore.methods.base.TrlMethodHandler` — that base
class's ``run()`` pipeline is TRL-``Trainer``-shaped end to end (builds a TRL
``Trainer`` subclass, calls ``.train()``/``.save_model()``). verl's SFT
trainer is a different animal: a plain-torchrun, Hydra/OmegaConf-configured
job (``verl.trainer.sft_trainer.SFTTrainer``) with its own dataset/engine/
optimizer/checkpoint composition. This handler implements the bare
``MethodHandler`` protocol directly and drives verl's trainer in-process.

Launch mechanics: verl's ``SFTTrainer`` expects ``torch.distributed`` to
already be initialized via plain ``torchrun`` env vars (``RANK``/
``WORLD_SIZE``/``MASTER_ADDR``/...) — confirmed from verl's own
``run_sft()``/``main()`` (``verl/trainer/sft_trainer.py``), which calls
``initialize_global_process_group()`` then constructs ``SFTTrainer`` and
calls ``.fit()``. This is exactly what SageMaker's SDK-native
``Torchrun(process_count_per_node=...)`` distribution already provides for
the ``aws_training_job`` backend when ``infra.vllm_mode: none`` — one
``train.py`` process per GPU, with those env vars pre-set. So this handler
calls verl's trainer construction directly rather than shelling out to a
nested ``torchrun -m verl.trainer.sft_trainer`` subprocess (which would
double-nest process groups).

All verl/hydra/omegaconf imports are function-local (see
:mod:`foresightrlcore.methods.verl_sft_dataset` docstring for why) so this
module stays importable without the GPU stack installed.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

from foresightrlcore.config.model_profiles import min_transformers_version
from foresightrlcore.methods import register_method

logger = logging.getLogger(__name__)

_HYDRA_CONFIG_MODULE = "verl.trainer.config"
_HYDRA_CONFIG_NAME = "sft_trainer_engine"

# The verl_sft recipe's DLC base image ships a pre-installed flash-attn
# built against an older CUDA runtime (ImportError: libcudart.so.12).
# Pinning `flash-attn @ <url>` in requirements_torchrun.txt is not
# sufficient — the SageMaker training toolkit's automatic
# `pip install -r requirements.txt` silently keeps the pre-existing
# install: pip downloads the correct wheel to read its metadata, then
# omits it from "Installing collected packages" because another
# already-satisfied package (transformer_engine) references the existing
# flash-attn — a pip batch-resolver quirk that can't be influenced from
# inside a requirements.txt file. A custom Docker image would give full
# control over pip invocation order, but building one isn't always
# possible (e.g. an IAM boundary blocking the base image pull) — so this
# forces the reinstall here, in code, with an isolated `pip uninstall` +
# `--force-reinstall` invocation, run once per node before verl's
# flash-attn-triggering import chain executes.
#
# The wheel URL is resolved per-GPU-family (foresightrlcore.config.recipes.
# resolve_flash_attn_wheel), not hardcoded here — see that module's
# GPU_FAMILIES/FLASH_ATTN_WHEELS docstring for why the same cu13 build
# already covers Ampere (p4d/p4de)/Hopper (p5/p5e/p5en)/Blackwell (p6/p6e)
# today, and how to point a family at a different build once one is
# verified against real hardware.


def _torchrun_launch_id() -> str:
    """A value shared by every local rank of ONE torchrun launch, but

    different across separate launches — even when they reuse the same
    container/instance (SageMaker Managed Warm Pools with
    infra.keep_alive_seconds > 0, or repeated ``rl-train train`` runs
    against the same "local" GPU box). torchrun forks/spawns every local
    rank as a direct child of itself, so ``os.getppid()`` is identical
    across LOCAL_RANK 0-7 on one node for one launch (confirmed against
    torchrun's own LocalElasticAgent process model) and gets a fresh value
    from the OS on the next launch. Used to namespace the
    :func:`_ensure_flash_attn_compatible`/
    :func:`_ensure_transformers_version_compatible` sentinel files so a
    /tmp leftover from an unrelated, earlier job on a reused container can
    never be mistaken for THIS run's sentinel — closing a real race where
    a non-zero rank's plain ``os.path.exists(sentinel)`` check could see a
    stale "done" file before rank0 gets around to deleting it (rank0's
    delete-then-rebuild isn't synchronized against other ranks' checks by
    anything — torch.distributed isn't initialized yet at this point in
    ``run()``, so no ``dist.barrier()`` is available to use instead).
    """
    return str(os.getppid())


def _ensure_flash_attn_compatible(wheel_url: str) -> None:
    """Force-reinstall flash-attn with the given (GPU-family-resolved)
    wheel, once per node, before any verl import touches it.

    Mirrors the LOCAL_RANK==0-does-the-work / others-wait-on-a-sentinel
    pattern already used for checkpoint resume (see
    :func:`foresightrlcore.methods.base._download_resume_checkpoint`) — 8
    local ranks share one container filesystem per node, so only one of
    them should run pip.
    """
    import subprocess
    import time

    run_id = _torchrun_launch_id()
    sentinel = f"/tmp/flash_attn_fix.{run_id}.done"
    failure = f"/tmp/flash_attn_fix.{run_id}.failed"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        for p in (sentinel, failure):
            if os.path.exists(p):
                os.remove(p)
        try:
            logger.info(
                f"[rank0] Forcing flash-attn to the verified build: {wheel_url}"
            )
            subprocess.run(["pip", "uninstall", "-y", "flash-attn"], check=False)
            subprocess.run(
                [
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    wheel_url,
                ],
                check=True,
            )
            with open(sentinel, "w") as f:
                f.write("ok")
            logger.info("[rank0] flash-attn fix applied")
        except Exception as e:
            with open(failure, "w") as f:
                f.write(str(e))
            raise
    else:
        timeout_s = int(os.environ.get("FLASH_ATTN_FIX_TIMEOUT", "600"))
        start = time.time()
        logger.info(
            f"[rank{local_rank}] Waiting for rank0 to fix flash-attn (timeout={timeout_s}s)"
        )
        while True:
            if os.path.exists(sentinel):
                logger.info(f"[rank{local_rank}] flash-attn fix ready")
                break
            if os.path.exists(failure):
                raise RuntimeError(f"rank0 failed to fix flash-attn (see {failure})")
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for rank0's flash-attn fix"
                )
            time.sleep(2)


def _ensure_transformers_version_compatible(min_version: str) -> None:
    """Swap the already-installed transformers for ``min_version`` (or

    newer), once per node, before any verl model-construction code runs.

    Why this exists: verl==0.9.1's own ``setup.py`` caps
    ``transformers<5.11`` — a workaround for a vLLM/SGLang import crash
    with certain kernel versions (confirmed via verl's own upstream PR
    history), which does not apply to ``sft_verl``'s vllm-free FSDP-SFT
    code path (confirmed directly: zero module-level ``import vllm``/
    ``import sglang`` anywhere in the code chain this method touches).
    Some newer model architectures (e.g. Muse Glimmer) only exist in
    transformers versions above that cap, so the container's
    ``requirements.txt`` bulk install (which lists plain ``verl==0.9.1``
    and therefore transitively resolves to verl's own capped transformers
    range) needs a follow-up swap — this function is that swap, called
    right alongside :func:`_ensure_flash_attn_compatible` for the same
    reason: 8 local ranks share one container filesystem per node, so
    only one of them should run pip (mirrors the
    LOCAL_RANK==0-does-the-work / others-wait-on-a-sentinel pattern used
    there and in
    :func:`foresightrlcore.methods.base._download_resume_checkpoint`).

    A no-op if the installed transformers already satisfies
    ``min_version`` (idempotent — safe to call on every job start).

    Deliberately checks the installed version via
    ``importlib.metadata.version`` rather than ``import transformers;
    transformers.__version__`` — this function runs in the same process
    that will later do the real ``from transformers import ...`` (inside
    verl's own model-construction code), and ``transformers/__init__.py``
    eagerly executes ``from .utils import (...)`` at import time (not
    lazily, confirmed directly against the real wheel). So merely
    importing the package here to read its version, pre-swap, would cache
    the OLD ``transformers``/``transformers.utils`` modules in
    ``sys.modules`` — and Python never re-imports an already-cached
    module, even after pip replaces the on-disk files underneath it. The
    real training import later only re-triggers a handful of
    lazily-loaded submodules (``transformers.video_utils``,
    ``transformers.processing_utils``, ...), which then get the *new*
    on-disk files while `transformers.utils` itself stays the *stale*
    cached pre-swap module — an inconsistent mix that surfaces as
    ``ImportError: cannot import name 'is_torchvision_greater_or_equal'
    from 'transformers.utils'`` (a symbol the older, pre-swap version
    never defined) the moment verl's checkpoint-handling code does
    ``from transformers import PreTrainedTokenizer, ProcessorMixin``.
    Reproduced directly: a real SageMaker run failed with exactly this
    traceback; downloading and diffing the transformers 5.10.4 vs 5.17.0
    wheels confirmed 5.10.4's ``transformers/utils/__init__.py`` has no
    ``is_torchvision_greater_or_equal`` (only the older
    ``is_torch_greater_or_equal``, which is what the error's own "Did you
    mean" suggested) while 5.17.0 defines and exports it consistently
    across every file that references it — i.e. 5.17.0 alone is
    internally fine; the bug was this function's own premature import of
    the pre-swap package poisoning the module cache for the rest of the
    process.

    The version check itself must happen *only* inside the rank0 branch
    below, not before the branch runs for every rank — confirmed directly
    against a second real SageMaker failure: checking
    ``importlib.metadata.version("transformers")`` unconditionally, before
    the rank0/wait split, has every one of the 8 local ranks race
    rank0's own ``pip uninstall`` + ``pip install`` sequence. ``pip``
    uninstalls the old dist-info *before* installing the new one, so
    there's a real (if short) window where no ``transformers`` metadata
    exists on disk at all — any other rank whose independent version
    check lands in that window gets
    ``importlib.metadata.PackageNotFoundError: No package metadata was
    found for transformers`` instead of a stale-but-harmless old-version
    read. Only rank0 may ever query or mutate the installed version;
    every other rank just waits on the sentinel, exactly like
    :func:`_ensure_flash_attn_compatible`.

    The swap's own ``pip install`` deliberately does *not* pass
    ``--no-deps`` (unlike :func:`_ensure_flash_attn_compatible`'s wheel
    install) — confirmed directly against a third real SageMaker failure:
    a newer transformers hard-requires a newer ``tokenizers`` too (checked
    eagerly at import time by transformers' own
    ``dependency_versions_check.py``), and verl's own resolved environment
    only has the older ``tokenizers`` version its *original*
    ``transformers<5.11`` constraint was satisfied by. ``--no-deps`` would
    skip bumping it, so the swapped-in transformers immediately fails its
    own internal version check with ``ImportError: tokenizers>=X,<Y is
    required ..., but found tokenizers==<old>``. transformers' actual
    dependency set (checked directly against the real wheel metadata) is
    small and framework-agnostic — huggingface-hub/numpy/packaging/pyyaml/
    regex/tokenizers/typer/safetensors/tqdm, no ``torch`` — so letting pip
    resolve it normally here can't disturb the pinned CUDA/torch/flash-attn
    stack the way an unconstrained upgrade of a heavier package might.
    """
    import subprocess
    import time
    from importlib.metadata import version as installed_pkg_version

    from packaging.version import Version

    run_id = _torchrun_launch_id()
    sentinel = f"/tmp/transformers_version_fix.{run_id}.done"
    failure = f"/tmp/transformers_version_fix.{run_id}.failed"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        for p in (sentinel, failure):
            if os.path.exists(p):
                os.remove(p)
        try:
            installed_version = installed_pkg_version("transformers")
            if Version(installed_version) >= Version(min_version):
                logger.info(
                    f"transformers=={installed_version} already satisfies "
                    f">={min_version}; no swap needed."
                )
            else:
                logger.info(
                    f"[rank0] Swapping transformers=={installed_version} "
                    f"for >={min_version} (verl's own <5.11 cap doesn't "
                    "apply to this vllm-free path — see this function's "
                    "docstring)."
                )
                subprocess.run(
                    [
                        "pip",
                        "install",
                        "--no-cache-dir",
                        f"transformers>={min_version}",
                    ],
                    check=True,
                )
                logger.info("[rank0] transformers version swap applied")
            with open(sentinel, "w") as f:
                f.write("ok")
        except Exception as e:
            with open(failure, "w") as f:
                f.write(str(e))
            raise
    else:
        timeout_s = int(os.environ.get("TRANSFORMERS_VERSION_FIX_TIMEOUT", "600"))
        start = time.time()
        logger.info(
            f"[rank{local_rank}] Waiting for rank0 to swap transformers "
            f"(timeout={timeout_s}s)"
        )
        while True:
            if os.path.exists(sentinel):
                logger.info(f"[rank{local_rank}] transformers swap ready")
                break
            if os.path.exists(failure):
                raise RuntimeError(f"rank0 failed to swap transformers (see {failure})")
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for rank0's "
                    "transformers version swap"
                )
            time.sleep(2)


def _expected_parquet_paths(files: list[str], cache_dir: str) -> list[str]:
    """The exact ``.parquet`` output path each ``files[i]`` maps to, keyed

    by its CURRENT content (path + mtime + size), computed fresh from a
    real ``os.stat`` every time this is called — never read from a
    manifest. Used by :func:`_cache_jsonl_as_parquet` on both branches:
    rank0 to decide what to build, and non-zero ranks to verify a manifest
    they're about to trust actually reflects THIS run's file content, not
    a stale one.
    """
    import hashlib

    paths = []
    for f in files:
        stat = os.stat(f)
        cache_key = hashlib.sha1(
            f"{f}:{stat.st_mtime_ns}:{stat.st_size}".encode()
        ).hexdigest()[:16]
        paths.append(os.path.join(cache_dir, f"{cache_key}.parquet"))
    return paths


def _cache_jsonl_as_parquet(files: list[str]) -> list[str]:
    """Convert JSONL files to Parquet once per node, so all local ranks read
    the much-cheaper binary format instead of each redundantly re-parsing
    the same JSON text.

    The underlying files are already local (SageMaker's input-channel
    mechanism downloads each channel once per NODE, not per rank) — the
    waste is 8 local ranks per node each independently calling
    ``pd.read_json`` on the same bytes. That's real CPU cost at scale (the
    production recipe's 1M-row/~18GB corpus, not just this smoke test's
    100K/1.26GB one). Same LOCAL_RANK==0-does-the-work /
    others-wait-on-a-sentinel pattern as
    :func:`_ensure_flash_attn_compatible` /
    :func:`foresightrlcore.methods.base._download_resume_checkpoint` —
    except the "done" sentinel/manifest here are deliberately NOT
    namespaced by :func:`_torchrun_launch_id` the way those two functions'
    sentinels (both done AND failed) are: this cache is meant to survive a
    job restart on the same node (a Spot interruption resume is a fresh
    torchrun launch, but should reuse a huge already-converted corpus
    rather than repay the ~18GB conversion cost again). The FAILURE marker
    is the one exception — it IS namespaced by launch id (see below) —
    since, unlike a completed cache, a failure has no content to verify
    before trusting it.

    That persistence-across-launches is exactly what makes this cache_dir
    (keyed only by the SORTED file *paths*) collide across two DIFFERENT,
    unrelated jobs on a reused container/warm pool: SageMaker mounts every
    input channel at the same fixed local path
    (``/opt/ml/input/data/<channel>/...``) regardless of which job or
    dataset it is, so a completely different job's leftover
    ``manifest.done``/``manifest.json`` can sit at this exact same
    cache_dir. Guarding against THAT (rather than namespacing by launch
    id, which would defeat the whole point above) is what
    :func:`_expected_parquet_paths` is for: every reader — rank0 deciding
    what to (re)build, and non-zero ranks about to trust a manifest they
    didn't write — recomputes the current, content-derived expected
    output paths from ``files``' real ``os.stat()`` and only trusts a
    manifest whose content matches exactly. A stale manifest from a
    different job's different data fails that comparison and is treated
    as not-ready-yet (poll on), never silently returned.

    Returns the list of cached ``.parquet`` paths, in the same order as
    ``files``. Idempotent: re-running with the same input paths AND the
    same file content reuses any already-converted file, so a job restart
    on the same node doesn't reconvert from scratch.
    """
    import json
    import time

    if not files:
        return files

    import hashlib

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dir_key = hashlib.sha1("\n".join(sorted(files)).encode()).hexdigest()[:16]
    cache_dir = f"/tmp/verl_sft_parquet_cache/{dir_key}"
    manifest = os.path.join(cache_dir, "manifest.json")
    sentinel = os.path.join(cache_dir, "manifest.done")
    # Unlike sentinel/manifest (deliberately NOT namespaced by
    # _torchrun_launch_id — see this function's docstring on why a
    # content-verified "done" cache should survive a job restart), a
    # failure marker must never be reused across launches: a non-zero
    # rank's plain os.path.exists(failure) check has no content to verify
    # a failure message against, so a stale manifest.failed left by an
    # EARLIER, unrelated job on this same reused container/path (e.g. one
    # that genuinely failed on different data) would abort this run's
    # ranks before rank0 even starts — reproduced by inspection: rank0's
    # own cleanup (immediately below) races the exact same way the
    # flash-attn/transformers sentinels did before they were namespaced.
    failure = os.path.join(cache_dir, f"manifest.failed.{_torchrun_launch_id()}")

    if local_rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        for p in (sentinel, failure):
            if os.path.exists(p):
                os.remove(p)
        try:
            import pandas as pd

            expected_paths = _expected_parquet_paths(files, cache_dir)
            for f, out in zip(files, expected_paths):
                if not os.path.exists(out):
                    logger.info(
                        f"[rank0] Caching {f} -> {out} (parquet, one-time per node)"
                    )
                    pd.read_json(f, lines=True).to_parquet(out)
            with open(manifest, "w") as mf:
                json.dump(expected_paths, mf)
            with open(sentinel, "w") as sf:
                sf.write("ok")
            logger.info(f"[rank0] parquet cache ready ({len(expected_paths)} file(s))")
            return expected_paths
        except Exception as e:
            with open(failure, "w") as ff:
                ff.write(str(e))
            raise
    else:
        timeout_s = int(os.environ.get("PARQUET_CACHE_TIMEOUT", "600"))
        start = time.time()
        logger.info(
            f"[rank{local_rank}] Waiting for rank0's parquet cache (timeout={timeout_s}s)"
        )
        while True:
            if os.path.exists(sentinel):
                expected_paths = _expected_parquet_paths(files, cache_dir)
                with open(manifest) as mf:
                    manifest_paths = json.load(mf)
                if manifest_paths == expected_paths:
                    return manifest_paths
                # Content mismatch: this manifest was written for a
                # DIFFERENT job's data that happened to land at the same
                # SageMaker channel path (warm-pool/container reuse) —
                # rank0 is (or is about to start) rebuilding it for THIS
                # job's actual files. Not a failure, just not ready yet.
                logger.info(
                    f"[rank{local_rank}] Ignoring a manifest for different "
                    "file content (stale from another job on this "
                    "container) — still waiting for rank0's current build."
                )
            if os.path.exists(failure):
                raise RuntimeError(
                    f"rank0 failed to build the parquet cache (see {failure})"
                )
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for rank0's parquet cache"
                )
            time.sleep(2)


class VerlSftMethodHandler:
    """SFT via verl's FSDP trainer (``verl.trainer.sft_trainer``), torchrun-launched."""

    method_type = "sft_verl"

    def validate(self, config: Any) -> list[str]:
        """Thin wrapper delegating to ``config.validation``'s canonical

        checks. Both functions delegated to are config-layer-owned
        (matching the established one-directional dependency in this
        codebase — ``methods/`` already imports from ``config/`` in
        several places, e.g. ``config.recipes``/``config.model_profiles``
        below; ``config/`` never imports from ``methods/``) and are the
        copies actually reachable from the real ``rl-train validate``/
        ``rl-train train`` path (``validate_config()``, which calls both
        as separate pipeline steps — mirrored here so calling this method
        standalone sees the same checks). This method used to be a full,
        independent ~150-line duplicate of those same checks — kept only
        as a protocol-conformance shim now, so the two can't silently
        drift apart. The two section-existence messages below are kept
        verbatim (not delegated) since existing tests assert on this
        exact wording.
        """
        if getattr(config, "training", None) is None:
            return ["Training config section is missing."]

        if getattr(config.training, "sft_verl", None) is None:
            return ["training.sft_verl section is missing."]

        from foresightrlcore.config.validation import (
            _validate_data_required,
            _validate_sft_verl,
        )

        # _validate_data_required is generic (data.train_path/output.s3_path
        # for any non-serverless method, with its own merge_lora_only
        # exemption already built in) — validate_config() calls it as a
        # separate pipeline step, not as part of _validate_sft_verl, so it
        # must be called here too for standalone-handler-call parity.
        return _validate_sft_verl(config) + _validate_data_required(config)

    def build_trainer_args(self, config: Any) -> dict[str, Any]:
        """Not used by this handler — kept only to satisfy the Protocol.

        verl's Hydra/OmegaConf config is built directly in ``run()`` since it
        has no TRL ``TrainingArguments``-shaped equivalent.
        """
        return {}

    def run(self, config: Any) -> None:
        sft_cfg = config.training.sft_verl
        lora = config.model.lora

        # Needed by BOTH the merge and the training path below — merge
        # loads the base model via plain transformers AutoModel classes
        # (see merge_lora_and_save), which for a model.profile like
        # muse_glimmer only exist in transformers>=5.15.0, not the
        # container's verl-capped install. Must run before the
        # merge_lora_only dispatch, not after it: reproduced directly
        # against a real SageMaker merge job, which otherwise fails trying
        # to import Muse Glimmer's config/model classes on the
        # unswapped, too-old transformers.
        min_transformers = min_transformers_version(config.model.profile)
        if min_transformers is not None:
            _ensure_transformers_version_compatible(min_transformers)

        # merge_lora_only is a single-process CPU/RAM-bound reconstruct +
        # merge, not a training run — it never touches the GPU/attention
        # path at all, so it must not need a GPU_FAMILIES-recognized
        # instance_type (examples/smoke_tests/sft_verl_merge_smoke.yaml's
        # own header explicitly allows a cheaper CPU-only instance for this
        # mode).
        # Checked and dispatched before the flash-attn/hydra/dataset setup
        # below so an unrecognized instance_type doesn't fail a merge job
        # only after the cluster is already up and billed — see
        # _validate_sft_verl's matching merge_lora_only exemption for the
        # GPU-family check.
        if sft_cfg.merge_lora_only:
            self._run_merge_lora_only(config, sft_cfg, lora)
            return

        # Unconditional regardless of model.attn_implementation — this is
        # NOT about which attention backend actually runs at inference/
        # training time, it's about import-time safety. verl.trainer.
        # sft_trainer.SFTTrainer's own import chain (sft_trainer.py ->
        # verl.utils.dataset.multiturn_sft_dataset ->
        # verl.models.transformers.qwen2_vl -> `from flash_attn import
        # flash_attn_func, flash_attn_varlen_func`) unconditionally imports
        # flash_attn at module load, for every sft_verl job, even one whose
        # model.profile forces sdpa (Gemma 4) and never calls into
        # flash_attn's kernels at runtime. Reproduced directly on a real
        # SageMaker job (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-
        # 20260929150913, ml.p4de.24xlarge, attn_implementation=sdpa):
        # gating this call on attn_implementation left the container's
        # generically pip-resolved flash-attn (linked against
        # libcudart.so.12) in place, and `import flash_attn` crashed with
        # `ImportError: libcudart.so.12: cannot open shared object file` on
        # both nodes before a single training step ran — this cu130 DLC
        # image has no libcudart.so.12 at all. A prior version of this
        # function gated the call on attn_implementation to avoid the
        # rank-0 pip uninstall/install cost on jobs that don't need
        # flash-attn's kernels; that trade-off is invalid given verl's own
        # unconditional import, so the call runs for every training job now.
        from foresightrlcore.config.recipes import resolve_flash_attn_wheel

        _ensure_flash_attn_compatible(
            resolve_flash_attn_wheel(config.infra.instance_type)
        )

        if config.model.profile == "muse_glimmer":
            # A pure in-memory class patch (no filesystem/pip work), unlike
            # the two fixes above — every rank's own process applies it
            # independently, no LOCAL_RANK==0/sentinel coordination needed.
            # Must run after the transformers swap above (needs the
            # muse_glimmer architecture actually importable) and before
            # verl's own SFTTrainer construction reads
            # hf_config.tie_word_embeddings — see this function's docstring.
            from foresightrlcore.methods.verl_compat import (
                patch_muse_glimmer_tie_word_embeddings,
            )

            patch_muse_glimmer_tie_word_embeddings()

        import hydra
        from omegaconf import OmegaConf

        from foresightrlcore.methods.verl_sft_dataset import (  # noqa: F401
            FlatPromptResponseSFTDataset,
            PackedSFTDataset,
        )

        if sft_cfg.data_format == "packed":
            # Already tokenized and packed offline (rl-train tokenize-pack)
            # — read the Parquet directly, no JSONL-to-Parquet caching step.
            # filename_prefix disambiguates training_*/validation_* when
            # train_path and eval_path point at the same flat output dir
            # (tokenize-pack's default layout) — see _glob_data_files.
            _check_packed_data_matches_config(
                str(config.data.train_path),
                config.model.enable_thinking,
                config.model.thinking_kwarg or "enable_thinking",
                sft_cfg,
            )
            # Checked separately, not assumed to be covered by the
            # train_path check above — eval_path can be a genuinely
            # different prefix (e.g. a validation set packed via its own
            # `rl-train tokenize-pack --split-name validation` invocation,
            # rather than split out of the same --input as training), with
            # its own *_metadata.jsonl sidecar. Without this, a thinking/
            # max_length mismatch on eval-only data would never be caught,
            # even though the exact same silent-wrong-chat-template-
            # rendering risk applies to it. A no-op when eval_path equals
            # train_path (same sidecar, already verified above).
            if config.data.eval_path and str(config.data.eval_path) != str(
                config.data.train_path
            ):
                _check_packed_data_matches_config(
                    str(config.data.eval_path),
                    config.model.enable_thinking,
                    config.model.thinking_kwarg or "enable_thinking",
                    sft_cfg,
                )
            train_files = _glob_data_files(
                str(config.data.train_path), "parquet", filename_prefix="training_"
            )
            val_files = (
                _glob_data_files(
                    str(config.data.eval_path), "parquet", filename_prefix="validation_"
                )
                if config.data.eval_path
                else None
            )
        else:
            train_files = _cache_jsonl_as_parquet(
                _glob_data_files(str(config.data.train_path))
            )
            val_files = (
                _cache_jsonl_as_parquet(_glob_data_files(str(config.data.eval_path)))
                if config.data.eval_path
                else None
            )

        output_dir = getattr(config, "_local_output_dir", None) or config.output.s3_path
        final_dir = getattr(config, "_final_model_dir", output_dir)

        logger.info(
            f"sft_verl training — model={config.model.name}, "
            f"train_files={len(train_files)} file(s), output_dir={output_dir}"
        )

        with hydra.initialize_config_module(
            config_module=_HYDRA_CONFIG_MODULE, version_base=None
        ):
            cfg = hydra.compose(config_name=_HYDRA_CONFIG_NAME)

        # --- model -----------------------------------------------------
        cfg.model.path = config.model.name
        # model.trust_remote_code is resolved from model.profile (see
        # foresightrlcore.config.model_profiles) when set in YAML; default
        # True when no profile/explicit value is given.
        cfg.model.trust_remote_code = (
            config.model.trust_remote_code
            if config.model.trust_remote_code is not None
            else True
        )
        cfg.model.use_remove_padding = sft_cfg.use_remove_padding
        if config.model.attn_implementation is not None:
            # See ModelConfig.attn_implementation / the gemma4_moe profile's
            # docstring (config/model_profiles.py) for why some models need
            # this (flash-attn's universal head_dim<=256 cap vs Gemma 4's
            # 512-dim global-attention layers). override_config is a real
            # key on hf_model.yaml (default {}), consumed by
            # verl/workers/config/model.py's HFModelConfig to build
            # AutoConfig.from_pretrained(attn_implementation=...).
            cfg.model.override_config = {
                "attn_implementation": config.model.attn_implementation
            }
        cfg.model.enable_gradient_checkpointing = config.training.gradient_checkpointing
        # See VerlSftConfig.use_liger's docstring for why this is safe to
        # default on generically (verified against liger-kernel's real
        # model-type registry, not assumed).
        cfg.model.use_liger = sft_cfg.use_liger
        cfg.model.lora_rank = lora.r
        cfg.model.lora_alpha = lora.alpha
        # verl's own auto-detect sentinel is the string "all-linear" (its
        # hf_model.yaml default) — NOT None/empty-list like PEFT's TRL-side
        # convention used in methods/base.py. An empty target_modules list in
        # our schema means "let verl/PEFT auto-detect", so it maps to the
        # string verl actually expects, not to None.
        cfg.model.target_modules = (
            list(lora.target_modules) if lora.target_modules else "all-linear"
        )
        if lora.exclude_modules:
            # A regex STRING excludes an entire subtree (PEFT's
            # re.fullmatch branch) — e.g. a nested vision tower a plain
            # list's exact-trailing-segment matching can't express (see
            # LoraConfig_.exclude_modules's docstring). Only wrap actual
            # lists; passing a string through unwrapped is what verl's own
            # HFModelConfig.exclude_modules field (Optional[str]) expects
            # anyway.
            cfg.model.exclude_modules = (
                list(lora.exclude_modules)
                if isinstance(lora.exclude_modules, list)
                else lora.exclude_modules
            )
        if lora.target_parameters:
            # For MoE models whose expert FFN weights are fused 3D
            # nn.Parameter tensors (confirmed directly for Gemma 4),
            # target_modules matching cannot reach them.
            # verl's FSDP engine threads this straight into
            # peft.LoraConfig(target_parameters=...) — see
            # verl/workers/engine/fsdp/transformer_impl.py. Plain attribute
            # assignment fails here: `target_parameters` is a real field on
            # verl's HFModelConfig dataclass but is NOT present in the
            # Hydra-composed model/hf_model.yaml (confirmed against the
            # installed package), so cfg.model is in struct mode for this
            # key — force_add=True is required to set it.
            OmegaConf.update(
                cfg,
                "model.target_parameters",
                list(lora.target_parameters),
                force_add=True,
            )

        # --- engine (FSDP) ----------------------------------------------
        cfg.engine.ulysses_sequence_parallel_size = (
            sft_cfg.ulysses_sequence_parallel_size
        )
        # Defaults to "fsdp2" rather than verl's own "fsdp" default — see
        # VerlSftConfig.fsdp_strategy's docstring (config/schema.py) for why.
        cfg.engine.strategy = sft_cfg.fsdp_strategy

        # --- optim --------------------------------------------------------
        cfg.optim.lr = config.training.learning_rate
        cfg.optim.lr_scheduler_type = (
            "cosine" if config.training.lr_scheduler == "cosine" else "constant"
        )
        cfg.optim.lr_warmup_steps_ratio = config.training.warmup_ratio
        # Without this, verl's own FSDPOptimizerConfig default
        # (weight_decay=0.01) silently applies instead of our schema's
        # default (0.0) or whatever the user actually set — training.
        # weight_decay is a plain float (not Optional), so there's no way
        # to distinguish "unset" from "explicitly 0.0"; always forwarding
        # it is the only behavior that respects the schema's own default.
        cfg.optim.weight_decay = config.training.weight_decay
        if config.training.max_grad_norm is not None:
            # None means "no override" (verl's own clip_grad default,
            # 1.0, applies) — same convention as model.attn_implementation
            # elsewhere in this file. An explicit value must still reach
            # verl, though, or it's silently dropped the same way
            # weight_decay was.
            cfg.optim.clip_grad = config.training.max_grad_norm

        # --- data -----------------------------------------------------------
        cfg.data.train_files = train_files
        cfg.data.val_files = val_files
        cfg.data.train_batch_size = sft_cfg.train_batch_size
        cfg.data.micro_batch_size_per_gpu = sft_cfg.micro_batch_size_per_gpu
        cfg.data.max_token_len_per_gpu = sft_cfg.max_token_len_per_gpu
        cfg.data.use_dynamic_bsz = sft_cfg.use_dynamic_bsz
        cfg.data.max_length = sft_cfg.max_length
        cfg.data.truncation = sft_cfg.truncation
        if sft_cfg.dataloader_num_workers is not None:
            # None leaves verl's own hydra default (8) in place — see
            # VerlSftConfig.dataloader_num_workers's docstring for the real
            # SageMaker job (a Gemma 4 26B-A4B run with data.eval_path set)
            # this was reproduced against: every rank's val_dataloader
            # construction crashed with `OSError: [Errno 12] Cannot
            # allocate memory` at os.fork() once the model/optimizer had
            # already grown the parent process's memory footprint.
            cfg.data.num_workers = sft_cfg.dataloader_num_workers
        if sft_cfg.data_format == "flat":
            # prompt_key/response_key/system_prompt are NOT real keys in
            # verl's data/sft_trainer_engine.yaml (confirmed against the
            # installed package — cfg.data.keys() has no such entries) —
            # they're our own convention consumed entirely by
            # FlatPromptResponseSFTDataset via config.get(...). Plain
            # attribute assignment fails in struct mode for unknown keys
            # (same class of bug as model.target_parameters above);
            # force_add=True is required.
            OmegaConf.update(cfg, "data.prompt_key", sft_cfg.prompt_key, force_add=True)
            OmegaConf.update(
                cfg, "data.response_key", sft_cfg.response_key, force_add=True
            )
            OmegaConf.update(
                cfg, "data.system_prompt", sft_cfg.system_prompt, force_add=True
            )
        # Force thinking mode via the STATIC apply_chat_template_kwargs, not
        # data.enable_thinking_default. Verified against the real Gemma 4
        # tokenizer: MultiTurnSFTDataset measures
        # `self.system_prompt`'s length once via
        # extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs)
        # and strips exactly that many leading tokens from every non-first
        # turn's rendering. If enable_thinking is instead applied dynamically
        # per-row (via enable_thinking_default), that baseline measurement is
        # taken WITHOUT thinking while the real per-turn render happens WITH
        # thinking — Gemma 4's thinking preamble is 7 tokens longer, so every
        # assistant turn ends up with leftover `<|turn>system\n<|think|>\n
        # <turn|>\n<|turn>model\n` tokens glued onto its input_ids, which get
        # loss_mask=1 and silently corrupt the loss on every example. Baking
        # the thinking kwarg into apply_chat_template_kwargs (rather than
        # data.enable_thinking_default) keeps the baseline measurement and
        # the per-turn render consistent — this holds for ANY model's
        # thinking-injection mechanism (system-turn like Gemma 4,
        # assistant-prefix like Qwen3, or otherwise), since it doesn't
        # depend on how the template implements it, only that the kwarg
        # name matches. The kwarg name itself IS model-specific — resolved
        # from model.profile (default "enable_thinking" if no profile sets
        # it, since that's the convention both Gemma 4 and Qwen3 use).
        thinking_kwarg = config.model.thinking_kwarg or "enable_thinking"
        cfg.data.apply_chat_template_kwargs = {
            thinking_kwarg: config.model.enable_thinking
        }
        # "flat" and "packed" each need a custom class; "messages" rows
        # already have a full messages list once cached to parquet, so
        # verl's own default (data.custom_cls left unset -> its native
        # MultiTurnSFTDataset) reads it directly — see verl_sft_dataset.py's
        # module docstring.
        if sft_cfg.data_format in ("flat", "packed"):
            # A bare dotted string here raises `FileNotFoundError: Custom
            # module file not found` — verl's load_module()
            # (verl/utils/import_utils.py) treats an unprefixed
            # custom_cls.path as a literal filesystem path, not a Python
            # import string. The "pkg://" prefix is required to get
            # importlib.import_module() dotted-path resolution instead.
            cfg.data.custom_cls.path = "pkg://foresightrlcore.methods.verl_sft_dataset"
            cfg.data.custom_cls.name = (
                "FlatPromptResponseSFTDataset"
                if sft_cfg.data_format == "flat"
                else "PackedSFTDataset"
            )

        # --- trainer -----------------------------------------------------
        cfg.trainer.default_local_dir = str(output_dir)
        cfg.trainer.project_name = config.job.name
        cfg.trainer.experiment_name = config.job.name
        cfg.trainer.total_epochs = config.training.epochs
        # training.data_points' auto-derivation (config/validation.py
        # _resolve_data_points) computes max_steps from TRL-shaped fields
        # (per_device_batch_size * gradient_accumulation_steps) this handler
        # never reads — do not rely on training.data_points for sft_verl;
        # set training.max_steps directly to cap a run (e.g. a smoke test)
        # short of a full epoch.
        if config.training.max_steps is not None and config.training.max_steps > 0:
            cfg.trainer.total_training_steps = config.training.max_steps
        cfg.trainer.save_freq = sft_cfg.save_freq
        cfg.trainer.resume_mode = sft_cfg.resume_mode
        cfg.trainer.logger = [config.training.report_to]
        cfg.trainer.nnodes = config.infra.instance_count
        # A real torchrun launch (this handler's only supported launch
        # mechanism — see the module docstring) always sets this; there is
        # no safe generic default to silently fall back to. In particular,
        # 8 (a plausible-looking former default) would be silently wrong
        # on any 4-GPU instance type (e.g. g5.12xlarge, already
        # special-cased elsewhere in this codebase's SageMaker launcher) —
        # fail loudly instead of telling verl the wrong GPU count.
        try:
            cfg.trainer.n_gpus_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
        except KeyError as e:
            raise RuntimeError(
                "LOCAL_WORLD_SIZE is not set — sft_verl must be launched "
                "via torchrun (directly or via SageMaker's SDK-native "
                "Torchrun distribution), which always sets this. Refusing "
                "to guess a GPU count."
            ) from e

        # --- checkpoint ----------------------------------------------------
        # Without this, each FSDP shard saves the FULL model state (frozen
        # base + LoRA), not just the adapters — verl's own CheckpointConfig
        # docstring quotes ~54 GiB vs ~150 MiB for a 27B model.
        # save_lora_only is auto-detected and merged back in on resume, no
        # other config needed. `save_lora_only` is a real field on verl's
        # CheckpointConfig dataclass (verl/trainer/config/config.py), but
        # the composed sft_trainer_engine.yaml's `checkpoint:` block only
        # lists `save_contents`/`load_contents` explicitly, so it stays in
        # struct mode for this key — same class of bug as
        # model.target_parameters/data.prompt_key above; force_add=True is
        # required.
        OmegaConf.update(
            cfg, "checkpoint.save_lora_only", sft_cfg.save_lora_only, force_add=True
        )

        self._run_verl_sft(cfg, lora.dropout)

        logger.info(f"sft_verl training complete. Final checkpoint dir: {final_dir}")

    @staticmethod
    def _run_verl_sft(cfg: Any, lora_dropout: float) -> None:
        """Mirrors verl's own ``run_sft()`` (``verl/trainer/sft_trainer.py``):
        init the process group (already torchrun-launched by SageMaker),
        build the trainer, fit, tear down. Not wrapped in hydra.main() since
        we build ``cfg`` ourselves above rather than parsing sys.argv.
        """
        from verl.trainer.sft_trainer import SFTTrainer
        from verl.utils.distributed import (
            destroy_global_process_group,
            initialize_global_process_group,
        )

        if cfg.model.lora_rank > 0:
            from foresightrlcore.methods.verl_compat import patch_verl_lora_dropout

            patch_verl_lora_dropout(lora_dropout)

        initialize_global_process_group()
        try:
            trainer = SFTTrainer(config=cfg)
            trainer.fit()
        finally:
            destroy_global_process_group()

    @staticmethod
    def _run_merge_lora_only(config: Any, sft_cfg: Any, lora: Any) -> None:
        """Skip training: reconstruct the SFT checkpoint's (adapter-only,

        under the default save_lora_only=True) state dict, merge the LoRA
        adapter into the REAL pretrained base model's weights, save a
        plain HF model. See
        :mod:`foresightrlcore.methods.verl_checkpoint_merge`'s module
        docstring for why a full merge (not verl's own adapter-export tool)
        is the correct path for target_parameters-based MoE LoRA, and why
        the base weights come from config.model.name rather than the
        checkpoint itself.

        Single-process, rank-0-only — no torch.distributed process group is
        started at all (this launches via the same torchrun entrypoint as a
        normal run, but the FSDP-shard reconstruction below reads saved
        ``.pt`` files directly and needs no live distributed state, mirroring
        verl's own ``model_merger`` CLI, which is a plain script for the
        same reason).
        """
        rank = int(os.environ.get("RANK", 0))
        if rank != 0:
            logger.info(f"[rank{rank}] merge_lora_only: not rank 0, nothing to do")
            return

        from foresightrlcore.methods.verl_checkpoint_merge import (
            check_lora_train_meta_matches,
            find_latest_checkpoint_step,
            merge_lora_and_save,
            reconstruct_full_state_dict,
        )

        checkpoint_root = getattr(config, "_local_output_dir", None) or str(
            config.output.checkpoint_s3_path
        )
        checkpoint_dir = find_latest_checkpoint_step(
            checkpoint_root, sft_cfg.merge_checkpoint_step
        )
        # Catches a copy-pasted merge config whose model.lora silently
        # drifted from the SFT run's — an alpha mismatch in particular
        # would otherwise mis-scale the merge with no other error at all
        # (no missing/unexpected state-dict key, no tensor-shape error).
        check_lora_train_meta_matches(checkpoint_dir, lora)
        # Same fallback semantics as the training path (run()'s
        # cfg.model.trust_remote_code resolution above): respect
        # model.profile's resolved value, only falling back to True when
        # genuinely unset. Previously hardcoded True here regardless,
        # silently ignoring an explicit model.trust_remote_code: false.
        trust_remote_code = (
            config.model.trust_remote_code
            if config.model.trust_remote_code is not None
            else True
        )
        logger.info(
            f"merge_lora_only: reconstructing adapter state dict from {checkpoint_dir}"
        )
        state_dict = reconstruct_full_state_dict(
            checkpoint_dir, trust_remote_code=trust_remote_code
        )

        final_dir = getattr(config, "_final_model_dir", None) or str(
            config.output.s3_path
        )
        # The ORIGINAL pretrained model, not the checkpoint — see
        # merge_lora_and_save's docstring. By the time this runs in
        # container, config.model.name has already been swapped to the
        # local pre-staged mount path if it was an s3:// URI (same
        # mechanism as a normal sft_verl run's model.name resolution).
        merge_lora_and_save(
            state_dict,
            str(config.model.name),
            lora,
            final_dir,
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"merge_lora_only complete. Merged model saved to {final_dir}")


def _check_packed_data_matches_config(
    data_path: str,
    expected_enable_thinking: bool,
    expected_thinking_kwarg: str,
    sft_cfg: Any,
) -> None:
    """Cross-check ``model.enable_thinking``/``thinking_kwarg`` and

    ``training.sft_verl.max_length`` against what was actually baked into
    offline-packed data by ``rl-train tokenize-pack``.

    Packed rows are never re-tokenized at train time — the chat template
    (thinking preamble or not) is rendered once, offline, and permanently
    fixed in ``input_ids``. Two ways this can silently drift between the
    offline pack and the training config, both fixed once and permanently
    baked in with no error anywhere at train time:

    - ``--enable-thinking``/``--thinking-kwarg`` were forgotten or typo'd
      (``--enable-thinking`` defaults to False; an unrecognized
      ``--thinking-kwarg`` is silently ignored by ``apply_chat_template``,
      which only renders kwargs its own template references).
    - ``training.sft_verl.max_length`` is smaller than the pack size used
      to produce the data, so every packed row gets truncated
      (``truncation: "right"``/``"left"`` — the default ``"error"`` is
      already safe, since an oversized row raises loudly on its own).

    Reads the ``*_metadata.jsonl`` sidecar ``rl-train tokenize-pack``
    writes alongside the packed Parquet — no network access needed, since
    data.train_path is already mounted locally by the time ``run()``
    executes in-container.

    A missing sidecar, one from before these fields were recorded, or one
    in a shape this function doesn't recognize (e.g. a genuine
    line-delimited JSONL sidecar from a different packer, rather than the
    single JSON-array-per-line shape ``rl-train tokenize-pack`` itself
    writes) is a soft no-op for ``data_format in ("flat", "messages")``
    — those formats never have a sidecar at all, so its absence is the
    normal case, not a sign anything is wrong.

    But for ``data_format == "packed"`` specifically, a missing/unreadable
    sidecar IS an error: that data_format means the data came from
    ``rl-train tokenize-pack``, which always writes this sidecar, so its
    absence (e.g. a parquet-only re-upload that dropped the sidecar file)
    is anomalous, not normal — and it's exactly the scenario that would
    otherwise let this whole check silently no-op instead of catching a
    real thinking/max_length mismatch.
    """
    import glob
    import json

    is_packed = sft_cfg.data_format == "packed"

    pattern = (
        os.path.join(data_path, "*_metadata.jsonl")
        if os.path.isdir(data_path)
        else (f"{data_path}*_metadata.jsonl")
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        if is_packed:
            raise ValueError(
                f"data_format='packed' but no *_metadata.jsonl sidecar "
                f"found under {data_path!r} (pattern {pattern!r}). "
                "rl-train tokenize-pack always writes this sidecar "
                "alongside the packed Parquet — its absence here means "
                "the enable_thinking/thinking_kwarg/max_length "
                "consistency check below can't run at all, silently. "
                "Re-upload including the sidecar, or fix data.train_path."
            )
        return

    try:
        with open(matches[0]) as f:
            entries = json.loads(f.readline())
    except (OSError, json.JSONDecodeError) as e:
        if is_packed:
            raise ValueError(
                f"data_format='packed' but the *_metadata.jsonl sidecar "
                f"at {matches[0]!r} could not be read/parsed: {e}. "
                "Refusing to silently skip the enable_thinking/"
                "thinking_kwarg/max_length consistency check."
            ) from e
        return
    if not isinstance(entries, list):
        if is_packed:
            raise ValueError(
                f"data_format='packed' but the *_metadata.jsonl sidecar "
                f"at {matches[0]!r} is not in the shape rl-train "
                "tokenize-pack writes (a single JSON array per line). "
                "Refusing to silently skip the enable_thinking/"
                "thinking_kwarg/max_length consistency check."
            )
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        packed_enable_thinking = entry.get("enable_thinking")
        packed_thinking_kwarg = entry.get("thinking_kwarg")
        if (
            packed_enable_thinking is not None
            and packed_thinking_kwarg is not None
            and (
                packed_enable_thinking != expected_enable_thinking
                or packed_thinking_kwarg != expected_thinking_kwarg
            )
        ):
            raise ValueError(
                f"data_format='packed' data at {data_path!r} (split="
                f"{entry.get('split')!r}) was packed with "
                f"enable_thinking={packed_enable_thinking}, "
                f"thinking_kwarg={packed_thinking_kwarg!r} (rl-train "
                f"tokenize-pack --enable-thinking/--thinking-kwarg), but "
                f"model.enable_thinking={expected_enable_thinking}, "
                f"thinking_kwarg={expected_thinking_kwarg!r}. Packed data "
                "is never re-tokenized at train time, so this mismatch "
                "would silently train on the wrong chat-template "
                "rendering. Re-pack with matching flags, or fix "
                "model.enable_thinking/model.thinking_kwarg to match."
            )

        pack_size = entry.get("pack_size")
        if (
            pack_size is not None
            and sft_cfg.truncation != "error"
            and sft_cfg.max_length < pack_size
        ):
            raise ValueError(
                f"data_format='packed' data at {data_path!r} (split="
                f"{entry.get('split')!r}) was packed with pack_size="
                f"{pack_size}, but training.sft_verl.max_length="
                f"{sft_cfg.max_length} with truncation="
                f"{sft_cfg.truncation!r}. Every packed row would be "
                "silently truncated (dropping whole documents, or "
                "training on a fragment of a boundary document) with no "
                "error. Set max_length >= pack_size, or set "
                "truncation: error to fail loudly instead."
            )


def _glob_data_files(
    path: str, extension: str = "jsonl", filename_prefix: str | None = None
) -> list[str]:
    """Resolve a train/eval data path (dir or glob) to a sorted list of files.

    ``config.data.train_path``/``eval_path`` are mounted locally by
    ``_apply_sagemaker_overrides`` (SM_CHANNEL_TRAIN/EVAL) as a directory of
    shards, not a single file. ``extension`` is ``"jsonl"`` for
    ``data_format`` "flat"/"messages" (raw, tokenized in-job) or
    ``"parquet"`` for "packed" (already-packed output of
    ``rl-train tokenize-pack`` — no in-job tokenization at all, so no
    ``_cache_jsonl_as_parquet`` step either).

    ``filename_prefix`` disambiguates train vs eval when both point at the
    *same* directory — ``rl-train tokenize-pack`` writes
    ``training_<N>.idx.parquet`` and ``validation_<N>.idx.parquet`` side by
    side in one flat output dir (matching the real NeMo-packed bucket's own
    layout), so a plain ``*.parquet`` glob on a shared ``train_path``/
    ``eval_path`` would silently pull validation rows into training (or
    vice versa) without this.
    """
    pattern = f"{filename_prefix or ''}*.{extension}"
    # filename_prefix is only ever passed for the packed train/eval
    # disambiguation case, where path is always directory-like (an S3
    # prefix or local mount) even when os.path.isdir() happens to say
    # otherwise (e.g. path not yet resolved to a real local directory) —
    # so a prefix always takes the directory-join branch. Otherwise (the
    # flat/messages case), path may itself be a complete glob expression
    # the caller built (e.g. "s3://bucket/shard-*.jsonl"), which must be
    # globbed as-is rather than having pattern spliced in.
    if filename_prefix or os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, pattern)))
    else:
        files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found under {path!r}")
    return files


register_method("sft_verl", VerlSftMethodHandler)
