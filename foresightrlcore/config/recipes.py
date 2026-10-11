"""Container and local-GPU recipes — named presets for pip requirements.

**Container recipes** map a model family to the DLC image version and the
exact pip dependencies needed inside the SageMaker training container.  The
launcher writes these to ``requirements.txt`` at packaging time so
``entrypoint.sh`` installs the correct stack without any code changes.

**Local-GPU recipes** define the pip dependencies for bare-metal GPU machines.
They are a separate registry because local environments have different
constraints (no DLC base image, no ``framework_version``, different version
pins).  Use ``rl-train setup-local-gpu`` to generate a requirements file.

Requirements files live in ``foresightrlcore/config/requirements/`` and are
the **single source of truth** for both the SageMaker launcher (loaded here
at import time) and the Docker builds (``intuit/docker/Dockerfile.*`` COPY them
directly).  Edit a ``.txt`` file to update both paths simultaneously.

To add a new recipe:
  1. Add a requirements file under ``foresightrlcore/config/requirements/``.
  2. Add an entry to ``CONTAINER_RECIPES`` or ``LOCAL_GPU_RECIPES`` below.
  3. Add a Docker variant under ``intuit/docker/`` (see
     ``intuit/docs/building-docker-images.md``).
  4. Add a test in ``tests/config/test_config_recipes.py``.
  5. Run ``uv run tox``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foresightrlcore.config.schema import ContainerConfig

# ---------------------------------------------------------------------------
# Requirements directory — single source of truth shared with docker builds.
# ---------------------------------------------------------------------------

_REQ_DIR: Path = Path(__file__).parent / "requirements"


def _load_requirements(filename: str) -> list[str]:
    """Read a requirements .txt file and return non-empty, non-comment lines."""
    path = _REQ_DIR / filename
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# ---------------------------------------------------------------------------
# Container recipes
# ---------------------------------------------------------------------------

_DJL_INFERENCE_IMAGE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/"
    "djl-inference:0.36.0-lmi25.0.0-cu130"
)
"""Pre-baked DJL Large-Model-Inference image used by the perf-test operation.

Ships with Java Corretto 17, vLLM, CUDA 13.0, and Python 3.12. Only Gradle
needs to be installed at container startup (handled by ``perf_entrypoint.sh``).
"""


CONTAINER_RECIPES: dict[str, dict[str, Any]] = {
    "qwen3": {
        "framework_version": "2.8.0",
        "framework": "pytorch",
        "requirements": _load_requirements("qwen3.txt"),
        "requirements_torchrun": _load_requirements("qwen3-torchrun.txt"),
        "perf_test_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_perf_test": _load_requirements("qwen3-perf-test.txt"),
        "optimizer_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_optimizer": _load_requirements("optimizer.txt"),
        "evaluate_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_evaluate": _load_requirements("evaluate.txt"),
    },
    "qwen3_5": {
        "framework_version": "2.9.0",
        "framework": "pytorch",
        "image_uri": "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker",
        "requirements": _load_requirements("qwen3_5.txt"),
        "requirements_torchrun": None,
        "perf_test_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_perf_test": _load_requirements("qwen3_5-perf-test.txt"),
        "optimizer_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_optimizer": _load_requirements("optimizer.txt"),
        "evaluate_image_uri": _DJL_INFERENCE_IMAGE,
        "requirements_evaluate": _load_requirements("evaluate.txt"),
    },
    "verl_sft": {
        # verl SFT (method.type: sft_verl) is torchrun-only, vllm_mode: none —
        # there is no vLLM-serving variant of this recipe, so `requirements`
        # and `requirements_torchrun` intentionally point at the same file.
        # Model-agnostic: verl's FSDP SFT stack is the same regardless of
        # model.profile — model-specific facts live in model.profile
        # (e.g. gemma4_moe, muse_glimmer), not in this recipe's name or
        # pins (renamed from "gemma4_verl", which fused engine+model
        # naming — the same reasoning ruled out ever adding a parallel
        # "muse_glimmer_verl" recipe: its requirements.txt would have been
        # byte-for-byte identical to this one's actual pip pins, since the
        # transformers-version swap Muse Glimmer needs already happens at
        # job runtime regardless of which requirements.txt built the
        # container — see
        # foresightrlcore.methods.sft_verl._ensure_transformers_version_compatible()
        # and foresightrlcore.config.model_profiles.min_transformers_version()).
        # Base DLC tag reused from qwen3_5's, verified against verl's
        # actual torch/CUDA/flash-attn requirements on real hardware — see
        # config/requirements/verl_sft.txt for the pin rationale.
        "framework_version": "2.9.0",
        "framework": "pytorch",
        "image_uri": "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker",
        "requirements": _load_requirements("verl_sft.txt"),
        "requirements_torchrun": _load_requirements("verl_sft.txt"),
    },
}

_FALLBACK_FRAMEWORK_VERSION = "2.8.0"


# ---------------------------------------------------------------------------
# GPU family resolution + per-family flash-attn wheel selection
# ---------------------------------------------------------------------------
#
# Verified against real sources (not assumed):
#   - AWS: p4d/p4de = NVIDIA A100 (Ampere, sm_80); p5/p5e/p5en = NVIDIA
#     H100/H200 (Hopper, sm_90); p6[-b200] = NVIDIA B200 (Blackwell, sm_100).
#   - flash-attention's own CUDA support line lists Ampere/Ada/Hopper under
#     ONE wheel build (github.com/Dao-AILab/flash-attention README) — no
#     separate per-architecture wheel for FA2 across those three.
#   - Blackwell (sm_100): FA2 runs on it via its generic kernel path (not
#     the fastest possible, but functionally correct) PROVIDED the wheel
#     was built with CUDA>=12.8 (setup.py only emits the sm_100 gencode
#     flag from CUDA 12.8 onward — confirmed via the flash-attention repo's
#     own build config). Our pinned wheel is a cu13 build, well past that
#     bar, so the SAME wheel already covers all three families below.
#
# This table exists so that IS NOT the same as "no GPU-awareness needed at
# all": (a) an unrecognized instance_type fails loudly instead of silently
# assuming Ampere, (b) if/when FA3 (Hopper-only, ~30% faster than FA2 on
# Hopper per Dao-AILab/flash-attention#1853-adjacent discussion) or FA4
# (Hopper+Blackwell, CuTeDSL) get verified against real p5en/p6 hardware,
# swapping a family's wheel URL here is a one-line config change, not a
# code change.
GPU_FAMILIES: dict[str, dict[str, str]] = {
    "p4d": {"name": "ampere", "gpu": "A100", "sm": "sm_80"},
    "p4de": {"name": "ampere", "gpu": "A100", "sm": "sm_80"},
    "p5": {"name": "hopper", "gpu": "H100", "sm": "sm_90"},
    "p5e": {"name": "hopper", "gpu": "H100", "sm": "sm_90"},
    "p5en": {"name": "hopper", "gpu": "H200", "sm": "sm_90"},
    "p6": {"name": "blackwell", "gpu": "B200", "sm": "sm_100"},
    # Real SageMaker/EC2 family tokens are hyphenated ("ml.p6-b200.48xlarge",
    # "ml.p6-b300.48xlarge" — confirmed in docs/checking-spot-capacity-scores.md),
    # not bare "p6". resolve_gpu_family()'s parsing already extracts the
    # hyphenated token correctly (it splits only on "."); the bare "p6" key
    # above is kept in case it's ever used, but the real instance types need
    # their own entries or they'd raise ValueError despite being supported.
    "p6-b200": {"name": "blackwell", "gpu": "B200", "sm": "sm_100"},
    "p6-b300": {"name": "blackwell", "gpu": "B300", "sm": "sm_100"},
    "p6e": {"name": "blackwell", "gpu": "GB200", "sm": "sm_100"},
    # Same hyphenation gap as p6-b200/p6-b300 above: the real instance type
    # is "ml.p6e-gb200.<size>", not bare "p6e".
    "p6e-gb200": {"name": "blackwell", "gpu": "GB200", "sm": "sm_100"},
}

_FLASH_ATTN_CU13_TORCH29_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

FLASH_ATTN_WHEELS: dict[str, str] = {
    "ampere": _FLASH_ATTN_CU13_TORCH29_WHEEL,
    "hopper": _FLASH_ATTN_CU13_TORCH29_WHEEL,
    "blackwell": _FLASH_ATTN_CU13_TORCH29_WHEEL,
}


def resolve_gpu_family(instance_type: str) -> dict[str, str]:
    """Map a SageMaker ``infra.instance_type`` string to its GPU family.

    SageMaker instance types have the shape ``ml.<family><gen>.<size>``
    (e.g. ``ml.p4de.24xlarge``, ``ml.p5en.48xlarge``). Matches on the family
    token exactly (not a prefix substring) so ``p5`` never accidentally
    swallows ``p5en``/``p5e`` or vice versa.

    Raises ``ValueError`` for any instance type this table hasn't been
    verified against, rather than silently guessing a GPU family — a wrong
    guess here would pick a flash-attn build for the wrong architecture on
    a real, expensive multi-node job.
    """
    parts = instance_type.split(".")
    family_token = parts[1] if len(parts) >= 2 and parts[0] == "ml" else parts[0]
    if family_token not in GPU_FAMILIES:
        available = sorted(GPU_FAMILIES.keys())
        raise ValueError(
            f"Unrecognized GPU family for instance_type={instance_type!r} "
            f"(parsed family token {family_token!r}). Known families: "
            f"{available}. Add an entry to GPU_FAMILIES (and, if needed, "
            "FLASH_ATTN_WHEELS) in foresightrlcore/config/recipes.py after "
            "verifying the right flash-attn build for this hardware."
        )
    return GPU_FAMILIES[family_token]


def resolve_flash_attn_wheel(instance_type: str) -> str:
    """Resolve the flash-attn wheel URL to use for a given instance type."""
    family = resolve_gpu_family(instance_type)["name"]
    return FLASH_ATTN_WHEELS[family]


def resolve_container_recipe(container: "ContainerConfig") -> None:
    """Apply recipe defaults to ContainerConfig in-place.

    Called from ``ContainerConfig``'s ``model_validator``.  Only sets fields
    that weren't explicitly provided by the user (i.e. are still ``None``).
    """
    if container.recipe is None:
        if container.framework_version is None:
            container.framework_version = _FALLBACK_FRAMEWORK_VERSION
        return

    recipe_name = container.recipe
    if recipe_name not in CONTAINER_RECIPES:
        available = sorted(CONTAINER_RECIPES.keys())
        raise ValueError(
            f"Unknown container recipe: '{recipe_name}'. Available: {available}"
        )

    recipe = CONTAINER_RECIPES[recipe_name]

    if container.framework_version is None:
        container.framework_version = recipe["framework_version"]

    if container.framework == "pytorch" and "framework" in recipe:
        container.framework = recipe["framework"]

    if container.image_uri is None and "image_uri" in recipe:
        region = os.environ.get("AWS_DEFAULT_REGION", "")
        container.image_uri = recipe["image_uri"].format(region=region)

    if container.requirements is None:
        container.resolved_requirements = recipe["requirements"]
        container.resolved_requirements_torchrun = recipe.get("requirements_torchrun")
        container.resolved_requirements_perf_test = recipe.get("requirements_perf_test")
        container.resolved_requirements_optimizer = recipe.get("requirements_optimizer")
        container.resolved_requirements_evaluate = recipe.get("requirements_evaluate")
    else:
        container.resolved_requirements = None
        container.resolved_requirements_torchrun = None
        container.resolved_requirements_perf_test = None
        container.resolved_requirements_optimizer = None
        container.resolved_requirements_evaluate = None

    if "perf_test_image_uri" in recipe:
        region = os.environ.get("AWS_DEFAULT_REGION", "")
        container.perf_test_image_uri = recipe["perf_test_image_uri"].format(
            region=region
        )

    if "optimizer_image_uri" in recipe and container.optimizer_image_uri is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "")
        container.optimizer_image_uri = recipe["optimizer_image_uri"].format(
            region=region
        )

    if "evaluate_image_uri" in recipe and container.evaluate_image_uri is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "")
        container.evaluate_image_uri = recipe["evaluate_image_uri"].format(
            region=region
        )


# ---------------------------------------------------------------------------
# Local-GPU recipes
# ---------------------------------------------------------------------------

LOCAL_GPU_RECIPES: dict[str, dict[str, Any]] = {
    "qwen3": {
        "requirements": [
            "torch >=2.6.0",
            "trl[vllm] >=0.28.0",
            "peft >=0.18.0",
            "accelerate >=1.12.0",
            "deepspeed >=0.18.0",
            "datasets >=4.5.0",
            "transformers >=4.57.0",
            "bitsandbytes >=0.49.0",
            "safetensors >=0.7.0",
            "tensorboard",
            "langchain_openai >=0.3.0",
            "boto3 >=1.34",
            "huggingface_hub[hf_transfer]",
        ],
    },
    "qwen3_5": {
        "requirements": [
            "torch >=2.11.0",
            "trl >=1.3.0",
            "vllm >=0.20.0",
            "peft >=0.19.1",
            "accelerate >=1.13.0",
            "deepspeed >=0.18.9",
            "datasets >=4.8.5",
            "transformers >=5.6.0",
            "bitsandbytes >=0.49.2",
            "safetensors >=0.7.0",
            "tensorboard",
            "langchain_openai >=1.2.0",
            "boto3 >=1.34",
            "huggingface_hub[hf_transfer] >=1.12.0",
            "liger-kernel >=0.5.0",
        ],
    },
}


def generate_local_gpu_requirements(
    recipe_name: str,
    output_path: str = "local-requirements.txt",
) -> Path:
    """Write a pip requirements file from a local-GPU recipe.

    Returns the path to the written file.

    Raises ``ValueError`` if *recipe_name* is not in ``LOCAL_GPU_RECIPES``.
    """
    if recipe_name not in LOCAL_GPU_RECIPES:
        available = sorted(LOCAL_GPU_RECIPES.keys())
        raise ValueError(
            f"Unknown local-GPU recipe: '{recipe_name}'. Available: {available}"
        )

    reqs = LOCAL_GPU_RECIPES[recipe_name]["requirements"]
    out = Path(output_path)
    out.write_text(
        f"# Auto-generated by: rl-train setup-local-gpu --recipe {recipe_name}\n"
        + "\n".join(reqs)
        + "\n"
    )
    return out
