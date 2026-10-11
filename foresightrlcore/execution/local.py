"""Local GPU execution backend.

Automates the manual workflow from grpo_advice.py:
  1. Split GPUs: vLLM GPUs vs training GPUs
  2. Optionally start vLLM (via ``vllm_launcher.py``) on the vLLM GPUs
  3. Launch `accelerate launch` on the training GPUs
  4. Wait for training to finish, then stop vLLM

All GPU assignments and process management are handled automatically
based on the config YAML (execution.local.* and infra.*).

vLLM is always launched via ``foresightrlcore.runtime.scripts.vllm_launcher``
(invoked as ``python -m ...``) rather than ``trl vllm-serve`` directly. The
launcher applies spawn-safety env vars and (optionally) patches TRL's
``vllm_serve.py`` for ``reasoning_config`` / ``compilation_config`` /
``disable_custom_all_reduce`` — see that module's docstring for the
contract. This keeps local, SageMaker, and HyperPod EKS backends on a
single, consistent vLLM startup path.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

from foresightrlcore.execution import register_backend

logger = logging.getLogger(__name__)


class LocalBackend:
    """Execution backend for local GPU training."""

    backend_type = "local"

    _REQUIRED_PACKAGES = ["trl", "accelerate", "peft", "datasets", "transformers"]

    def validate(self, config: Any) -> list[str]:
        errors: list[str] = []
        local_cfg = config.execution.local or {}

        if not local_cfg.get("gpus"):
            errors.append(
                "Local backend requires execution.local.gpus "
                "(e.g. '0,1,2,3,4,5,6,7')"
            )
            return errors

        gpu_ids = [int(g.strip()) for g in str(local_cfg["gpus"]).split(",")]

        vllm_mode = config.infra.vllm_mode.value
        if vllm_mode == "server":
            vllm_count = config.infra.vllm_gpu_count
            training_count = config.infra.training_gpu_count
            if len(gpu_ids) < vllm_count + training_count:
                errors.append(
                    f"Not enough GPUs: need {vllm_count} (vLLM) + "
                    f"{training_count} (training) = {vllm_count + training_count}, "
                    f"but only {len(gpu_ids)} specified."
                )

        missing = self._check_gpu_packages()
        if missing:
            errors.append(
                f"Missing GPU packages: {', '.join(missing)}. "
                "Run: rl-train setup-local-gpu --recipe <recipe> --install"
            )

        return errors

    @classmethod
    def _check_gpu_packages(cls) -> list[str]:
        """Return names of required packages that cannot be imported."""
        missing: list[str] = []
        for pkg in cls._REQUIRED_PACKAGES:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        return missing

    def launch(self, config: Any, wait: bool = False) -> str:
        """Launch local training.

        Follows the grpo_advice.py pattern:
          CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve ...
          CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch ...
        """
        local_cfg = config.execution.local or {}
        gpu_ids = [int(g.strip()) for g in str(local_cfg["gpus"]).split(",")]
        vllm_mode = config.infra.vllm_mode.value
        method_type = config.method.type

        output_dir = local_cfg.get("output_dir")
        if not output_dir:
            s3_path = config.output.s3_path
            if s3_path.startswith("s3://"):
                output_dir = os.path.join("output", config.job.name)
            else:
                output_dir = s3_path
        os.makedirs(output_dir, exist_ok=True)

        config_path = local_cfg.get("config_path")
        if not config_path:
            config_path = os.path.join(output_dir, "args.yaml")
            config.to_yaml(config_path)
            logger.info(f"Wrote config to {config_path}")

        vllm_proc = None
        try:
            if vllm_mode == "server":
                vllm_proc = self._start_vllm_server(config, gpu_ids)
                train_gpus = gpu_ids[config.infra.vllm_gpu_count :]
            elif vllm_mode == "none":
                train_gpus = gpu_ids
            else:
                train_gpus = gpu_ids

            job_id = self._run_training(
                config=config,
                train_gpus=train_gpus,
                config_path=config_path,
                output_dir=output_dir,
                method_type=method_type,
                vllm_mode=vllm_mode,
            )
            return job_id

        finally:
            if vllm_proc is not None:
                self._stop_vllm_server(vllm_proc)

    def get_status(self, job_id: str) -> dict[str, Any]:
        return {"job_id": job_id, "status": "LOCAL_COMPLETED"}

    def _start_vllm_server(self, config: Any, gpu_ids: list[int]) -> subprocess.Popen:
        """Start vLLM server on the designated GPUs."""
        vllm_gpus = gpu_ids[: config.infra.vllm_gpu_count]
        vllm_cfg = config.infra.vllm

        local_cfg = config.execution.local or {}
        output_dir = local_cfg.get("output_dir") or os.path.join(
            "output", config.job.name
        )
        os.makedirs(output_dir, exist_ok=True)

        cuda_devices = ",".join(str(g) for g in vllm_gpus)
        # Invoke via `python -m` so the module is found via package import
        # rather than a script path — keeps the local backend agnostic to
        # where the package is installed.
        cmd = [
            sys.executable,
            "-m",
            "foresightrlcore.runtime.scripts.vllm_launcher",
            "--model",
            config.model.name,
            "--host",
            "0.0.0.0",
            "--port",
            str(vllm_cfg.port),
            "--gpu_memory_utilization",
            str(vllm_cfg.gpu_memory_utilization),
            "--tensor_parallel_size",
            str(vllm_cfg.tensor_parallel_size),
        ]

        if vllm_cfg.enforce_eager:
            cmd.append("--enforce_eager")

        if vllm_cfg.enable_prefix_caching:
            cmd.extend(["--enable-prefix-caching", "yes"])

        if vllm_cfg.speculative is not None:
            spec_dict = vllm_cfg.speculative.model_dump(exclude_none=True)
            draft = spec_dict.get("model")
            if isinstance(draft, str) and draft.startswith("s3://"):
                stage_dir = os.path.join(output_dir, "speculative_draft_model")
                self._stage_s3_draft_model(draft, stage_dir)
                spec_dict["model"] = stage_dir
            spec_json = json.dumps(spec_dict, separators=(",", ":"))
            cmd.extend(["--speculative-config", spec_json])

        # Forward TRL-patch opt-ins via the launcher's documented env-var
        # contract. The launcher reads these once at startup and calls
        # patch_trl_vllm.patch(...) before importing TRL — see vllm_launcher.py.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        env.update(self._build_vllm_patch_env(vllm_cfg))
        logger.info(f"Starting vLLM server on GPUs [{cuda_devices}]: {' '.join(cmd)}")
        vllm_log_path = os.path.join(output_dir, "vllm_server.log")
        vllm_log_file = open(vllm_log_path, "w")
        logger.info(f"vLLM server logs: {vllm_log_path}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=vllm_log_file,
            stderr=subprocess.STDOUT,
        )

        startup_wait = 30
        logger.info(f"Waiting {startup_wait}s for vLLM server to initialize...")
        time.sleep(startup_wait)

        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited prematurely with code {proc.returncode}"
            )

        logger.info("vLLM server appears to be running.")
        return proc

    @staticmethod
    def _build_vllm_patch_env(vllm_cfg: Any) -> dict[str, str]:
        """Translate ``infra.vllm.*`` opt-ins into the launcher's env contract.

        Returns a dict suitable for ``env.update(...)``. Only keys with active
        opt-ins are present — absent keys leave the launcher in its no-op
        fast path. See ``vllm_launcher.py`` for the env-variable contract.
        """
        out: dict[str, str] = {}
        # reasoning_parser must be set alongside reasoning_config for the
        # patch to do anything useful (the parser identifier itself is read
        # elsewhere; the launcher only needs the start/end delimiter dict).
        reasoning_config = getattr(vllm_cfg, "reasoning_config", None)
        if reasoning_config:
            out["VLLM_REASONING_CONFIG_JSON"] = json.dumps(
                reasoning_config, separators=(",", ":")
            )
        compilation_config = getattr(vllm_cfg, "compilation_config", None)
        if compilation_config:
            out["VLLM_COMPILATION_CONFIG_JSON"] = json.dumps(
                compilation_config, separators=(",", ":")
            )
        if getattr(vllm_cfg, "disable_custom_all_reduce", False):
            out["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
        return out

    def _stage_s3_draft_model(self, s3_uri: str, local_dir: str) -> None:
        """Sync an s3:// speculative draft model to a local directory.

        vLLM's --speculative-config `model:` field only accepts a local path
        or HF Hub ID, so S3 checkpoints must be staged first.
        """
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Staging speculative draft from {s3_uri} -> {local_dir}")
        result = subprocess.run(
            ["aws", "s3", "sync", s3_uri, local_dir, "--only-show-errors"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to stage speculative draft from {s3_uri}: {result.stderr}"
            )
        logger.info(f"Speculative draft staged to {local_dir}")

    def _run_training(
        self,
        config: Any,
        train_gpus: list[int],
        config_path: str,
        output_dir: str,
        method_type: str,
        vllm_mode: str,
    ) -> str:
        """Launch training via accelerate or torchrun."""
        cuda_devices = ",".join(str(g) for g in train_gpus)
        num_processes = len(train_gpus)

        if vllm_mode == "none":
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_processes}",
                "-m",
                "foresightrlcore.runtime.scripts.train",
                "--config",
                config_path,
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
                "--num_processes",
                str(num_processes),
                "-m",
                "foresightrlcore.runtime.scripts.train",
                "--config",
                config_path,
            ]

        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        model_dir = os.path.join(output_dir, "model")
        train_data_dir = os.path.join(output_dir, "data", "train")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        train_path = config.data.train_path
        if train_path:
            self._ensure_local_data(train_path, train_data_dir)

        env = {
            **os.environ,
            **config.environment,
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "SM_CHANNEL_TRAIN": os.path.abspath(train_data_dir),
            "SM_CHECKPOINT_DIR": os.path.abspath(checkpoint_dir),
            "SM_MODEL_DIR": os.path.abspath(model_dir),
        }

        logger.info(
            f"Launching {method_type} training on GPUs [{cuda_devices}]: "
            f"{' '.join(cmd)}"
        )

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Training process exited with code {result.returncode}")

        job_id = f"local-{config.job.name}-{int(time.time())}"
        logger.info(f"Local training completed: {job_id}")
        return job_id

    def _ensure_local_data(self, data_path: str, local_dir: str) -> None:
        """Download training data to *local_dir* if not already present.

        Supports:
          - S3 URIs  (``s3://bucket/key``)  — downloaded via ``aws s3 cp``
          - Local paths — symlinked or copied into *local_dir*
        """
        existing = glob.glob(os.path.join(local_dir, "*.json*"))
        if existing:
            logger.info(
                f"Training data already present in {local_dir} "
                f"({len(existing)} file(s)), skipping download."
            )
            return

        if data_path.startswith("s3://"):
            logger.info(f"Downloading training data from {data_path} → {local_dir}/")
            result = subprocess.run(
                ["aws", "s3", "cp", data_path, local_dir + "/"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to download training data from S3: {result.stderr.strip()}"
                )
            downloaded = glob.glob(os.path.join(local_dir, "*.json*"))
            logger.info(f"Downloaded {len(downloaded)} file(s) to {local_dir}/")
        else:
            src = os.path.abspath(data_path)
            if os.path.isfile(src):
                dst = os.path.join(local_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                    logger.info(f"Symlinked local data: {src} → {dst}")
            elif os.path.isdir(src):
                for f in os.listdir(src):
                    if f.endswith((".json", ".jsonl")):
                        dst = os.path.join(local_dir, f)
                        if not os.path.exists(dst):
                            os.symlink(os.path.join(src, f), dst)
                logger.info(f"Symlinked local data directory: {src} → {local_dir}/")
            else:
                raise FileNotFoundError(f"Training data path not found: {src}")

    def _stop_vllm_server(self, proc: subprocess.Popen) -> None:
        """Gracefully stop the vLLM server process."""
        logger.info("Stopping vLLM server...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        if proc.stdout and hasattr(proc.stdout, "close"):
            try:
                proc.stdout.close()
            except OSError:
                pass
        logger.info("vLLM server stopped.")


register_backend("local", LocalBackend)
