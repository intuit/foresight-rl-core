"""Flag-only `vllm serve` args: store_true options need a bare flag."""

from __future__ import annotations

from foresightrlcore.common.vllm_process import build_serve_command
from foresightrlcore.config.perf_schema import VllmServingConfig


class TestFlagOnlyServeArgs:
    """vllm's store_true options are unreachable if a value is always appended."""

    def _cmd(self, extra_args):
        return build_serve_command(
            "/models/m", VllmServingConfig(extra_args=extra_args)
        )

    def test_an_empty_value_emits_a_bare_flag(self):
        cmd = self._cmd({"enable-lora": ""})
        assert "--enable-lora" in cmd
        assert "" not in cmd

    def test_a_value_is_still_passed(self):
        cmd = self._cmd({"max-lora-rank": "64"})
        assert cmd[cmd.index("--max-lora-rank") + 1] == "64"

    def test_a_lora_module_can_be_served(self):
        # The combination this unblocks: serve a base model with an adapter
        # mounted under its own name.
        cmd = self._cmd(
            {"enable-lora": "", "lora-modules": "eval_lora=/opt/ml/input/data/adapter"}
        )
        assert cmd[cmd.index("--lora-modules") + 1].startswith("eval_lora=")
        assert cmd[cmd.index("--enable-lora") + 1] == "--lora-modules"
