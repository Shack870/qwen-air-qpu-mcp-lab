from __future__ import annotations

import json
import subprocess
from typing import Any

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config


PROMPT_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": PROMPT_CONTINUE,
    "ctx_size": 16384,
    "batch_size": 2304,
    "ubatch_size": 96,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "impl-flags-drive",
    "timeout_seconds": 420,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "impl_flags_drive_start"}), flush=True)
    emit(run_config(case("impl_warmup_n32", [], n_predict=32)))

    cases = [
        ("impl_baseline_repeat1", []),
        ("impl_baseline_repeat2", []),
        ("impl_nocb", ["-nocb"]),
        ("impl_ckpt0", ["--ctx-checkpoints", "0"]),
        ("impl_ckpt_interval0", ["--ctx-checkpoints-interval", "0"]),
        ("impl_amb256", ["-amb", "256"]),
        ("impl_amb512", ["-amb", "512"]),
        ("impl_amb1024", ["-amb", "1024"]),
        ("impl_nocb_ckpt0", ["-nocb", "--ctx-checkpoints", "0"]),
        ("impl_nocb_amb512", ["-nocb", "-amb", "512"]),
        ("impl_ckpt0_amb512", ["--ctx-checkpoints", "0", "-amb", "512"]),
        ("impl_qpu_top_nocb_ckpt0_amb512", ["-nocb", "--ctx-checkpoints", "0", "-amb", "512"]),
        ("impl_cram0", ["-cram", "0"]),
        ("impl_cram_unlimited", ["-cram", "-1"]),
        ("impl_mqkv", ["-mqkv"]),
        ("impl_muge", ["-muge"]),
        ("impl_mqkv_muge", ["-mqkv", "-muge"]),
        ("impl_ger", ["-ger"]),
        ("impl_mqkv_muge_ger", ["-mqkv", "-muge", "-ger"]),
        ("impl_sas", ["-sas"]),
        ("impl_mla0", ["-mla", "0"]),
        ("impl_mla1", ["-mla", "1"]),
        ("impl_khad", ["-khad"]),
        ("impl_vhad", ["-vhad"]),
        ("impl_khad_vhad", ["-khad", "-vhad"]),
        ("impl_nocb_ckpt0_amb512_mqkv_muge", ["-nocb", "--ctx-checkpoints", "0", "-amb", "512", "-mqkv", "-muge"]),
    ]

    best = None
    for label, extra_args in cases:
        run = run_config(case(label, extra_args))
        emit(run)
        if run.get("exit_code") == 0 and run.get("gen_tps") is not None:
            if best is None or float(run["gen_tps"]) > float(best["gen_tps"]):
                best = run
                print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "event": "impl_flags_drive_done",
                "local_best": summarize(best) if best else None,
                "global_best": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def case(label: str, extra_args: list[str], **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, extra_args=extra_args, **overrides)


def emit(run: dict[str, Any]) -> None:
    print(json.dumps({"event": "run", **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": run.get("label"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "metrics": run.get("metrics"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    main()
