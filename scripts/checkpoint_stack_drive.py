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
    "source": "checkpoint-stack-drive",
    "timeout_seconds": 720,
}
BASELINE_FLOOR_TPS = 11.5
COOLDOWN_SECONDS = 60


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "checkpoint_stack_start"}), flush=True)
    emit(run_config(case("ckstack_warmup_n32", [], n_predict=32)))
    establish_baseline()

    cases = [
        ("ckstack_ckpt0_a", ["--ctx-checkpoints", "0"]),
        ("ckstack_ckpt0_b", ["--ctx-checkpoints", "0"]),
        ("ckstack_ckpt1", ["--ctx-checkpoints", "1"]),
        ("ckstack_ckpt2", ["--ctx-checkpoints", "2"]),
        ("ckstack_baseline_b", []),
        ("ckstack_ckpt4", ["--ctx-checkpoints", "4"]),
        ("ckstack_ckpt8", ["--ctx-checkpoints", "8"]),
        ("ckstack_ckpt16", ["--ctx-checkpoints", "16"]),
        ("ckstack_ckpt64", ["--ctx-checkpoints", "64"]),
        ("ckstack_baseline_c", []),
        ("ckstack_interval1024", ["--ctx-checkpoints-interval", "1024"]),
        ("ckstack_interval2048", ["--ctx-checkpoints-interval", "2048"]),
        ("ckstack_tolerance0", ["--ctx-checkpoints-tolerance", "0"]),
        ("ckstack_tolerance10", ["--ctx-checkpoints-tolerance", "10"]),
        ("ckstack_baseline_d", []),
        ("ckstack_sas", ["-sas"]),
        ("ckstack_khad", ["-khad"]),
        ("ckstack_vhad", ["-vhad"]),
        ("ckstack_khad_vhad", ["-khad", "-vhad"]),
        ("ckstack_ckpt0_sas", ["--ctx-checkpoints", "0", "-sas"]),
        ("ckstack_ckpt0_khad", ["--ctx-checkpoints", "0", "-khad"]),
        ("ckstack_ckpt0_vhad", ["--ctx-checkpoints", "0", "-vhad"]),
        ("ckstack_baseline_final", []),
    ]

    best: dict[str, Any] | None = None
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
                "event": "checkpoint_stack_done",
                "local_best": summarize(best) if best else None,
                "global_best": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def establish_baseline() -> None:
    for attempt in range(1, 4):
        run = run_config(case(f"ckstack_baseline_gate_{attempt}", []))
        emit(run)
        if run.get("exit_code") == 0 and run.get("gen_tps") is not None:
            if float(run["gen_tps"]) >= BASELINE_FLOOR_TPS:
                print(
                    json.dumps(
                        {"event": "baseline_gate_pass", "attempt": attempt, "gen_tps": run["gen_tps"]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return
        if attempt < 3:
            print(
                json.dumps(
                    {
                        "event": "baseline_gate_cooldown",
                        "attempt": attempt,
                        "cooldown_seconds": COOLDOWN_SECONDS,
                        "gen_tps": run.get("gen_tps"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time_sleep(COOLDOWN_SECONDS)
    print(json.dumps({"event": "baseline_gate_low_state"}, sort_keys=True), flush=True)


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


def time_sleep(seconds: int) -> None:
    import time

    time.sleep(seconds)


if __name__ == "__main__":
    main()
