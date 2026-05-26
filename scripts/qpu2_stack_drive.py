from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config


PROMPT_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

BASELINE_FLOOR_TPS = 12.4
COOLDOWN_SECONDS = 180

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
    "source": "qpu2-stack-drive",
    "timeout_seconds": 720,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "qpu2_stack_start"}), flush=True)
    emit(run_config(case("qpu2_warmup_n32", [], n_predict=32)))
    baseline = establish_baseline()

    cases = [
        ("qpu2_ckpt0_repeat_a", ["--ctx-checkpoints", "0"]),
        ("qpu2_nocb_ckpt0", ["-nocb", "--ctx-checkpoints", "0"]),
        ("qpu2_nocb_ckpt0_cram0", ["-nocb", "--ctx-checkpoints", "0", "-cram", "0"]),
        ("qpu2_baseline_mid_a", []),
        ("qpu2_nocb_ckpt0_sas", ["-nocb", "--ctx-checkpoints", "0", "-sas"]),
        ("qpu2_nocb_ckpt0_cram0_sas", ["-nocb", "--ctx-checkpoints", "0", "-cram", "0", "-sas"]),
        ("qpu2_nocb_ckpt0_ger_sas", ["-nocb", "--ctx-checkpoints", "0", "-ger", "-sas"]),
        ("qpu2_baseline_mid_b", []),
        ("qpu2_ckpt0_cram0", ["--ctx-checkpoints", "0", "-cram", "0"]),
        ("qpu2_ckpt0_sas", ["--ctx-checkpoints", "0", "-sas"]),
        ("qpu2_ckpt0_ger", ["--ctx-checkpoints", "0", "-ger"]),
        ("qpu2_ckpt0_repeat_b", ["--ctx-checkpoints", "0"]),
        ("qpu2_baseline_final", []),
    ]

    best = baseline
    for label, extra_args in cases:
        run = run_config(case(label, extra_args))
        emit(run)
        if is_success(run) and (best is None or float(run["gen_tps"]) > float(best["gen_tps"])):
            best = run
            print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "event": "qpu2_stack_done",
                "local_best": summarize(best) if best else None,
                "global_best": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def establish_baseline() -> dict[str, Any] | None:
    last = None
    for attempt in range(1, 4):
        run = run_config(case(f"qpu2_baseline_gate_{attempt}", []))
        last = run
        emit(run)
        if is_success(run) and float(run["gen_tps"]) >= BASELINE_FLOOR_TPS:
            print(json.dumps({"event": "baseline_gate_pass", "attempt": attempt, "gen_tps": run["gen_tps"]}, sort_keys=True), flush=True)
            return run
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
            time.sleep(COOLDOWN_SECONDS)
    print(json.dumps({"event": "baseline_gate_low_state"}, sort_keys=True), flush=True)
    return last


def case(label: str, extra_args: list[str], **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, extra_args=extra_args, **overrides)


def emit(run: dict[str, Any] | None) -> None:
    print(json.dumps({"event": "run", **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "label": run.get("label"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "metrics": run.get("metrics"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def is_success(run: dict[str, Any] | None) -> bool:
    return bool(run and run.get("exit_code") == 0 and run.get("gen_tps") is not None)


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    main()
