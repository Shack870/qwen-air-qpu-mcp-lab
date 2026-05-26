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

BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": PROMPT_CONTINUE,
    "ctx_size": 16384,
    "batch_size": 2304,
    "ubatch_size": 104,
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
    "source": "moonshot-micro-ridge-drive",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    best_before = db.best_runs(limit=1)[0]
    print(json.dumps({"event": "micro_ridge_start", "global_best_before": summarize(best_before)}, sort_keys=True), flush=True)
    local_best = None
    for cfg in cases():
        run = run_config(cfg)
        emit(run)
        if is_success(run) and (local_best is None or float(run["gen_tps"]) > float(local_best["gen_tps"])):
            local_best = run
            print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)
        if is_success(run) and float(run["gen_tps"]) > float(best_before["gen_tps"]):
            print(json.dumps({"event": "new_global_candidate", **summarize(run)}, sort_keys=True), flush=True)
        maybe_cool(run)
    print(
        json.dumps(
            {
                "event": "micro_ridge_done",
                "local_best": summarize(local_best),
                "global_best_after": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resume_codex_gpu_helper()


def cases() -> list[dict[str, Any]]:
    points: list[tuple[str, int, int]] = [
        ("b2272_ub104", 2272, 104),
        ("b2304_ub100", 2304, 100),
        ("b2304_ub104_repeat", 2304, 104),
        ("b2304_ub108", 2304, 108),
        ("b2320_ub104", 2320, 104),
        ("b2336_ub104", 2336, 104),
        ("b2352_ub104", 2352, 104),
        ("b2368_ub100", 2368, 100),
        ("b2368_ub104", 2368, 104),
        ("b2384_ub104", 2384, 104),
        ("b2400_ub104", 2400, 104),
        ("b2304_ub104_final", 2304, 104),
    ]
    return [dict(BASE, label=f"micro_{label}", batch_size=batch, ubatch_size=ubatch) for label, batch, ubatch in points]


def emit(run: dict[str, Any] | None) -> None:
    print(json.dumps({"event": "run", **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "label": run.get("label"),
        "source": run.get("source"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "metrics": run.get("metrics"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def is_success(run: dict[str, Any] | None) -> bool:
    return bool(run and run.get("exit_code") == 0 and run.get("gen_tps") is not None)


def maybe_cool(run: dict[str, Any] | None) -> None:
    if not run:
        return
    metrics = run.get("metrics") or {}
    invol = int(metrics.get("involuntary_context_switches") or 0)
    gen_tps = float(run.get("gen_tps") or 0.0)
    if invol > 100000 or (gen_tps and gen_tps < 12.0):
        print(json.dumps({"event": "cooldown", "seconds": 75, "gen_tps": gen_tps, "involuntary_context_switches": invol}, sort_keys=True), flush=True)
        time.sleep(75)


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


def resume_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,stat,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ && $2 ~ /^T/ {print $1}' | xargs -r kill -CONT"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        resume_codex_gpu_helper()
        raise
