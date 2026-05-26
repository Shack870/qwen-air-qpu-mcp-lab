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
    "source": "top-lane-replay",
    "timeout_seconds": 720,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "top_lane_replay_start"}), flush=True)
    emit(run_config(dict(BASE, label="top_lane_replay_warmup_n32", n_predict=32)))
    best = None
    for i in range(1, 9):
        run = run_config(dict(BASE, label=f"top_lane_replay_{i:02d}"))
        emit(run)
        if run.get("exit_code") == 0 and run.get("gen_tps") is not None:
            if best is None or float(run["gen_tps"]) > float(best["gen_tps"]):
                best = run
                print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "event": "top_lane_replay_done",
                "local_best": summarize(best) if best else None,
                "global_best": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def emit(run: dict[str, Any]) -> None:
    print(json.dumps({"event": "run", **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": run.get("label"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    main()
