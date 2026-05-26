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
    "threads": 4,
    "threads_batch": 4,
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "ridge-v2-drive",
    "timeout_seconds": 720,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "ridge_v2_drive_start"}), flush=True)
    emit(
        run_config(
            case(
                "ridge2_warmup_b2304_ub96_n32",
                2304,
                96,
                n_predict=32,
            )
        )
    )

    cases = [
        ("ridge2_record_repeat1", 2304, 96),
        ("ridge2_record_repeat2", 2304, 96),
        ("ridge2_b2176_ub96", 2176, 96),
        ("ridge2_b2240_ub96", 2240, 96),
        ("ridge2_b2368_ub96", 2368, 96),
        ("ridge2_b2432_ub96", 2432, 96),
        ("ridge2_b2304_ub72", 2304, 72),
        ("ridge2_b2304_ub80", 2304, 80),
        ("ridge2_b2304_ub88", 2304, 88),
        ("ridge2_b2304_ub104", 2304, 104),
        ("ridge2_b2240_ub88", 2240, 88),
        ("ridge2_b2240_ub104", 2240, 104),
        ("ridge2_b2368_ub88", 2368, 88),
        ("ridge2_b2368_ub104", 2368, 104),
        ("ridge2_b2400_ub96", 2400, 96),
        ("ridge2_b2464_ub96", 2464, 96),
    ]

    local_best: dict[str, Any] | None = None
    for label, batch, ubatch in cases:
        run = run_config(case(label, batch, ubatch))
        emit(run)
        if run.get("exit_code") == 0 and run.get("gen_tps") is not None:
            if local_best is None or float(run["gen_tps"]) > float(local_best["gen_tps"]):
                local_best = run
                print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)

    global_best = db.best_runs(limit=1)[0]
    print(
        json.dumps(
            {
                "event": "ridge_v2_drive_done",
                "local_best": summarize(local_best) if local_best else None,
                "global_best": summarize(global_best),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def case(label: str, batch: int, ubatch: int, **overrides: Any) -> dict[str, Any]:
    return dict(
        BASE,
        label=label,
        batch_size=batch,
        ubatch_size=ubatch,
        cache_type_k="q6_0",
        cache_type_v="q6_0",
        **overrides,
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
