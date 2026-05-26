from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from typing import Any

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config
from qpu_mcp_lab.qpu_strategy import counts_to_candidates


PROMPT_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

IMPL_JOB_ID = "d8a9jg2s46sc73fbit5g"
BASELINE_FLOOR_TPS = 11.5
COOLDOWN_SECONDS = 45

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
    "source": "state-gated-impl-retest",
    "timeout_seconds": 720,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "state_gated_impl_retest_start"}), flush=True)
    emit(run_config(case("gated_warmup_n32", [], n_predict=32)))

    baseline = establish_baseline()
    candidates = list(hand_candidates())
    candidates.extend(qpu_candidates())
    print(
        json.dumps(
            {
                "event": "candidate_plan",
                "count": len(candidates),
                "labels": [label for label, _ in candidates],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best = baseline
    last_baseline = baseline
    for index, (label, extra_args) in enumerate(candidates, start=1):
        if index > 1 and (index - 1) % 5 == 0:
            last_baseline = run_config(case(f"gated_baseline_mid_{index:02d}", []))
            emit(last_baseline)
        run = run_config(case(label, extra_args))
        emit(run, baseline=last_baseline)
        if is_success(run) and (best is None or float(run["gen_tps"]) > float(best["gen_tps"])):
            best = run
            print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)

    final_baseline = run_config(case("gated_baseline_final", []))
    emit(final_baseline)
    if is_success(final_baseline) and (best is None or float(final_baseline["gen_tps"]) > float(best["gen_tps"])):
        best = final_baseline

    print(
        json.dumps(
            {
                "event": "state_gated_impl_retest_done",
                "local_best": summarize(best) if best else None,
                "global_best": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def establish_baseline() -> dict[str, Any] | None:
    for attempt in range(1, 4):
        run = run_config(case(f"gated_baseline_gate_{attempt}", []))
        emit(run)
        if is_success(run) and float(run["gen_tps"]) >= BASELINE_FLOOR_TPS:
            print(
                json.dumps(
                    {
                        "event": "baseline_gate_pass",
                        "attempt": attempt,
                        "gen_tps": run["gen_tps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
    return run


def hand_candidates() -> Iterable[tuple[str, list[str]]]:
    return [
        ("gated_ckpt_interval0", ["--ctx-checkpoints-interval", "0"]),
        ("gated_ckpt0", ["--ctx-checkpoints", "0"]),
        ("gated_amb512", ["-amb", "512"]),
        ("gated_mqkv", ["-mqkv"]),
        ("gated_muge", ["-muge"]),
        ("gated_mqkv_muge", ["-mqkv", "-muge"]),
        ("gated_mqkv_muge_ger", ["-mqkv", "-muge", "-ger"]),
        ("gated_sas", ["-sas"]),
        ("gated_khad", ["-khad"]),
        ("gated_vhad", ["-vhad"]),
        ("gated_ckpti0_amb512", ["--ctx-checkpoints-interval", "0", "-amb", "512"]),
        ("gated_ckpti0_mqkv", ["--ctx-checkpoints-interval", "0", "-mqkv"]),
        ("gated_ckpti0_muge", ["--ctx-checkpoints-interval", "0", "-muge"]),
        ("gated_ckpti0_mqkv_muge", ["--ctx-checkpoints-interval", "0", "-mqkv", "-muge"]),
        ("gated_ckpti0_mqkv_muge_ger", ["--ctx-checkpoints-interval", "0", "-mqkv", "-muge", "-ger"]),
    ]


def qpu_candidates() -> list[tuple[str, list[str]]]:
    job = db.get_quantum_job(IMPL_JOB_ID)
    if not job or not job.get("counts") or not job.get("payload"):
        return []
    rows = counts_to_candidates(IMPL_JOB_ID, job["counts"], job["payload"], top_k=10)
    result: list[tuple[str, list[str]]] = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        cfg = row["config"]
        extra_args = list(cfg.get("extra_args") or [])
        key = tuple(extra_args)
        if key in seen:
            continue
        seen.add(key)
        suffix = "_".join(arg.strip("-").replace(",", "-") for arg in extra_args if not arg.isdigit()) or "baseline"
        result.append((f"gated_qpu_impl_{index:02d}_{suffix}"[:80], extra_args))
    return result


def case(label: str, extra_args: list[str], **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, extra_args=extra_args, **overrides)


def emit(run: dict[str, Any] | None, baseline: dict[str, Any] | None = None) -> None:
    payload = {"event": "run", **summarize(run)}
    if baseline and is_success(run) and is_success(baseline):
        payload["delta_vs_baseline_tps"] = round(float(run["gen_tps"]) - float(baseline["gen_tps"]), 3)
    print(json.dumps(payload, sort_keys=True), flush=True)


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
