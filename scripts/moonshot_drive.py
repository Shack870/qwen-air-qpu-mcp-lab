from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config


ROOT = Path(__file__).resolve().parents[1]
SUFFIX_CORPUS = ROOT / "data" / "moonshot_suffix_corpus.json"

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
    "source": "moonshot-drive",
    "timeout_seconds": 900,
}

BASELINE_FLOOR_TPS = 12.2
COOLDOWN_SECONDS = 150


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "moonshot_start", "suffix_corpus": str(SUFFIX_CORPUS)}), flush=True)
    best_before = db.best_runs(limit=1)[0]
    print(json.dumps({"event": "global_best_before", **summarize(best_before)}, sort_keys=True), flush=True)

    baseline = establish_baseline()
    local_best = baseline

    for cfg in moonshot_cases():
        run = run_config(cfg)
        emit(run)
        if is_success(run) and (local_best is None or float(run["gen_tps"]) > float(local_best["gen_tps"])):
            local_best = run
            print(json.dumps({"event": "new_local_best", **summarize(run)}, sort_keys=True), flush=True)
        if is_success(run) and float(run["gen_tps"]) > float(best_before["gen_tps"]):
            print(json.dumps({"event": "new_global_candidate", **summarize(run)}, sort_keys=True), flush=True)
        cool_if_hot(run)

    print(
        json.dumps(
            {
                "event": "moonshot_done",
                "local_best": summarize(local_best),
                "global_best_after": summarize(db.best_runs(limit=1)[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resume_codex_gpu_helper()


def establish_baseline() -> dict[str, Any] | None:
    last = None
    emit(run_config(case("moonshot_warmup_n32", [], n_predict=32)))
    for attempt in range(1, 4):
        run = run_config(case(f"moonshot_baseline_gate_{attempt}", []))
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
                        "gen_tps": run.get("gen_tps") if run else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(COOLDOWN_SECONDS)
    print(json.dumps({"event": "baseline_gate_low_state"}, sort_keys=True), flush=True)
    return last


def moonshot_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # Honest single-stream record lane. These seek another raw decode high-water mark.
    cases += [
        case("moonshot_raw_b2304_ub96_repeat_a", []),
        case("moonshot_raw_b2240_ub96", [], batch_size=2240),
        case("moonshot_raw_b2368_ub96", [], batch_size=2368),
        case("moonshot_raw_b2304_ub88", [], ubatch_size=88),
        case("moonshot_raw_b2304_ub104", [], ubatch_size=104),
        case("moonshot_raw_b2304_ub96_ckpt0", ["--ctx-checkpoints", "0"]),
    ]

    # Aggregate throughput lane. If these win, they are multi-sequence throughput records,
    # not single-user latency records.
    for parallel, batch, ubatch in [
        (2, 2304, 96),
        (2, 2560, 96),
        (4, 2304, 96),
        (4, 3072, 96),
        (8, 2304, 96),
    ]:
        cases.append(
            case(
                f"moonshot_aggregate_np{parallel}_b{batch}_ub{ubatch}",
                ["-np", str(parallel), "-ns", str(parallel), "-pps"],
                batch_size=batch,
                ubatch_size=ubatch,
                n_predict=96,
            )
        )

    # Speculative/corpus-assisted lane. These are allowed to look magical on repetitive
    # list/code workloads, but must be reported separately from raw decode.
    cases += [
        case(
            "moonshot_spec_ngram_mod_autotune",
            ["--spec-type", "ngram-mod", "--draft-max", "64", "--draft-min", "2", "--spec-autotune"],
            n_predict=160,
        ),
        case(
            "moonshot_spec_ngram_map_k_8_64",
            [
                "--spec-type",
                "ngram-map-k",
                "--spec-ngram-size-n",
                "8",
                "--spec-ngram-size-m",
                "64",
                "--spec-ngram-min-hits",
                "1",
                "--draft-max",
                "64",
                "--draft-min",
                "2",
            ],
            n_predict=160,
        ),
        case(
            "moonshot_spec_suffix_mars_depth64",
            [
                "--spec-type",
                "suffix",
                "--suffix-corpus",
                str(SUFFIX_CORPUS),
                "--suffix-pattern-len",
                "3",
                "--suffix-max-depth",
                "64",
                "--draft-max",
                "64",
                "--draft-min",
                "2",
            ],
            n_predict=160,
        ),
        case(
            "moonshot_spec_suffix_mars_depth128",
            [
                "--spec-type",
                "suffix",
                "--suffix-corpus",
                str(SUFFIX_CORPUS),
                "--suffix-pattern-len",
                "2",
                "--suffix-max-depth",
                "128",
                "--draft-max",
                "96",
                "--draft-min",
                "2",
                "--spec-autotune",
            ],
            n_predict=192,
        ),
    ]

    # Prompt-cache lane. This targets perceived repeated-prompt startup, not steady-state tg t/s.
    cache_path = ROOT / "data" / "moonshot_prompt.cache"
    cases += [
        case("moonshot_prompt_cache_write", ["--prompt-cache", str(cache_path)], n_predict=96),
        case("moonshot_prompt_cache_ro", ["--prompt-cache", str(cache_path), "--prompt-cache-ro"], n_predict=96),
    ]

    cases.append(case("moonshot_raw_b2304_ub96_repeat_b", []))
    return cases


def case(label: str, extra_args: list[str], **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, extra_args=extra_args, **overrides)


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


def cool_if_hot(run: dict[str, Any] | None) -> None:
    if not run:
        return
    metrics = run.get("metrics") or {}
    invol = int(metrics.get("involuntary_context_switches") or 0)
    page_reclaims = int(metrics.get("page_reclaims") or 0)
    gen_tps = float(run.get("gen_tps") or 0.0)
    if invol > 70000 or page_reclaims > 1800000 or (gen_tps and gen_tps < 7.0):
        print(
            json.dumps(
                {
                    "event": "cooldown",
                    "seconds": 90,
                    "reason": "hot_or_noisy_state",
                    "gen_tps": run.get("gen_tps"),
                    "involuntary_context_switches": invol,
                    "page_reclaims": page_reclaims,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(90)


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
