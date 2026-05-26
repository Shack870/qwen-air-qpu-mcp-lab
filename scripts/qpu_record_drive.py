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
    "source": "qpu-record-drive",
    "timeout_seconds": 720,
}


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "qpu_record_drive_start"}), flush=True)

    warmup = dict(
        BASE,
        label="qpu_drive_warmup_b2304_ub96_n32",
        batch_size=2304,
        ubatch_size=96,
        cache_type_k="q6_0",
        cache_type_v="q6_0",
        n_predict=32,
    )
    emit(run_config(warmup))

    cases: list[dict[str, Any]] = [
        # IBM record-lane QPU suggestions, highest energy advantage first.
        case("qpu_top1_b2304_ub96_repeat1", 2304, 96, "q6_0", "q6_0"),
        case("qpu_top1_b2304_ub96_repeat2", 2304, 96, "q6_0", "q6_0"),
        case("qpu_b3072_ub96_q6q6", 3072, 96, "q6_0", "q6_0"),
        case("qpu_b2304_ub96_k6v4", 2304, 96, "q6_0", "q4_1"),
        case("qpu_b2304_ub128_q6q6", 2304, 128, "q6_0", "q6_0"),
        case("qpu_b2560_ub96_q6q6", 2560, 96, "q6_0", "q6_0"),
        case("qpu_b2816_ub96_q6q6", 2816, 96, "q6_0", "q6_0"),
        case("qpu_b2304_ub112_q6q6", 2304, 112, "q6_0", "q6_0"),
        case("qpu_b2304_ub96_q4q4", 2304, 96, "q4_1", "q4_1"),
        case("qpu_b3328_ub96_q6q6", 3328, 96, "q6_0", "q6_0"),
        case("qpu_b2304_ub80_q6q6", 2304, 80, "q6_0", "q6_0"),
        case("qpu_b1792_ub96_q6q6", 1792, 96, "q6_0", "q6_0"),
        case("qpu_b3072_ub96_q4q4", 3072, 96, "q4_1", "q4_1"),
        # Cheap runtime mutants around the winning lane.
        case("runtime_b2304_ub96_mallocnano0", 2304, 96, "q6_0", "q6_0", env_malloc_nano_zone="0"),
        case("runtime_b2304_ub96_nowarmup", 2304, 96, "q6_0", "q6_0", no_warmup=True),
        case("runtime_b2304_ub96_omp_passive", 2304, 96, "q6_0", "q6_0", env_omp_wait_policy="PASSIVE"),
        case("runtime_b2304_ub96_t3", 2304, 96, "q6_0", "q6_0", threads=3, threads_batch=3),
        case("runtime_b2304_ub96_tb2", 2304, 96, "q6_0", "q6_0", threads_batch=2),
        # Routerclamp-ish expert reduction probes. These are the wild cards.
        case("router_b2304_ub96_ser2_1", 2304, 96, "q6_0", "q6_0", smart_expert_reduction="2,1"),
        case("router_b2304_ub96_ser2_085", 2304, 96, "q6_0", "q6_0", smart_expert_reduction="2,0.85"),
        case("router_b2304_ub96_ser4_1", 2304, 96, "q6_0", "q6_0", smart_expert_reduction="4,1"),
        case("router_b2304_ub96_noser", 2304, 96, "q6_0", "q6_0", smart_expert_reduction=None),
        # Speculative/n-gram probes, useful if this prompt's continuation becomes predictable.
        case(
            "spec_b2304_ub96_ngram16",
            2304,
            96,
            "q6_0",
            "q6_0",
            extra_args=["--spec-type", "ngram-mod", "--draft-max", "16", "--draft-min", "1"],
        ),
        case(
            "spec_b2304_ub96_ngram32",
            2304,
            96,
            "q6_0",
            "q6_0",
            extra_args=["--spec-type", "ngram-mod", "--draft-max", "32", "--draft-min", "1"],
        ),
    ]

    local_best: dict[str, Any] | None = None
    for cfg in cases:
        run = run_config(cfg)
        emit(run)
        if run.get("exit_code") == 0 and run.get("gen_tps") is not None:
            if local_best is None or float(run["gen_tps"]) > float(local_best["gen_tps"]):
                local_best = run
                print(
                    json.dumps(
                        {
                            "event": "new_local_best",
                            "label": run.get("label"),
                            "gen_tps": run.get("gen_tps"),
                            "log_path": run.get("log_path"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    global_best = db.best_runs(limit=1)[0]
    print(
        json.dumps(
            {
                "event": "qpu_record_drive_done",
                "local_best": summarize(local_best) if local_best else None,
                "global_best": summarize(global_best),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def case(label: str, batch: int, ubatch: int, ctk: str, ctv: str, **overrides: Any) -> dict[str, Any]:
    return dict(
        BASE,
        label=label,
        batch_size=batch,
        ubatch_size=ubatch,
        cache_type_k=ctk,
        cache_type_v=ctv,
        **overrides,
    )


def emit(run: dict[str, Any]) -> None:
    print(json.dumps({"event": "run", **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
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
