from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import (
    QUALITY_PROMPTS,
    generated_answer,
    resume_codex_gpu_helper,
    score_answer,
    stop_codex_gpu_helper,
    summarize,
)


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
    "batch_size": 2496,
    "ubatch_size": 128,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "env_ser_cheap_ranges": "24:30",
    "env_ser_cheap_min": 2,
    "env_ser_cheap_thresh": 1.0,
    "extra_args": [],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "record-edge-knob-drive",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "record_edge_knob_drive_start"}), flush=True)
        candidates = [
            case("edge_baseline_b2496_ub128"),
            case("edge_repeat_b2496_ub128"),
            case("edge_b2304_ub144", batch_size=2304, ubatch_size=144),
            case("edge_b2560_ub128", batch_size=2560, ubatch_size=128),
            case("edge_amb256", extra_args=["-amb", "256"]),
            case("edge_amb512", extra_args=["-amb", "512"]),
            case("edge_amb768", extra_args=["-amb", "768"]),
            case("edge_amb1024", extra_args=["-amb", "1024"]),
            case("edge_nocb", extra_args=["-nocb"]),
            case("edge_nocb_amb512", extra_args=["-nocb", "-amb", "512"]),
            case("edge_ckpt0", extra_args=["--ctx-checkpoints", "0"]),
            case("edge_ckpt_interval0", extra_args=["--ctx-checkpoints-interval", "0"]),
            case("edge_no_warmup", no_warmup=True),
            case("edge_runtime_repack", extra_args=["--run-time-repack"]),
            case("edge_mqkv", extra_args=["-mqkv"]),
            case("edge_muge", extra_args=["-muge"]),
            case("edge_mqkv_muge", extra_args=["-mqkv", "-muge"]),
            case("edge_malloc_nano0", env_malloc_nano_zone="0"),
            case("edge_veclib2", env_veclib_threads=2),
            case("edge_omp_passive", env_omp_wait_policy="PASSIVE"),
            case("edge_omp_bind_close", env_omp_proc_bind="close", env_omp_places="cores"),
            case("edge_omp_bind_spread", env_omp_proc_bind="spread", env_omp_places="cores"),
        ]

        runs = []
        for cfg in candidates:
            run = run_config(cfg)
            runs.append(run)
            print(json.dumps({"event": "speed", **summarize(run)}, sort_keys=True), flush=True)

        successful = [
            run
            for run in runs
            if run.get("exit_code") == 0 and run.get("gen_tps") is not None
        ]
        successful.sort(key=lambda run: float(run["gen_tps"]), reverse=True)

        quality_targets = []
        for run in successful[:4]:
            quality_targets.append(run)
        for run in successful:
            if float(run["gen_tps"]) >= 14.0 and run not in quality_targets:
                quality_targets.append(run)

        quality_summaries = []
        for winner in quality_targets:
            quality_summaries.append(validate_quality(winner))

        print("=== RECORD EDGE KNOB DRIVE ===", flush=True)
        for run in successful:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<28} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} args={cfg.get('extra_args')} "
                f"warmup={cfg.get('no_warmup')} vec={cfg.get('env_veclib_threads')} "
                f"wait={cfg.get('env_omp_wait_policy')} faults={run.get('metrics', {}).get('page_faults')} "
                f"rss={run.get('peak_rss_bytes')}",
                flush=True,
            )

        passed = [
            row
            for row in quality_summaries
            if row["strict_passes"] == row["strict_total"] and row["strict_total"] == len(QUALITY_PROMPTS)
        ]
        passed.sort(key=lambda row: float(row["speed_gen_tps"] or 0), reverse=True)
        print("=== STRICT-PASSED EDGE LEADERS ===", flush=True)
        for row in passed:
            print(
                f"{row['candidate']:<28} gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} "
                f"strict={row['strict_passes']}/{row['strict_total']}",
                flush=True,
            )

        print(
            json.dumps(
                {
                    "event": "record_edge_knob_drive_done",
                    "speed_best": summarize(successful[0] if successful else None),
                    "quality_best": passed[0] if passed else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        resume_codex_gpu_helper()


def case(label: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(label=label)
    cfg.update(overrides)
    return cfg


def validate_quality(speed_run: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(speed_run["config_json"])
    rows = []
    for q in QUALITY_PROMPTS:
        run_cfg = dict(
            cfg,
            label=f"edge_quality_{q['name']}_{speed_run['label']}",
            prompt=q["prompt"],
            n_predict=q["n_predict"],
            ignore_eos=False,
            source="record-edge-knob-quality",
        )
        run = run_config(run_cfg)
        answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
        passed, reason = score_answer(q["name"], answer)
        row = {
            "quality": q["name"],
            "passed": passed,
            "reason": reason,
            "answer": answer[:260],
            **summarize(run),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "quality",
                    "candidate": speed_run["label"],
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not passed:
            break

    summary = {
        "candidate": speed_run["label"],
        "speed_gen_tps": speed_run.get("gen_tps"),
        "speed_pp_tps": speed_run.get("pp_tps"),
        "strict_passes": sum(1 for row in rows if row["passed"]),
        "strict_total": len(rows),
        "failed": next((row["quality"] for row in rows if not row["passed"]), None),
    }
    print(json.dumps({"event": "quality_summary", **summary}, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
