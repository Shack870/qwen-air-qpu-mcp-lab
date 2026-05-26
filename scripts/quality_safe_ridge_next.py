from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import (
    BASE,
    QUALITY_PROMPTS,
    generated_answer,
    resume_codex_gpu_helper,
    score_answer,
    stop_codex_gpu_helper,
    summarize,
)

CURRENT_QUALITY_SAFE_RECORD = 14.03
# Apple reports these as relative thermal pressure levels, not degrees.
# Warm runs are fine; heat-soaked starts have repeatedly produced page-fault
# storms and worse throughput, so record attempts wait for a moderate signal.
SPEED_THERMAL_GATE = 60
STRICT_THERMAL_GATE = 80


def main() -> None:
    stop_codex_gpu_helper()
    best: dict[str, Any] | None = None
    try:
        print_event("start", thermal=thermal())
        wait_for_thermal(SPEED_THERMAL_GATE, "initial_cooldown")
        for case_cfg in speed_cases():
            wait_for_thermal(SPEED_THERMAL_GATE, "between_runs")
            run = run_config(case_cfg)
            row = summarize(run)
            row["thermal_after"] = thermal()
            print_event("speed_result", **row)
            if run.get("gen_tps") and (best is None or float(run["gen_tps"]) > float(best.get("gen_tps") or 0.0)):
                best = run
            time.sleep(20)

        if best and float(best.get("gen_tps") or 0.0) > CURRENT_QUALITY_SAFE_RECORD:
            print_event("record_candidate", **summarize(best), thermal=thermal())
            quality_rows = strict_quality(best)
            all_passed = len(quality_rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in quality_rows)
            print_event("record_quality_done", all_passed=all_passed, rows=quality_rows)
            if all_passed:
                crown(best, quality_rows)
        else:
            print_event("no_new_speed_record", best=summarize(best) if best else None, thermal=thermal())
    finally:
        resume_codex_gpu_helper()


def speed_cases() -> list[dict[str, Any]]:
    base = record_base()
    candidates = [
        ("ridge_repeat_b2456_ub144", 2456, 144, "3,1"),
        ("ridge_b2464_ub140_r2", 2464, 140, "3,1"),
        ("ridge_b2464_ub148_r2", 2464, 148, "3,1"),
        ("ridge_b2460_ub144", 2460, 144, "3,1"),
        ("ridge_b2468_ub144", 2468, 144, "3,1"),
        ("qpu_mutation_b2456_ub144_ser2_085", 2456, 144, "2,0.85"),
    ]
    return [
        dict(
            base,
            label=label,
            batch_size=batch,
            ubatch_size=ubatch,
            smart_expert_reduction=ser,
        )
        for label, batch, ubatch, ser in candidates
    ]


def record_base() -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(
        {
            "ctx_size": 16384,
            "threads": 4,
            "threads_batch": 4,
            "cache_type_k": "q6_0",
            "cache_type_v": "q6_0",
            "smart_expert_reduction": "3,1",
            "env_ser_cheap_ranges": "24:30",
            "env_ser_cheap_min": 2,
            "env_ser_cheap_thresh": 1.0,
            "n_predict": 128,
            "temp": 0.0,
            "ignore_eos": True,
            "no_display_prompt": True,
            "source": "quality-safe-ridge-next",
            "timeout_seconds": 900,
        }
    )
    return cfg


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update(
        {
            "source": "quality-safe-ridge-next-strict",
            "ignore_eos": False,
            "no_display_prompt": True,
            "timeout_seconds": 900,
        }
    )
    rows = []
    for prompt in QUALITY_PROMPTS:
        wait_for_thermal(STRICT_THERMAL_GATE, f"strict_cooldown_{prompt['name']}")
        run = run_config(
            dict(
                cfg,
                label=f"{speed_run['label']}_strict_{prompt['name']}",
                prompt=prompt["prompt"],
                n_predict=prompt["n_predict"],
            )
        )
        answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
        passed, reason = score_answer(prompt["name"], answer)
        row = {
            "quality": prompt["name"],
            "passed": passed,
            "reason": reason,
            "answer": answer[:500],
            "thermal_after": thermal(),
            **summarize(run),
        }
        rows.append(row)
        print_event("strict_result", **row)
        if not passed:
            break
    return rows


def crown(speed_run: dict[str, Any], quality_rows: list[dict[str, Any]]) -> None:
    with db.connect() as con:
        con.execute(
            """
            UPDATE runs
            SET quality_flag = 'strict_passed',
                notes = COALESCE(notes || '; ', '') || ?
            WHERE id = ?
            """,
            (
                "strict quality passed by quality_safe_ridge_next: "
                + ",".join(row["quality"] for row in quality_rows),
                int(speed_run["id"]),
            ),
        )
    print_event("crowned", **summarize(speed_run))


def thermal() -> int | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
            text=True,
            timeout=5,
        ).strip()
        return int(out)
    except Exception:
        return None


def wait_for_thermal(max_level: int, event: str) -> None:
    while True:
        level = thermal()
        if level is None or level <= max_level:
            return
        print_event(event, thermal=level, max_level=max_level)
        time.sleep(30)


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
