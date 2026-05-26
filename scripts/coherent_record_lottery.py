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

RECORD_TPS = 14.03
THERMAL_SPEED_GATE = 65
THERMAL_STRICT_GATE = 85


def main() -> None:
    stop_codex_gpu_helper()
    best: dict[str, Any] | None = None
    try:
        print_event("coherent_record_lottery_start", thermal=thermal())
        wait_for_thermal(THERMAL_SPEED_GATE, "initial_cooldown")
        warmup = run_config(make_case("lottery_warmup_b2456_ub144_n32", 2456, 144, n_predict=32))
        print_event("warmup", **summarize(warmup), thermal_after=thermal())
        time.sleep(10)

        for label, batch, ubatch, extra_args in candidates():
            wait_for_thermal(THERMAL_SPEED_GATE, "speed_cooldown")
            run = run_config(make_case(label, batch, ubatch, extra_args=extra_args))
            print_event("speed", **summarize(run), thermal_after=thermal())
            if run.get("gen_tps") and (best is None or float(run["gen_tps"]) > float(best.get("gen_tps") or 0.0)):
                best = run
            time.sleep(8)

        if best and float(best.get("gen_tps") or 0.0) > RECORD_TPS:
            print_event("new_record_candidate", **summarize(best), thermal=thermal())
            rows = strict_quality(best)
            all_passed = len(rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in rows)
            print_event("strict_done", all_passed=all_passed, rows=rows)
            if all_passed:
                crown(best, rows)
        else:
            print_event("no_new_record", best=summarize(best) if best else None, thermal=thermal())
    finally:
        resume_codex_gpu_helper()


def candidates() -> list[tuple[str, int, int, list[str] | None]]:
    base = [
        ("lottery_exact_b2456_ub144_r1", 2456, 144, None),
        ("lottery_exact_b2456_ub144_r2", 2456, 144, None),
        ("lottery_exact_b2456_ub144_r3", 2456, 144, None),
        ("lottery_b2456_ub146_r1", 2456, 146, None),
        ("lottery_b2456_ub146_r2", 2456, 146, None),
        ("lottery_b2456_ub146_ckpti0", 2456, 146, ["--ctx-checkpoints-interval", "0"]),
        ("lottery_b2464_ub140_r1", 2464, 140, None),
        ("lottery_b2464_ub148_r1", 2464, 148, None),
        ("lottery_b2496_ub128_r1", 2496, 128, None),
        ("lottery_b2560_ub128_r1", 2560, 128, None),
        ("lottery_b2560_ub120_r1", 2560, 120, None),
        ("lottery_b2304_ub144_r1", 2304, 144, None),
        ("lottery_b2320_ub144_r1", 2320, 144, None),
    ]
    # Re-test the strongest two after the machine is fully warm and the model
    # pages are settled. Several previous highs happened after a discard run.
    base.extend(
        [
            ("lottery_late_b2456_ub144", 2456, 144, None),
            ("lottery_late_b2456_ub146", 2456, 146, None),
            ("lottery_late_b2496_ub128", 2496, 128, None),
        ]
    )
    return base


def make_case(
    label: str,
    batch: int,
    ubatch: int,
    *,
    extra_args: list[str] | None = None,
    n_predict: int = 128,
) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(
        {
            "label": label,
            "ctx_size": 16384,
            "batch_size": batch,
            "ubatch_size": ubatch,
            "threads": 4,
            "threads_batch": 4,
            "cache_type_k": "q6_0",
            "cache_type_v": "q6_0",
            "smart_expert_reduction": "3,1",
            "env_ser_cheap_ranges": "24:30",
            "env_ser_cheap_min": 2,
            "env_ser_cheap_thresh": 1.0,
            "n_predict": n_predict,
            "temp": 0.0,
            "ignore_eos": True,
            "no_display_prompt": True,
            "source": "coherent-record-lottery",
            "timeout_seconds": 900,
        }
    )
    if extra_args:
        cfg["extra_args"] = extra_args
    return cfg


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "coherent-record-lottery-strict", "ignore_eos": False, "no_display_prompt": True})
    rows = []
    for prompt in QUALITY_PROMPTS:
        wait_for_thermal(THERMAL_STRICT_GATE, f"strict_cooldown_{prompt['name']}")
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
            "answer": answer[:400],
            "thermal_after": thermal(),
            **summarize(run),
        }
        rows.append(row)
        print_event("strict", **row)
        if not passed:
            break
    return rows


def crown(speed_run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    with db.connect() as con:
        con.execute(
            """
            UPDATE runs
            SET quality_flag = 'strict_passed',
                notes = COALESCE(notes || '; ', '') || ?
            WHERE id = ?
            """,
            (
                "strict quality passed by coherent_record_lottery: "
                + ",".join(row["quality"] for row in rows),
                int(speed_run["id"]),
            ),
        )
    print_event("crowned", **summarize(speed_run))


def thermal() -> int | None:
    try:
        return int(
            subprocess.check_output(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                text=True,
                timeout=5,
            ).strip()
        )
    except Exception:
        return None


def wait_for_thermal(max_level: int, event: str) -> None:
    while True:
        level = thermal()
        if level is None or level <= max_level:
            return
        print_event(event, thermal=level, max_level=max_level)
        time.sleep(20)


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
