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
    BASE,
    QUALITY_PROMPTS,
    generated_answer,
    score_answer,
    stop_codex_gpu_helper,
    resume_codex_gpu_helper,
    summarize,
)


RECORD_OVERRIDES: dict[str, Any] = {
    "batch_size": 2560,
    "ubatch_size": 96,
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "record_replay_start"}), flush=True)
        candidates = [
            ("record_baseline_ser3_1", "3,1", None),
            ("record_ser2_065_full_18_24", "2,0.65", "18:24"),
            ("record_ser2_065_full_24_30", "2,0.65", "24:30"),
            ("record_ser2_065_full_42_48", "2,0.65", "42:48"),
            ("record_ser2_065_full_18_24_30_36", "2,0.65", "18:24,30:36"),
        ]

        summaries = []
        for label, ser, ranges in candidates:
            speed = run_config(case(f"record_speed_{label}", ser, ranges))
            print(json.dumps({"event": "speed", "candidate": label, "ranges": ranges, **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"record_strict_{q['name']}_{label}",
                    ser,
                    ranges,
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                    ignore_eos=False,
                ))
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                passed, reason = score_answer(q["name"], answer)
                row = {
                    "quality": q["name"],
                    "passed": passed,
                    "reason": reason,
                    "answer": answer[:220],
                    "gen_tps": run.get("gen_tps"),
                }
                quality_rows.append(row)
                print(json.dumps({"event": "quality", "candidate": label, "ranges": ranges, **row}, sort_keys=True), flush=True)
                if not passed:
                    break

            summary = {
                "candidate": label,
                "ser": ser,
                "ranges": ranges,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in quality_rows if row["passed"]),
                "strict_total": len(quality_rows),
                "failed": next((row["quality"] for row in quality_rows if not row["passed"]), None),
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        leaders = sorted(summaries, key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0), reverse=True)
        print("=== RECORD REPLAY LEADERS ===", flush=True)
        for row in leaders:
            print(
                f"{row['candidate']:<40} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} ranges={row['ranges']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "record_replay_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ser: str, ranges: str | None, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(RECORD_OVERRIDES)
    cfg.update(
        label=label,
        source="routerclamp-record-replay",
        smart_expert_reduction=ser,
    )
    cfg.update(overrides)
    if ranges:
        cfg["env_ser_full_ranges"] = ranges
    return cfg


if __name__ == "__main__":
    main()
