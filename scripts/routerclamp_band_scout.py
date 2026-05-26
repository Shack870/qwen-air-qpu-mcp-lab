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


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "band_scout_start"}), flush=True)

        candidates = [
            ("baseline_ser3_1", "3,1", None),
            ("cheap_ser2_065", "2,0.65", None),
            ("ser2_065_full_00_06", "2,0.65", "0:6"),
            ("ser2_065_full_06_12", "2,0.65", "6:12"),
            ("ser2_065_full_12_18", "2,0.65", "12:18"),
            ("ser2_065_full_18_24", "2,0.65", "18:24"),
            ("ser2_065_full_24_30", "2,0.65", "24:30"),
            ("ser2_065_full_30_36", "2,0.65", "30:36"),
            ("ser2_065_full_36_42", "2,0.65", "36:42"),
            ("ser2_065_full_42_48", "2,0.65", "42:48"),
            ("ser2_065_full_12_24", "2,0.65", "12:24"),
            ("ser2_065_full_18_30", "2,0.65", "18:30"),
            ("ser2_065_full_24_36", "2,0.65", "24:36"),
            ("ser2_065_full_30_42", "2,0.65", "30:42"),
            ("ser2_065_full_12_18_30_36", "2,0.65", "12:18,30:36"),
            ("ser2_065_full_18_24_30_36", "2,0.65", "18:24,30:36"),
        ]

        summaries = []
        for label, ser, ranges in candidates:
            speed = run_config(case(f"band_speed_{label}", ser, ranges))
            print(json.dumps({"event": "speed", "candidate": label, "ranges": ranges, **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"band_strict_{q['name']}_{label}",
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
                    "answer": answer[:240],
                    "gen_tps": run.get("gen_tps"),
                    "log_path": run.get("log_path"),
                }
                quality_rows.append(row)
                print(json.dumps({"event": "quality", "candidate": label, "ranges": ranges, **row}, sort_keys=True), flush=True)
                if not passed:
                    print(json.dumps({
                        "event": "early_stop_candidate",
                        "candidate": label,
                        "ranges": ranges,
                        "failed": q["name"],
                        "reason": reason,
                    }, sort_keys=True), flush=True)
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

        leaders = sorted(
            summaries,
            key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0),
            reverse=True,
        )
        print("=== BAND SCOUT LEADERS ===", flush=True)
        for row in leaders:
            print(
                f"{row['candidate']:<32} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} ranges={row['ranges']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "band_scout_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ser: str, ranges: str | None, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE, label=label, source="routerclamp-band-scout", smart_expert_reduction=ser, **overrides)
    if ranges:
        cfg["env_ser_full_ranges"] = ranges
    return cfg


if __name__ == "__main__":
    main()
