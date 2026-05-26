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


RECORD = {
    "batch_size": 2560,
    "ubatch_size": 96,
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "cheap_band_probe_start"}), flush=True)
        candidates = [
            ("baseline_ser3_1", None, None, None),
            ("cheap1_00_06", "0:6", 1, 5.0),
            ("cheap1_06_12", "6:12", 1, 5.0),
            ("cheap1_12_18", "12:18", 1, 5.0),
            ("cheap1_18_24", "18:24", 1, 5.0),
            ("cheap1_24_30", "24:30", 1, 5.0),
            ("cheap1_30_36", "30:36", 1, 5.0),
            ("cheap1_36_42", "36:42", 1, 5.0),
            ("cheap1_42_48", "42:48", 1, 5.0),
            ("cheap2_00_06", "0:6", 2, 1.0),
            ("cheap2_18_24", "18:24", 2, 1.0),
            ("cheap2_24_30", "24:30", 2, 1.0),
            ("cheap2_42_48", "42:48", 2, 1.0),
        ]

        summaries = []
        for label, ranges, cheap_min, cheap_thresh in candidates:
            speed = run_config(case(f"cheap_band_speed_{label}", ranges, cheap_min, cheap_thresh))
            print(json.dumps({"event": "speed", "candidate": label, "ranges": ranges, "cheap_min": cheap_min, "cheap_thresh": cheap_thresh, **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"cheap_band_strict_{q['name']}_{label}",
                    ranges,
                    cheap_min,
                    cheap_thresh,
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
                "ranges": ranges,
                "cheap_min": cheap_min,
                "cheap_thresh": cheap_thresh,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in quality_rows if row["passed"]),
                "strict_total": len(quality_rows),
                "failed": next((row["quality"] for row in quality_rows if not row["passed"]), None),
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        leaders = sorted(summaries, key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0), reverse=True)
        print("=== CHEAP BAND PROBE LEADERS ===", flush=True)
        for row in leaders:
            print(
                f"{row['candidate']:<20} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} ranges={row['ranges']} "
                f"cheap={row['cheap_min']},{row['cheap_thresh']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "cheap_band_probe_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ranges: str | None, cheap_min: int | None, cheap_thresh: float | None, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(RECORD)
    cfg.update(
        label=label,
        source="routerclamp-cheap-band-probe",
        smart_expert_reduction="3,1",
    )
    if ranges:
        cfg.update(
            env_ser_cheap_ranges=ranges,
            env_ser_cheap_min=cheap_min,
            env_ser_cheap_thresh=cheap_thresh,
        )
    cfg.update(overrides)
    return cfg


if __name__ == "__main__":
    main()
