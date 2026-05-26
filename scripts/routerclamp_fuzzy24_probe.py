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


COMMON = {
    "prompt_key": "mars_fact_list",
    "prompt": "<|im_start|>user\nContinue this comma-separated list of Mars facts: red planet, thin atmosphere,<|im_end|>\n<|im_start|>assistant\n",
    "ctx_size": 16384,
    "batch_size": 2336,
    "ubatch_size": 104,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "env_ser_cheap_min": 1,
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-fuzzy24-probe",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "fuzzy24_probe_start"}), flush=True)
        candidates = [
            ("fuzzy24_30_t098", "24:30", 0.98),
            ("fuzzy24_30_t095", "24:30", 0.95),
            ("fuzzy24_30_t090", "24:30", 0.90),
            ("fuzzy24_30_t085", "24:30", 0.85),
            ("fuzzy24_30_t080", "24:30", 0.80),
            ("fuzzy24_30_t070", "24:30", 0.70),
            ("fuzzy24_28_t090", "24:28", 0.90),
            ("fuzzy24_28_t080", "24:28", 0.80),
            ("fuzzy25_30_t090", "25:30", 0.90),
            ("fuzzy25_30_t080", "25:30", 0.80),
            ("fuzzy26_30_t090", "26:30", 0.90),
            ("fuzzy26_30_t080", "26:30", 0.80),
        ]

        summaries = []
        for label, ranges, thresh in candidates:
            speed = run_config(case(f"fuzzy24_speed_{label}", ranges, thresh))
            print(json.dumps({"event": "speed", "candidate": label, "ranges": ranges, "thresh": thresh, **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"fuzzy24_strict_{q['name']}_{label}",
                    ranges,
                    thresh,
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
                print(json.dumps({"event": "quality", "candidate": label, "ranges": ranges, "thresh": thresh, **row}, sort_keys=True), flush=True)
                if not passed:
                    break

            summary = {
                "candidate": label,
                "ranges": ranges,
                "thresh": thresh,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in quality_rows if row["passed"]),
                "strict_total": len(quality_rows),
                "failed": next((row["quality"] for row in quality_rows if not row["passed"]), None),
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        leaders = sorted(summaries, key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0), reverse=True)
        print("=== FUZZY24 PROBE LEADERS ===", flush=True)
        for row in leaders:
            print(
                f"{row['candidate']:<20} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} ranges={row['ranges']} "
                f"thresh={row['thresh']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "fuzzy24_probe_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ranges: str, thresh: float, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(COMMON)
    cfg.update(
        label=label,
        env_ser_cheap_ranges=ranges,
        env_ser_cheap_thresh=thresh,
    )
    cfg.update(overrides)
    return cfg


if __name__ == "__main__":
    main()
