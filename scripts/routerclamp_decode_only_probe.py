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
    "ctx_size": 16384,
    "batch_size": 2496,
    "ubatch_size": 96,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "env_ser_cheap_min": 2,
    "env_ser_cheap_thresh": 1.0,
    "env_ser_cheap_max_ntokens": 4,
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-decode-only-probe",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "decode_only_probe_start"}), flush=True)
        ranges = [
            "24:30",
            "6:12,24:30",
            "12:18,24:30",
            "6:18,24:30",
            "24:30,36:42",
            "6:12,12:18,24:30",
        ]
        summaries = []
        for r in ranges:
            speed = run_config(case(f"decode_only_speed_{labelize(r)}", r))
            print(json.dumps({"event": "speed", "ranges": r, **summarize(speed)}, sort_keys=True), flush=True)
            rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"decode_only_strict_{q['name']}_{labelize(r)}",
                    r,
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                    ignore_eos=False,
                ))
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                passed, reason = score_answer(q["name"], answer)
                row = {"quality": q["name"], "passed": passed, "reason": reason, "answer": answer[:220]}
                rows.append(row)
                print(json.dumps({"event": "quality", "ranges": r, **row, **summarize(run)}, sort_keys=True), flush=True)
                if not passed:
                    break
            summary = {
                "ranges": r,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in rows if row["passed"]),
                "strict_total": len(rows),
                "failed": next((row["quality"] for row in rows if not row["passed"]), None),
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        summaries.sort(key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0), reverse=True)
        print("=== DECODE ONLY PROBE ===", flush=True)
        for row in summaries:
            print(
                f"{row['ranges']:<22} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "decode_only_probe_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ranges: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(COMMON)
    cfg.update(label=label, env_ser_cheap_ranges=ranges)
    cfg.update(overrides)
    return cfg


def labelize(ranges: str) -> str:
    return ranges.replace(":", "_").replace(",", "_")


if __name__ == "__main__":
    main()
