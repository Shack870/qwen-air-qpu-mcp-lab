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
    "threads": 4,
    "threads_batch": 4,
    "smart_expert_reduction": "3,1",
    "extra_args": [],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "qpu-candidate-strict-probe",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "qpu_candidate_strict_probe_start"}), flush=True)
        candidates = [
            ("qpu_01001010_b2048_ub96_kq6_vq4", 2048, 96, "q6_0", "q4_1"),
            ("qpu_01000110_b2048_ub104_kq6_vq4", 2048, 104, "q6_0", "q4_1"),
            ("qpu_10001010_b1920_ub96_kq6_vq4", 1920, 96, "q6_0", "q4_1"),
            ("qpu_01010010_b2048_ub80_kq6_vq4", 2048, 80, "q6_0", "q4_1"),
            ("qpu_01001001_b2048_ub96_kq4_vq4", 2048, 96, "q4_1", "q4_1"),
            ("qpu_01001000_b2048_ub96_kq6_vq6", 2048, 96, "q6_0", "q6_0"),
        ]

        summaries = []
        for label, batch, ubatch, cache_k, cache_v in candidates:
            speed = run_config(case(f"{label}_speed", batch, ubatch, cache_k, cache_v))
            print(json.dumps({"event": "speed", "candidate": label, **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"{label}_strict_{q['name']}",
                    batch,
                    ubatch,
                    cache_k,
                    cache_v,
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
                print(json.dumps({"event": "quality", "candidate": label, **row}, sort_keys=True), flush=True)
                if not passed:
                    break

            summary = {
                "candidate": label,
                "batch": batch,
                "ubatch": ubatch,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in quality_rows if row["passed"]),
                "strict_total": len(quality_rows),
                "failed": next((row["quality"] for row in quality_rows if not row["passed"]), None),
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        leaders = sorted(summaries, key=lambda row: (row["strict_passes"], row["speed_gen_tps"] or 0), reverse=True)
        print("=== QPU CANDIDATE STRICT PROBE ===", flush=True)
        for row in leaders:
            print(
                f"{row['candidate']:<38} pass={row['strict_passes']}/{row['strict_total']} "
                f"gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} "
                f"k/v={row['cache_k']}/{row['cache_v']} b={row['batch']} ub={row['ubatch']} failed={row['failed']}",
                flush=True,
            )
        print(json.dumps({"event": "qpu_candidate_strict_probe_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, batch: int, ubatch: int, cache_k: str, cache_v: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(COMMON)
    cfg.update(
        label=label,
        batch_size=batch,
        ubatch_size=ubatch,
        cache_type_k=cache_k,
        cache_type_v=cache_v,
    )
    cfg.update(overrides)
    return cfg


if __name__ == "__main__":
    main()
