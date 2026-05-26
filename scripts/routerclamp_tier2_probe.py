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


PROMPT = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": PROMPT,
    "ctx_size": 16384,
    "batch_size": 2560,
    "ubatch_size": 128,
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
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-tier2-probe",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "tier2_probe_start"}), flush=True)
        candidates = [
            case("tier2_control_b2560_ub128"),
            case("tier2_control_b2496_ub128", batch_size=2496),
        ]
        for thresh in (1.0, 5.0):
            suffix = f"t{str(thresh).replace('.', '_')}"
            for r in ("24:25", "25:26", "26:27", "27:28", "28:29", "29:30"):
                candidates.append(case(f"tier2_top1_{labelize(r)}_{suffix}", cheap2_ranges=r, cheap2_min=1, cheap2_thresh=thresh))
            for r in ("24:26", "25:27", "26:28", "27:29", "28:30"):
                candidates.append(case(f"tier2_top1_{labelize(r)}_{suffix}", cheap2_ranges=r, cheap2_min=1, cheap2_thresh=thresh))

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

        targets = []
        for run in successful[:5]:
            targets.append(run)
        for run in successful:
            if float(run["gen_tps"]) >= 14.0 and run not in targets:
                targets.append(run)

        quality_summaries = [validate_quality(run) for run in targets]

        print("=== TIER2 ROUTERCLAMP PROBE ===", flush=True)
        for run in successful:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<34} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} "
                f"cheap2={cfg.get('env_ser_cheap2_ranges')}:{cfg.get('env_ser_cheap2_min')},{cfg.get('env_ser_cheap2_thresh')} "
                f"faults={run.get('metrics', {}).get('page_faults')}",
                flush=True,
            )

        passed = [
            row for row in quality_summaries
            if row["strict_total"] == len(QUALITY_PROMPTS) and row["strict_passes"] == len(QUALITY_PROMPTS)
        ]
        passed.sort(key=lambda row: float(row["speed_gen_tps"] or 0), reverse=True)
        print("=== STRICT-PASSED TIER2 LEADERS ===", flush=True)
        for row in passed:
            print(
                f"{row['candidate']:<34} gen={row['speed_gen_tps']} pp={row['speed_pp_tps']} "
                f"strict={row['strict_passes']}/{row['strict_total']}",
                flush=True,
            )
        print(
            json.dumps(
                {
                    "event": "tier2_probe_done",
                    "speed_best": summarize(successful[0] if successful else None),
                    "quality_best": passed[0] if passed else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        resume_codex_gpu_helper()


def case(
    label: str,
    *,
    cheap2_ranges: str | None = None,
    cheap2_min: int | None = None,
    cheap2_thresh: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(label=label)
    if cheap2_ranges:
        cfg.update(
            env_ser_cheap2_ranges=cheap2_ranges,
            env_ser_cheap2_min=cheap2_min,
            env_ser_cheap2_thresh=cheap2_thresh,
        )
    cfg.update(overrides)
    return cfg


def validate_quality(speed_run: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(speed_run["config_json"])
    rows = []
    for q in QUALITY_PROMPTS:
        run = run_config(dict(
            cfg,
            label=f"tier2_quality_{q['name']}_{speed_run['label']}",
            source="routerclamp-tier2-quality",
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
            "answer": answer[:260],
            **summarize(run),
        }
        rows.append(row)
        print(json.dumps({"event": "quality", "candidate": speed_run["label"], **row}, sort_keys=True), flush=True)
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


def labelize(ranges: str) -> str:
    return ranges.replace(":", "_").replace(",", "_")


if __name__ == "__main__":
    main()
