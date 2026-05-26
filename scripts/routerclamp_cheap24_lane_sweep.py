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
        print(json.dumps({"event": "cheap24_lane_sweep_start"}), flush=True)
        candidates = [
            case("lane_baseline_np2_b2560_ub96", 2560, 96, None, None, None, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_np2_b2560_ub96", 2560, 96, "24:30", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_raw_b2304_ub104", 2304, 104, "24:30", 2, 1.0, []),
            case("lane_cheap24_raw_b2368_ub96", 2368, 96, "24:30", 2, 1.0, []),
            case("lane_cheap24_np2_b2304_ub104", 2304, 104, "24:30", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_np2_b2368_ub96", 2368, 96, "24:30", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_np2_b2560_ub104", 2560, 104, "24:30", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_np2_b2688_ub96", 2688, 96, "24:30", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_42_np2_b2560_ub96", 2560, 96, "24:30,42:48", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
            case("lane_cheap24_36_np2_b2560_ub96", 2560, 96, "24:30,36:42", 2, 1.0, ["-np", "2", "-ns", "2", "-pps"]),
        ]

        runs = []
        for cfg in candidates:
            run = run_config(cfg)
            runs.append(run)
            print(json.dumps({"event": "speed", **summarize(run)}, sort_keys=True), flush=True)

        successful = [run for run in runs if run.get("exit_code") == 0 and run.get("gen_tps") is not None]
        successful.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
        winners = successful[:3]
        for winner in winners:
            cfg = json.loads(winner["config_json"])
            for q in QUALITY_PROMPTS:
                run = run_config(dict(
                    cfg,
                    label=f"lane_quality_{q['name']}_{winner['label']}",
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                    ignore_eos=False,
                    source="routerclamp-cheap24-lane-quality",
                ))
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                passed, reason = score_answer(q["name"], answer)
                print(json.dumps({
                    "event": "quality",
                    "candidate": winner["label"],
                    "quality": q["name"],
                    "passed": passed,
                    "reason": reason,
                    "answer": answer[:240],
                    **summarize(run),
                }, sort_keys=True), flush=True)
                if not passed:
                    break

        print("=== CHEAP24 LANE SWEEP ===", flush=True)
        for run in successful:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<38} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} args={cfg.get('extra_args')} "
                f"ranges={cfg.get('env_ser_cheap_ranges')} faults={run.get('metrics', {}).get('page_faults')}",
                flush=True,
            )
        print(json.dumps({"event": "cheap24_lane_sweep_done"}), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, batch: int, ubatch: int, ranges: str | None, cheap_min: int | None, cheap_thresh: float | None, extra_args: list[str]) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(
        label=label,
        source="routerclamp-cheap24-lane-sweep",
        batch_size=batch,
        ubatch_size=ubatch,
        n_predict=128,
        timeout_seconds=900,
        extra_args=extra_args,
        smart_expert_reduction="3,1",
    )
    if ranges:
        cfg.update(
            env_ser_cheap_ranges=ranges,
            env_ser_cheap_min=cheap_min,
            env_ser_cheap_thresh=cheap_thresh,
        )
    return cfg


if __name__ == "__main__":
    main()
