from __future__ import annotations

import json
import sys
import time
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
        print(json.dumps({"event": "cheap24_replay_start"}), flush=True)
        runs = []
        sequence = [
            ("warmup_baseline_n32", None, None, None, 32),
            ("baseline_a", None, None, None, 128),
            ("cheap2_24_30_a", "24:30", 2, 1.0, 128),
            ("baseline_b", None, None, None, 128),
            ("cheap2_24_30_b", "24:30", 2, 1.0, 128),
            ("cheap2_24_30_c", "24:30", 2, 1.0, 128),
        ]
        for label, ranges, cheap_min, cheap_thresh, n_predict in sequence:
            run = run_config(case(f"cheap24_replay_{label}", ranges, cheap_min, cheap_thresh, n_predict=n_predict))
            runs.append(run)
            print(json.dumps({"event": "speed", "candidate": label, "ranges": ranges, "cheap_min": cheap_min, "cheap_thresh": cheap_thresh, **summarize(run)}, sort_keys=True), flush=True)
            if run.get("metrics", {}).get("swaps", 0):
                time.sleep(30)

        for q in QUALITY_PROMPTS:
            run = run_config(case(
                f"cheap24_quality_{q['name']}",
                "24:30",
                2,
                1.0,
                prompt=q["prompt"],
                n_predict=q["n_predict"],
                ignore_eos=False,
            ))
            answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
            passed, reason = score_answer(q["name"], answer)
            print(json.dumps({
                "event": "quality",
                "candidate": "cheap2_24_30",
                "quality": q["name"],
                "passed": passed,
                "reason": reason,
                "answer": answer[:240],
                **summarize(run),
            }, sort_keys=True), flush=True)

        print("=== CHEAP24 REPLAY ===", flush=True)
        for run in runs:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<30} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"ranges={cfg.get('env_ser_cheap_ranges')} cheap={cfg.get('env_ser_cheap_min')},{cfg.get('env_ser_cheap_thresh')} "
                f"faults={run.get('metrics', {}).get('page_faults')} swaps={run.get('metrics', {}).get('swaps')}",
                flush=True,
            )
        print(json.dumps({"event": "cheap24_replay_done"}), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ranges: str | None, cheap_min: int | None, cheap_thresh: float | None, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(RECORD)
    cfg.update(
        label=label,
        source="routerclamp-cheap24-replay",
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
