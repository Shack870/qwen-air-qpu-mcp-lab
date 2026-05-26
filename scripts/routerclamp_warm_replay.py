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
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-warm-replay",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "warm_replay_start"}), flush=True)
        candidates = [
            {"label": "control_np2_b2560_ub96", "batch_size": 2560, "ubatch_size": 96},
            {"label": "control_np2_b2304_ub104", "batch_size": 2304, "ubatch_size": 104},
            {"label": "cheap24_b2336_ub104", "batch_size": 2336, "ubatch_size": 104, "env_ser_cheap_ranges": "24:30", "env_ser_cheap_min": 2, "env_ser_cheap_thresh": 1.0},
            {"label": "cheap24_b2496_ub96", "batch_size": 2496, "ubatch_size": 96, "env_ser_cheap_ranges": "24:30", "env_ser_cheap_min": 2, "env_ser_cheap_thresh": 1.0},
            {"label": "top1_24_25_b2304_ub104", "batch_size": 2304, "ubatch_size": 104, "env_ser_cheap_ranges": "24:25", "env_ser_cheap_min": 1, "env_ser_cheap_thresh": 5.0},
            {"label": "top1_25_26_b2304_ub104", "batch_size": 2304, "ubatch_size": 104, "env_ser_cheap_ranges": "25:26", "env_ser_cheap_min": 1, "env_ser_cheap_thresh": 5.0},
        ]

        speed_runs = []
        for repeat in range(2):
            for cand in candidates:
                cand_cfg = dict(cand)
                cand_label = cand_cfg.pop("label")
                run = run_config(case(f"{cand_label}_r{repeat + 1}", **cand_cfg))
                speed_runs.append(run)
                print(json.dumps({"event": "speed", "repeat": repeat + 1, **summarize(run)}, sort_keys=True), flush=True)

        winners = [run for run in speed_runs if run.get("exit_code") == 0 and run.get("gen_tps") is not None]
        winners.sort(key=lambda run: float(run["gen_tps"]), reverse=True)

        print("=== WARM REPLAY SPEED LEADERS ===", flush=True)
        for run in winners:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<34} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg['batch_size']} ub={cfg['ubatch_size']} cheap={cfg.get('env_ser_cheap_ranges')} "
                f"faults={run.get('metrics', {}).get('page_faults')}",
                flush=True,
            )

        strict_target = winners[0] if winners else None
        if strict_target:
            cfg = json.loads(strict_target["config_json"])
            print(json.dumps({"event": "strict_replay_target", "label": strict_target["label"], "gen_tps": strict_target.get("gen_tps")}), flush=True)
            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"{strict_target['label']}_strict_{q['name']}",
                    batch_size=cfg["batch_size"],
                    ubatch_size=cfg["ubatch_size"],
                    env_ser_cheap_ranges=cfg.get("env_ser_cheap_ranges"),
                    env_ser_cheap_min=cfg.get("env_ser_cheap_min"),
                    env_ser_cheap_thresh=cfg.get("env_ser_cheap_thresh"),
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                    ignore_eos=False,
                ))
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                passed, reason = score_answer(q["name"], answer)
                row = {"quality": q["name"], "passed": passed, "reason": reason, "answer": answer[:220], "gen_tps": run.get("gen_tps")}
                quality_rows.append(row)
                print(json.dumps({"event": "quality", **row, **summarize(run)}, sort_keys=True), flush=True)
                if not passed:
                    break
            print(json.dumps({"event": "warm_replay_strict_summary", "target": strict_target["label"], "passes": sum(1 for row in quality_rows if row["passed"]), "total": len(quality_rows)}), flush=True)

        print(json.dumps({"event": "warm_replay_done", "best": summarize(winners[0] if winners else None)}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(COMMON)
    cfg.update(label=label)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


if __name__ == "__main__":
    main()
