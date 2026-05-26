from __future__ import annotations

import json
import sys
from typing import Any

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import stop_codex_gpu_helper, resume_codex_gpu_helper, summarize


BASE = {
    "prompt_key": "mars_fact_list",
    "prompt": "<|im_start|>user\nContinue this comma-separated list of Mars facts: red planet, thin atmosphere,<|im_end|>\n<|im_start|>assistant\n",
    "ctx_size": 16384,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "env_ser_cheap_ranges": "24:30",
    "env_ser_cheap_min": 2,
    "env_ser_cheap_thresh": 1.0,
    "extra_args": ["-np", "2", "-ns", "2", "-pps"],
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-cheap24-ridge",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "cheap24_ridge_start"}), flush=True)
        candidates = [
            (2304, 96),
            (2304, 104),
            (2336, 96),
            (2336, 104),
            (2368, 88),
            (2368, 96),
            (2368, 104),
            (2400, 88),
            (2400, 96),
            (2400, 104),
            (2432, 96),
            (2432, 104),
            (2496, 96),
            (2496, 104),
            (2560, 96),
            (2560, 104),
        ]
        runs = []
        for batch, ubatch in candidates:
            run = run_config(case(batch, ubatch))
            runs.append(run)
            print(json.dumps({"event": "speed", "batch": batch, "ubatch": ubatch, **summarize(run)}, sort_keys=True), flush=True)

        winners = [run for run in runs if run.get("gen_tps") is not None and run.get("exit_code") == 0]
        winners.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
        print("=== CHEAP24 RIDGE ===", flush=True)
        for run in winners:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<28} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg['batch_size']} ub={cfg['ubatch_size']} faults={run.get('metrics', {}).get('page_faults')} "
                f"rss={run.get('peak_rss_bytes')}",
                flush=True,
            )
        print(json.dumps({"event": "cheap24_ridge_done", "best": summarize(winners[0] if winners else None)}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(batch: int, ubatch: int) -> dict[str, Any]:
    return dict(BASE, label=f"cheap24_ridge_b{batch}_ub{ubatch}", batch_size=batch, ubatch_size=ubatch)


if __name__ == "__main__":
    main()
