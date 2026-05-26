from __future__ import annotations

import json
import subprocess
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
from routerclamp_strict_quality_scout import resume_codex_gpu_helper, stop_codex_gpu_helper, summarize


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
    "batch_size": 2496,
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
    "source": "record-recovery-push",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "record_recovery_start", "thermal": thermal_state()}), flush=True)
        for seconds in (60, 60):
            print(json.dumps({"event": "cooldown", "seconds": seconds, "thermal": thermal_state()}), flush=True)
            time.sleep(seconds)
        print(json.dumps({"event": "cooldown_done", "thermal": thermal_state()}), flush=True)

        cases = [
            case("recovery_b2496_ub128_r1"),
            case("recovery_b2496_ub128_r2"),
            case("recovery_b2560_ub128_r1", batch_size=2560),
            case("recovery_b2304_ub144_r1", batch_size=2304, ubatch_size=144),
            case("recovery_b2496_ub136_r1", ubatch_size=136),
            case("recovery_b2528_ub128_r1", batch_size=2528),
        ]
        runs = []
        for cfg in cases:
            run = run_config(cfg)
            runs.append(run)
            print(json.dumps({"event": "run", "thermal": thermal_state(), **summarize(run)}, sort_keys=True), flush=True)
            time.sleep(15)

        leaders = [run for run in runs if run.get("exit_code") == 0 and run.get("gen_tps") is not None]
        leaders.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
        print("=== RECORD RECOVERY PUSH ===", flush=True)
        for run in leaders:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<28} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} "
                f"faults={run.get('metrics', {}).get('page_faults')} wall={run.get('metrics', {}).get('wall_seconds')}",
                flush=True,
            )
        print(json.dumps({"event": "record_recovery_done", "best": summarize(leaders[0] if leaders else None)}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(label=label)
    cfg.update(overrides)
    return cfg


def thermal_state() -> dict[str, int | None]:
    out = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.xcpm.cpu_thermal_level", "machdep.xcpm.io_thermal_level"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    vals = []
    for line in out.stdout.splitlines():
        try:
            vals.append(int(line.strip()))
        except ValueError:
            vals.append(None)
    return {"cpu": vals[0] if vals else None, "io": vals[1] if len(vals) > 1 else None}


if __name__ == "__main__":
    main()
