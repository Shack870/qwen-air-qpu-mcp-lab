from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import (
    QUALITY_PROMPTS,
    generated_answer,
    resume_codex_gpu_helper,
    score_answer,
    stop_codex_gpu_helper,
    summarize,
)

RECORD_TPS = 14.03
THERMAL_GATE = 110


def chat(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": chat("Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"),
    "ctx_size": 16384,
    "batch_size": 2304,
    "ubatch_size": 104,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "1,5",
    "n_predict": 128,
    "temp": 0.0,
    "ignore_eos": True,
    "no_display_prompt": True,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routercensus-rescue-drive",
    "timeout_seconds": 900,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print_event("routercensus_rescue_start", thermal=thermal())
        warm = run_config(make_case("rescue_warmup_record_shape", "3,1", 2456, 144, rescue_a=("24:30", 2, 1.0), n_predict=32))
        print_event("warmup", **summarize(warm), thermal=thermal())
        time.sleep(8)

        runs: list[dict[str, Any]] = []
        for cfg in candidates():
            wait_for_thermal(THERMAL_GATE, f"cooldown_{cfg['label']}")
            run = run_config(cfg)
            runs.append(run)
            print_event("speed", **summarize(run), thermal=thermal(), answer=generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))[:220])
            time.sleep(6)

        leaders = [run for run in runs if run.get("exit_code") == 0 and run.get("gen_tps") is not None]
        leaders.sort(key=lambda r: float(r["gen_tps"]), reverse=True)
        print("=== ROUTERCENSUS RESCUE SCOREBOARD ===", flush=True)
        for run in leaders:
            cfg = json.loads(run["config_json"])
            print(
                f"{run['label']:<36} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"ser={cfg.get('smart_expert_reduction')} b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} "
                f"a={cfg.get('env_ser_cheap_ranges')}:{cfg.get('env_ser_cheap_min')},{cfg.get('env_ser_cheap_thresh')} "
                f"b={cfg.get('env_ser_cheap2_ranges')}:{cfg.get('env_ser_cheap2_min')},{cfg.get('env_ser_cheap2_thresh')} "
                f"faults={run.get('metrics', {}).get('page_faults')} ctxsw={run.get('metrics', {}).get('involuntary_context_switches')}",
                flush=True,
            )

        strict_candidates = [run for run in leaders if float(run["gen_tps"]) > RECORD_TPS]
        for run in strict_candidates[:4]:
            rows = strict_quality(run)
            all_passed = len(rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in rows)
            print_event("strict_done", candidate=run["label"], all_passed=all_passed, rows=rows)
            if all_passed:
                crown(run, rows)
                break

        print_event("routercensus_rescue_done", best=summarize(leaders[0] if leaders else None), thermal=thermal())
    finally:
        resume_codex_gpu_helper()


def candidates() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Control: the quality-safe record family.
    cases.append(make_case("control_record_family_b2456_ub144", "3,1", 2456, 144, rescue_a=("24:30", 2, 1.0)))

    # Inverted Routerclamp: global top-1 fast path, then rescue fragile
    # census bands back toward top-3/top-2.
    rescue_shapes = [
        ("top1_rescue_mid24_30", ("24:30", 3, 1.0), None),
        ("top1_rescue_mid24_30_top2late40_48", ("24:30", 3, 1.0), ("40:48", 2, 1.0)),
        ("top1_rescue_mid24_30_top2early0_6", ("24:30", 3, 1.0), ("0:6", 2, 1.0)),
        ("top1_rescue_diffuse6_21_mid24_30", ("6:21,24:30", 3, 1.0), None),
        ("top1_rescue_early0_6_mid24_30_late40_48", ("0:6,24:30,40:48", 3, 1.0), None),
        ("top1_rescue_mid24_30_late36_48", ("24:30,36:48", 3, 1.0), None),
        ("top1_rescue_mid24_30_late42_48", ("24:30,42:48", 3, 1.0), None),
        ("top1_rescue_mid24_30_late47", ("24:30,47:48", 3, 1.0), None),
    ]
    for label, rescue_a, rescue_b in rescue_shapes:
        cases.append(make_case(label, "1,5", 2304, 104, rescue_a=rescue_a, rescue_b=rescue_b))
        cases.append(make_case(label + "_b2456_ub144", "1,5", 2456, 144, rescue_a=rescue_a, rescue_b=rescue_b))

    return cases


def make_case(
    label: str,
    ser: str,
    batch: int,
    ubatch: int,
    *,
    rescue_a: tuple[str, int, float] | None = None,
    rescue_b: tuple[str, int, float] | None = None,
    n_predict: int = 128,
    **overrides: Any,
) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(label=label, smart_expert_reduction=ser, batch_size=batch, ubatch_size=ubatch, n_predict=n_predict)
    if rescue_a:
        cfg.update(env_ser_cheap_ranges=rescue_a[0], env_ser_cheap_min=rescue_a[1], env_ser_cheap_thresh=rescue_a[2])
    if rescue_b:
        cfg.update(env_ser_cheap2_ranges=rescue_b[0], env_ser_cheap2_min=rescue_b[1], env_ser_cheap2_thresh=rescue_b[2])
    cfg.update(overrides)
    return cfg


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "routercensus-rescue-strict", "ignore_eos": False, "no_display_prompt": True})
    rows = []
    for prompt in QUALITY_PROMPTS:
        wait_for_thermal(THERMAL_GATE, f"strict_cooldown_{speed_run['label']}_{prompt['name']}")
        run = run_config(
            dict(
                cfg,
                label=f"{speed_run['label']}_strict_{prompt['name']}",
                prompt=prompt["prompt"],
                n_predict=prompt["n_predict"],
            )
        )
        answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
        passed, reason = score_answer(prompt["name"], answer)
        row = {
            "quality": prompt["name"],
            "passed": passed,
            "reason": reason,
            "answer": answer[:360],
            **summarize(run),
        }
        rows.append(row)
        print_event("strict", candidate=speed_run["label"], **row, thermal=thermal())
        if not passed:
            break
    return rows


def crown(speed_run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    with db.connect() as con:
        con.execute(
            """
            UPDATE runs
            SET quality_flag = 'strict_passed',
                notes = COALESCE(notes || '; ', '') || ?
            WHERE id = ?
            """,
            (
                "strict quality passed by routercensus_rescue_drive: "
                + ",".join(row["quality"] for row in rows),
                int(speed_run["id"]),
            ),
        )
    print_event("crowned", **summarize(speed_run))


def thermal() -> int | None:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"], text=True, timeout=5).strip())
    except Exception:
        return None


def wait_for_thermal(max_level: int, event: str) -> None:
    while True:
        level = thermal()
        if level is None or level <= max_level:
            return
        print_event(event, thermal=level, max_level=max_level)
        time.sleep(15)


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
