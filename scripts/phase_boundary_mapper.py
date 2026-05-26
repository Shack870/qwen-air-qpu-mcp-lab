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

from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import (
    BASE,
    QUALITY_PROMPTS,
    generated_answer,
    resume_codex_gpu_helper,
    score_answer,
    stop_codex_gpu_helper,
    summarize,
)

CURRENT_RECORD_TPS = 14.03
THERMAL_SPEED_GATE = 60
THERMAL_QUALITY_GATE = 80


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print_event("phase_boundary_start", thermal=thermal())
        wait_for_thermal(THERMAL_SPEED_GATE, "initial_cooldown")

        warmup = run_config(case("phase_warmup_discard", n_predict=32))
        print_event("warmup", **summarize(warmup), thermal_after=thermal())
        time.sleep(20)

        best: dict[str, Any] | None = None
        summaries = []
        for candidate in candidates():
            wait_for_thermal(THERMAL_SPEED_GATE, "speed_cooldown")
            run = run_config(candidate)
            speed_row = summarize(run)
            speed = float(run.get("gen_tps") or 0.0)
            print_event("speed", **speed_row, thermal_after=thermal())

            quality_rows: list[dict[str, Any]] = []
            if speed >= 12.0:
                quality_rows = quick_quality(run, full_code_check=speed > CURRENT_RECORD_TPS)

            passed_quick = len(quality_rows) >= 2 and all(row["passed"] for row in quality_rows[:2])
            candidate_summary = {
                "label": run.get("label"),
                "speed_gen_tps": run.get("gen_tps"),
                "speed_pp_tps": run.get("pp_tps"),
                "quick_quality_passed": passed_quick,
                "quality_rows": quality_rows,
            }
            summaries.append(candidate_summary)
            print_event("candidate_summary", **candidate_summary)

            if passed_quick and speed > float(best.get("gen_tps") or 0.0) if best else passed_quick:
                best = run
            time.sleep(15)

        if best and float(best.get("gen_tps") or 0.0) > CURRENT_RECORD_TPS:
            print_event("strict_candidate", **summarize(best), thermal=thermal())
            strict_rows = strict_quality(best)
            print_event(
                "strict_done",
                all_passed=len(strict_rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in strict_rows),
                rows=strict_rows,
            )
        else:
            print_event("phase_boundary_done", best=summarize(best) if best else None, summaries=summaries)
    finally:
        resume_codex_gpu_helper()


def candidates() -> list[dict[str, Any]]:
    """Boundary probes, not random knobs.

    The first group compresses only part of the known-good CHEAP24 band to a
    top-1 floor. The second group starts from the fast-bad ser1 lane and adds
    back full routing on suspected fragile layer bands.
    """

    specs: list[tuple[str, dict[str, Any]]] = [
        ("safe_control_b2456_ub144", {}),
        ("safe_ridge_b2460_ub148", {"batch_size": 2460, "ubatch_size": 148}),
        ("top1_band_24_26", {"env_ser_cheap_ranges": "24:26", "env_ser_cheap_min": 1}),
        ("top1_band_26_28", {"env_ser_cheap_ranges": "26:28", "env_ser_cheap_min": 1}),
        ("top1_band_28_30", {"env_ser_cheap_ranges": "28:30", "env_ser_cheap_min": 1}),
        ("top1_band_24_28", {"env_ser_cheap_ranges": "24:28", "env_ser_cheap_min": 1}),
        ("top1_band_26_30", {"env_ser_cheap_ranges": "26:30", "env_ser_cheap_min": 1}),
        ("top1_band_24_30", {"env_ser_cheap_ranges": "24:30", "env_ser_cheap_min": 1}),
        (
            "badfast_ser1_5_rescue_24_30",
            {"smart_expert_reduction": "1,5", "env_ser_cheap_ranges": None, "env_ser_full_ranges": "24:30"},
        ),
        (
            "badfast_ser1_5_rescue_18_24",
            {"smart_expert_reduction": "1,5", "env_ser_cheap_ranges": None, "env_ser_full_ranges": "18:24"},
        ),
        (
            "badfast_ser1_5_rescue_30_36",
            {"smart_expert_reduction": "1,5", "env_ser_cheap_ranges": None, "env_ser_full_ranges": "30:36"},
        ),
        (
            "badfast_ser1_5_rescue_18_24_30_36",
            {"smart_expert_reduction": "1,5", "env_ser_cheap_ranges": None, "env_ser_full_ranges": "18:24,30:36"},
        ),
    ]
    return [case(label, **overrides) for label, overrides in specs]


def case(label: str, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(
        {
            "label": label,
            "ctx_size": 16384,
            "batch_size": 2456,
            "ubatch_size": 144,
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
            "ignore_eos": True,
            "no_display_prompt": True,
            "source": "phase-boundary-mapper",
            "timeout_seconds": 900,
        }
    )
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def quick_quality(speed_run: dict[str, Any], full_code_check: bool) -> list[dict[str, Any]]:
    prompts = QUALITY_PROMPTS if full_code_check else QUALITY_PROMPTS[:2]
    cfg = json.loads(speed_run["config_json"])
    cfg.update(
        {
            "source": "phase-boundary-mapper-quality",
            "ignore_eos": False,
            "no_display_prompt": True,
            "timeout_seconds": 900,
        }
    )
    rows = []
    for prompt in prompts:
        wait_for_thermal(THERMAL_QUALITY_GATE, f"quick_quality_cooldown_{prompt['name']}")
        run = run_config(
            dict(
                cfg,
                label=f"{speed_run['label']}_quick_{prompt['name']}",
                prompt=prompt["prompt"],
                n_predict=prompt["n_predict"],
            )
        )
        row = quality_row(prompt["name"], run)
        rows.append(row)
        print_event("quick_quality", candidate=speed_run.get("label"), **row)
        if not row["passed"]:
            break
    return rows


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "phase-boundary-mapper-strict", "ignore_eos": False, "no_display_prompt": True})
    rows = []
    for prompt in QUALITY_PROMPTS:
        wait_for_thermal(THERMAL_QUALITY_GATE, f"strict_quality_cooldown_{prompt['name']}")
        run = run_config(
            dict(
                cfg,
                label=f"{speed_run['label']}_strict_{prompt['name']}",
                prompt=prompt["prompt"],
                n_predict=prompt["n_predict"],
            )
        )
        row = quality_row(prompt["name"], run)
        rows.append(row)
        print_event("strict_quality", candidate=speed_run.get("label"), **row)
        if not row["passed"]:
            break
    return rows


def quality_row(name: str, run: dict[str, Any]) -> dict[str, Any]:
    answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
    passed, reason = score_answer(name, answer)
    return {
        "quality": name,
        "passed": passed,
        "reason": reason,
        "answer": answer[:400],
        "thermal_after": thermal(),
        **summarize(run),
    }


def thermal() -> int | None:
    try:
        return int(
            subprocess.check_output(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                text=True,
                timeout=5,
            ).strip()
        )
    except Exception:
        return None


def wait_for_thermal(max_level: int, event: str) -> None:
    while True:
        level = thermal()
        if level is None or level <= max_level:
            return
        print_event(event, thermal=level, max_level=max_level)
        time.sleep(30)


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
