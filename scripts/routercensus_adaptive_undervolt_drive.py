from __future__ import annotations

import json
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


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print_event("adaptive_undervolt_start")
        warmup = run_config(case("adaptive_warmup_discard", n_predict=32))
        print_event("warmup", **summarize(warmup))
        time.sleep(10)

        summaries: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for label, overrides in candidates():
            run = run_config(case(label, **overrides))
            print_event("speed", **summarize(run), config=interesting_config(run))

            speed = float(run.get("gen_tps") or 0.0)
            rows: list[dict[str, Any]] = []
            if speed >= 12.0:
                rows = quick_quality(run, strict_prime=speed > CURRENT_RECORD_TPS)

            summary = {
                "label": label,
                "gen_tps": run.get("gen_tps"),
                "pp_tps": run.get("pp_tps"),
                "quick_passes": sum(1 for row in rows if row["passed"]),
                "quick_total": len(rows),
                "failed": next((row["quality"] for row in rows if not row["passed"]), None),
                "quality_rows": rows,
                "config": interesting_config(run),
            }
            summaries.append(summary)
            print_event("candidate_summary", **summary)

            if rows and all(row["passed"] for row in rows) and speed > float(best.get("gen_tps") or 0.0) if best else rows and all(row["passed"] for row in rows):
                best = run
            time.sleep(8)

        print("=== ADAPTIVE UNDERVOLT SCOREBOARD ===", flush=True)
        for row in sorted(summaries, key=lambda item: (item["quick_passes"], item["gen_tps"] or 0), reverse=True):
            print(
                f"{row['label']:<48} pass={row['quick_passes']}/{row['quick_total']} "
                f"gen={row['gen_tps']} pp={row['pp_tps']} failed={row['failed']} "
                f"cheap={row['config'].get('cheap_ranges')} adapt={row['config'].get('adaptive_ranges')}@{row['config'].get('adaptive_ratio')}",
                flush=True,
            )

        if best and float(best.get("gen_tps") or 0.0) > CURRENT_RECORD_TPS:
            rows = strict_quality(best)
            print_event(
                "strict_done",
                candidate=best.get("label"),
                all_passed=len(rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in rows),
                rows=rows,
            )
        print_event("adaptive_undervolt_done", summaries=summaries)
    finally:
        resume_codex_gpu_helper()


def candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("control_record_static_24_30", {}),
        ("adaptive_record_24_30_r085", adaptive("24:30", 0.85)),
        ("adaptive_record_24_30_r090", adaptive("24:30", 0.90)),
        ("record_plus_adapt_6_10_r085", record_plus_adapt("6:10", 0.85)),
        ("record_plus_adapt_30_34_r085", record_plus_adapt("30:34", 0.85)),
        ("record_plus_adapt_6_10_30_34_r085", record_plus_adapt("6:10,30:34", 0.85)),
        ("record_plus_adapt_6_10_18_22_30_34_r085", record_plus_adapt("6:10,18:22,30:34", 0.85)),
        ("record_plus_adapt_6_10_30_34_r090", record_plus_adapt("6:10,30:34", 0.90)),
        ("record_plus_adapt_6_10_18_22_30_34_r090", record_plus_adapt("6:10,18:22,30:34", 0.90)),
    ]


def adaptive(ranges: str, ratio: float) -> dict[str, Any]:
    return {
        "env_ser_cheap_ranges": ranges,
        "env_ser_cheap_min": 2,
        "env_ser_cheap_thresh": 1.0,
        "env_ser_adaptive_ranges": ranges,
        "env_ser_adaptive_third_ratio": ratio,
    }


def record_plus_adapt(extra_ranges: str, ratio: float) -> dict[str, Any]:
    return {
        "env_ser_cheap2_ranges": "24:30",
        "env_ser_cheap2_min": 2,
        "env_ser_cheap2_thresh": 1.0,
        "env_ser_cheap_ranges": extra_ranges,
        "env_ser_cheap_min": 2,
        "env_ser_cheap_thresh": 1.0,
        "env_ser_adaptive_ranges": extra_ranges,
        "env_ser_adaptive_third_ratio": ratio,
    }


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
            "source": "routercensus-adaptive-undervolt",
            "timeout_seconds": 900,
        }
    )
    cfg.update(overrides)
    return cfg


def quick_quality(speed_run: dict[str, Any], *, strict_prime: bool) -> list[dict[str, Any]]:
    prompts = QUALITY_PROMPTS if strict_prime else QUALITY_PROMPTS[:2]
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "routercensus-adaptive-undervolt-quality", "ignore_eos": False, "no_display_prompt": True})
    rows = []
    for prompt in prompts:
        run = run_config(
            dict(
                cfg,
                label=f"{speed_run['label']}_quality_{prompt['name']}",
                prompt=prompt["prompt"],
                n_predict=prompt["n_predict"],
            )
        )
        answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
        passed, reason = score_answer(prompt["name"], answer)
        row = {"quality": prompt["name"], "passed": passed, "reason": reason, "answer": answer[:340], **summarize(run)}
        rows.append(row)
        print_event("quality", candidate=speed_run.get("label"), **row)
        if not passed:
            break
    return rows


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "routercensus-adaptive-undervolt-strict", "ignore_eos": False, "no_display_prompt": True})
    rows = []
    for prompt in QUALITY_PROMPTS:
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
        row = {"quality": prompt["name"], "passed": passed, "reason": reason, "answer": answer[:440], **summarize(run)}
        rows.append(row)
        print_event("strict_quality", candidate=speed_run.get("label"), **row)
        if not passed:
            break
    return rows


def interesting_config(run: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(run["config_json"])
    return {
        "ser": cfg.get("smart_expert_reduction"),
        "cheap_ranges": cfg.get("env_ser_cheap_ranges"),
        "cheap2_ranges": cfg.get("env_ser_cheap2_ranges"),
        "adaptive_ranges": cfg.get("env_ser_adaptive_ranges"),
        "adaptive_ratio": cfg.get("env_ser_adaptive_third_ratio"),
        "b": cfg.get("batch_size"),
        "ub": cfg.get("ubatch_size"),
    }


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
