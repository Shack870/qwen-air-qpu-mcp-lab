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
        print_event("protective_subnet_start")
        warmup = run_config(case("protective_warmup_discard", n_predict=32))
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

        print("=== PROTECTIVE SUBNET SCOREBOARD ===", flush=True)
        for row in sorted(summaries, key=lambda item: (item["quick_passes"], item["gen_tps"] or 0), reverse=True):
            print(
                f"{row['label']:<44} pass={row['quick_passes']}/{row['quick_total']} "
                f"gen={row['gen_tps']} pp={row['pp_tps']} failed={row['failed']} "
                f"cheap={row['config'].get('cheap_ranges')}",
                flush=True,
            )

        if best and float(best.get("gen_tps") or 0.0) > CURRENT_RECORD_TPS:
            strict_rows = strict_quality(best)
            print_event(
                "strict_done",
                candidate=best.get("label"),
                all_passed=len(strict_rows) == len(QUALITY_PROMPTS) and all(row["passed"] for row in strict_rows),
                rows=strict_rows,
            )
        print_event("protective_subnet_done", summaries=summaries)
    finally:
        resume_codex_gpu_helper()


def candidates() -> list[tuple[str, dict[str, Any]]]:
    fragile = [(1, 2), (11, 14), (22, 26), (36, 39)]
    fragile_plus_late = [(1, 2), (11, 14), (22, 26), (36, 43)]
    fragile_plus_edges = [(0, 2), (11, 14), (22, 26), (36, 43), (46, 48)]
    return [
        ("control_record_family_b2456_ub144", {}),
        ("global_top2_except_fragile", {"env_ser_cheap_ranges": complement_ranges(fragile)}),
        ("global_top2_except_fragile_late", {"env_ser_cheap_ranges": complement_ranges(fragile_plus_late)}),
        ("global_top2_except_fragile_edges", {"env_ser_cheap_ranges": complement_ranges(fragile_plus_edges)}),
        ("midlate_top2_except_fragile", {"env_ser_cheap_ranges": intersect_ranges(complement_ranges(fragile), "6:48")}),
        ("midlate_top2_except_fragile_late", {"env_ser_cheap_ranges": intersect_ranges(complement_ranges(fragile_plus_late), "6:48")}),
        (
            "decode_only_global_except_fragile",
            {"env_ser_cheap_ranges": complement_ranges(fragile), "env_ser_cheap_max_ntokens": 1},
        ),
        (
            "broad_with_record_lane_boost",
            {
                "env_ser_cheap_ranges": complement_ranges(fragile_plus_late),
                "env_ser_cheap2_ranges": "24:30",
                "env_ser_cheap2_min": 2,
                "env_ser_cheap2_thresh": 1.0,
            },
        ),
    ]


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
            "source": "routercensus-protective-subnet",
            "timeout_seconds": 900,
        }
    )
    cfg.update(overrides)
    return cfg


def quick_quality(speed_run: dict[str, Any], *, strict_prime: bool) -> list[dict[str, Any]]:
    prompts = QUALITY_PROMPTS if strict_prime else QUALITY_PROMPTS[:2]
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "routercensus-protective-subnet-quality", "ignore_eos": False, "no_display_prompt": True})
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
        row = {
            "quality": prompt["name"],
            "passed": passed,
            "reason": reason,
            "answer": answer[:320],
            **summarize(run),
        }
        rows.append(row)
        print_event("quality", candidate=speed_run.get("label"), **row)
        if not passed:
            break
    return rows


def strict_quality(speed_run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = json.loads(speed_run["config_json"])
    cfg.update({"source": "routercensus-protective-subnet-strict", "ignore_eos": False, "no_display_prompt": True})
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
        row = {
            "quality": prompt["name"],
            "passed": passed,
            "reason": reason,
            "answer": answer[:420],
            **summarize(run),
        }
        rows.append(row)
        print_event("strict_quality", candidate=speed_run.get("label"), **row)
        if not passed:
            break
    return rows


def complement_ranges(exclusions: list[tuple[int, int]], *, n_layers: int = 48) -> str:
    merged = merge_ranges(exclusions)
    ranges = []
    start = 0
    for first, last in merged:
        if start < first:
            ranges.append((start, first))
        start = max(start, last)
    if start < n_layers:
        ranges.append((start, n_layers))
    return render_ranges(ranges)


def intersect_ranges(left: str, right: str) -> str:
    a = parse_ranges(left)
    b = parse_ranges(right)
    out = []
    for af, al in a:
        for bf, bl in b:
            first = max(af, bf)
            last = min(al, bl)
            if first < last:
                out.append((first, last))
    return render_ranges(merge_ranges(out))


def parse_ranges(value: str) -> list[tuple[int, int]]:
    out = []
    for part in value.split(","):
        first, last = part.split(":", 1)
        out.append((int(first), int(last)))
    return out


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for first, last in sorted(ranges):
        if not merged or first > merged[-1][1]:
            merged.append((first, last))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
    return merged


def render_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{first}:{last}" for first, last in ranges)


def interesting_config(run: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(run["config_json"])
    return {
        "ser": cfg.get("smart_expert_reduction"),
        "cheap_ranges": cfg.get("env_ser_cheap_ranges"),
        "cheap_min": cfg.get("env_ser_cheap_min"),
        "cheap_thresh": cfg.get("env_ser_cheap_thresh"),
        "cheap_max_ntokens": cfg.get("env_ser_cheap_max_ntokens"),
        "cheap2_ranges": cfg.get("env_ser_cheap2_ranges"),
        "b": cfg.get("batch_size"),
        "ub": cfg.get("ubatch_size"),
    }


def print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
