from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from qpu_mcp_lab import db
from qpu_mcp_lab.bench import run_config


ROOT = Path(__file__).resolve().parents[1]

PROMPT_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": PROMPT_CONTINUE,
    "ctx_size": 16384,
    "batch_size": 2304,
    "ubatch_size": 104,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-supernova",
    "timeout_seconds": 900,
}

SER_VALUES = ["3,1", "1,8", "1,7", "1,6", "1,5", "1,4", "1,3", "1,2", "1,1", "2,1", "4,1", "5,1"]

LANES: list[dict[str, Any]] = [
    {"name": "raw_b2304_ub104", "batch_size": 2304, "ubatch_size": 104, "extra_args": []},
    {"name": "raw_b2368_ub96", "batch_size": 2368, "ubatch_size": 96, "extra_args": []},
    {"name": "np2_b2560_ub96", "batch_size": 2560, "ubatch_size": 96, "extra_args": ["-np", "2", "-ns", "2", "-pps"]},
]

def chat(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

QUALITY_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "serbia",
        "prompt": chat("What is the capital of Serbia? Answer with one short sentence."),
        "n_predict": 48,
        "pass_regex": r"\bBelgrade\b",
    },
    {
        "name": "mars",
        "prompt": chat("What is the capital of Mars? Answer factually in one short sentence."),
        "n_predict": 64,
        "pass_regex": r"\b(no|not|none|does not|isn't|is not)\b.{0,80}\b(capital|city|government|country|nation)\b",
    },
    {
        "name": "prime",
        "prompt": chat("Write a compact Python function that checks whether n is prime."),
        "n_predict": 112,
        "pass_regex": r"def\s+\w+.*%.*return",
    },
    {
        "name": "moe",
        "prompt": chat("Explain a mixture-of-experts model in two concise sentences."),
        "n_predict": 80,
        "pass_regex": r"\b(expert|experts)\b.*\b(router|routing|select|selected|activates?)\b",
    },
    {
        "name": "logic",
        "prompt": chat("If all bloops are razzes and no razzes are blue, are any bloops blue? Answer yes or no and why."),
        "n_predict": 64,
        "pass_regex": r"\b(no|not)\b.*\bblue\b",
    },
]


def main() -> None:
    stop_codex_gpu_helper()
    best_before = db.best_runs(limit=1)[0]
    print(json.dumps({"event": "routerclamp_start", "global_best_before": summarize(best_before)}, sort_keys=True), flush=True)

    warm = run_config(case("routerclamp_warmup_discard", LANES[0], "3,1", n_predict=32))
    emit(warm, event="warmup_discard")

    scout_runs: list[dict[str, Any]] = []
    for lane in LANES:
        for ser in SER_VALUES:
            run = run_config(case(f"routerclamp_speed_{lane['name']}_ser{safe_ser(ser)}", lane, ser))
            scout_runs.append(run)
            emit(run)
            if is_success(run) and float(run["gen_tps"]) > float(best_before["gen_tps"]):
                print(json.dumps({"event": "new_global_candidate", **summarize(run)}, sort_keys=True), flush=True)
            maybe_cool(run)

    winners = choose_quality_candidates(scout_runs)
    print(json.dumps({"event": "quality_candidates", "labels": [run["label"] for run in winners]}, sort_keys=True), flush=True)

    quality_rows: list[dict[str, Any]] = []
    for winner in winners:
        cfg = json.loads(winner["config_json"])
        passes = 0
        for q in QUALITY_PROMPTS:
            qcfg = dict(
                cfg,
                label=f"quality_{q['name']}_{winner['label']}",
                prompt=q["prompt"],
                n_predict=q["n_predict"],
                source="routerclamp-quality",
                timeout_seconds=900,
            )
            run = run_config(qcfg)
            ok = quality_pass(run, q["pass_regex"])
            passes += int(ok)
            row = {"candidate": winner["label"], "quality": q["name"], "passed": ok, **summarize(run)}
            quality_rows.append(row)
            print(json.dumps({"event": "quality_run", **row}, sort_keys=True), flush=True)
            maybe_cool(run)
        print(
            json.dumps(
                {
                    "event": "quality_summary",
                    "candidate": winner["label"],
                    "passes": passes,
                    "total": len(QUALITY_PROMPTS),
                    "speed_gen_tps": winner.get("gen_tps"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "routerclamp_done",
                "speed_best": summarize(best_success(scout_runs)),
                "global_best_after": summarize(db.best_runs(limit=1)[0]),
                "quality_pass_rows": sum(1 for row in quality_rows if row["passed"]),
                "quality_total_rows": len(quality_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resume_codex_gpu_helper()


def case(label: str, lane: dict[str, Any], ser: str, **overrides: Any) -> dict[str, Any]:
    return dict(
        BASE,
        label=label,
        batch_size=lane["batch_size"],
        ubatch_size=lane["ubatch_size"],
        extra_args=list(lane["extra_args"]),
        smart_expert_reduction=ser,
        **overrides,
    )


def safe_ser(ser: str) -> str:
    return ser.replace(",", "_").replace(".", "p")


def choose_quality_candidates(runs: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    successful = [run for run in runs if is_success(run)]
    successful.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
    chosen: list[dict[str, Any]] = []
    seen_ser: set[str] = set()
    for run in successful:
        cfg = json.loads(run["config_json"])
        ser = cfg.get("smart_expert_reduction")
        if ser in seen_ser and len(chosen) >= 2:
            continue
        chosen.append(run)
        seen_ser.add(str(ser))
        if len(chosen) >= limit:
            break
    baseline = next((run for run in successful if json.loads(run["config_json"]).get("smart_expert_reduction") == "3,1"), None)
    if baseline and all(run["id"] != baseline["id"] for run in chosen):
        chosen.append(baseline)
    return chosen


def quality_pass(run: dict[str, Any], pattern: str) -> bool:
    log_path = run.get("log_path")
    if not log_path:
        return False
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    answer = generated_answer(text)
    return bool(re.search(pattern, answer, flags=re.I | re.S))


def generated_answer(text: str) -> str:
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    return text.strip()


def emit(run: dict[str, Any] | None, event: str = "run") -> None:
    print(json.dumps({"event": event, **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "id": run.get("id"),
        "label": run.get("label"),
        "source": run.get("source"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "metrics": run.get("metrics"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def is_success(run: dict[str, Any] | None) -> bool:
    return bool(run and run.get("exit_code") == 0 and run.get("gen_tps") is not None)


def best_success(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [run for run in runs if is_success(run)]
    if not successful:
        return None
    return max(successful, key=lambda run: float(run["gen_tps"]))


def maybe_cool(run: dict[str, Any] | None) -> None:
    if not run:
        return
    metrics = run.get("metrics") or {}
    invol = int(metrics.get("involuntary_context_switches") or 0)
    gen_tps = float(run.get("gen_tps") or 0.0)
    page_faults = int(metrics.get("page_faults") or 0)
    if invol > 140000 or page_faults > 50000 or (gen_tps and gen_tps < 8.0):
        print(
            json.dumps(
                {
                    "event": "cooldown",
                    "seconds": 75,
                    "gen_tps": gen_tps,
                    "involuntary_context_switches": invol,
                    "page_faults": page_faults,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(75)


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


def resume_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,stat,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ && $2 ~ /^T/ {print $1}' | xargs -r kill -CONT"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        resume_codex_gpu_helper()
        raise
