from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from qpu_mcp_lab.bench import run_config


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
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "routerclamp-safe-edge",
    "timeout_seconds": 900,
}

QUALITY_PROMPTS: list[dict[str, Any]] = [
    {"name": "serbia", "prompt": chat("What is the capital of Serbia? Answer with one short sentence."), "n_predict": 48, "pass_regex": r"\bBelgrade\b"},
    {"name": "mars", "prompt": chat("What is the capital of Mars? Answer factually in one short sentence."), "n_predict": 64, "pass_regex": r"\b(no|not|none|does not|isn't|is not)\b.{0,100}\b(capital|city|government|country|nation|political)\b"},
    {"name": "prime", "prompt": chat("Write a compact Python function that checks whether n is prime."), "n_predict": 128, "pass_regex": r"def\s+\w+.*(%|sqrt|range).*return"},
    {"name": "moe", "prompt": chat("Explain a mixture-of-experts model in two concise sentences."), "n_predict": 96, "pass_regex": r"\b(expert|experts)\b.*\b(router|routing|select|selected|activates?|gate|gating)\b"},
    {"name": "logic", "prompt": chat("If all bloops are razzes and no razzes are blue, are any bloops blue? Answer yes or no and why."), "n_predict": 80, "pass_regex": r"\b(no|not)\b.*\bblue\b"},
]


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "safe_edge_start"}), flush=True)
    candidates = [
        ("baseline_ser3_1", "3,1"),
        ("safe_ser1_8", "1,8"),
        ("safe_ser1_7", "1,7"),
        ("safe_ser1_6", "1,6"),
        ("unsafe_reference_ser1_5", "1,5"),
    ]
    summaries = []
    emit(run_config(config("safe_edge_warmup_discard", "3,1", n_predict=32)), "warmup_discard")
    time.sleep(45)
    for label, ser in candidates:
        speed_runs = []
        for repeat in range(2):
            speed = run_config(config(f"speed_{label}_r{repeat + 1}", ser))
            speed_runs.append(speed)
            emit(speed, "speed")
            maybe_cool(speed)
        quality_rows = []
        for q in QUALITY_PROMPTS:
            run = run_config(config(f"quality_{q['name']}_{label}", ser, prompt=q["prompt"], n_predict=q["n_predict"]))
            answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
            regex_pass = bool(re.search(q["pass_regex"], answer, flags=re.I | re.S))
            gibberish = looks_like_gibberish(answer)
            passed = regex_pass and not gibberish
            row = {
                "quality": q["name"],
                "passed": passed,
                "regex_pass": regex_pass,
                "gibberish": gibberish,
                "answer": answer[:220],
                "gen_tps": run.get("gen_tps"),
                "log_path": run.get("log_path"),
            }
            quality_rows.append(row)
            print(json.dumps({"event": "quality", "candidate": label, **row, **summarize(run)}, sort_keys=True), flush=True)
            maybe_cool(run)
            if q["name"] in {"serbia", "mars"} and not passed:
                print(json.dumps({"event": "early_stop_candidate", "candidate": label, "failed": q["name"]}, sort_keys=True), flush=True)
                break
        summary = {
            "candidate": label,
            "ser": ser,
            "speed_best": max((float(run.get("gen_tps") or 0.0) for run in speed_runs), default=0.0),
            "speed_median": median([float(run.get("gen_tps") or 0.0) for run in speed_runs]),
            "quality_passes": sum(1 for row in quality_rows if row["passed"]),
            "quality_total": len(quality_rows),
            "quality_rows": quality_rows,
        }
        summaries.append(summary)
        print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)
    print(json.dumps({"event": "safe_edge_done", "summaries": summaries}, sort_keys=True), flush=True)
    resume_codex_gpu_helper()


def config(label: str, ser: str, **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, smart_expert_reduction=ser, **overrides)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def generated_answer(text: str) -> str:
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    return re.sub(r"\s+", " ", text.strip())


def looks_like_gibberish(answer: str) -> bool:
    if len(answer) < 20:
        return False
    alpha = sum(ch.isalpha() for ch in answer)
    spaces = answer.count(" ")
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"`+-=*/_%<>#") for ch in answer)
    very_short_words = sum(1 for word in answer.split() if len(word) == 1)
    words = max(1, len(answer.split()))
    return (
        alpha / max(1, len(answer)) < 0.45
        or weird > 8
        or very_short_words / words > 0.35
        or bool(re.search(r"[A-Za-z]{12,}[A-Z][a-z]{0,2}[A-Z]", answer))
    )


def emit(run: dict[str, Any], event: str) -> None:
    print(json.dumps({"event": event, **summarize(run)}, sort_keys=True), flush=True)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run.get("id"),
        "label": run.get("label"),
        "gen_tps": run.get("gen_tps"),
        "pp_tps": run.get("pp_tps"),
        "rss": run.get("peak_rss_bytes"),
        "metrics": run.get("metrics"),
        "exit_code": run.get("exit_code"),
        "log_path": run.get("log_path"),
    }


def maybe_cool(run: dict[str, Any]) -> None:
    metrics = run.get("metrics") or {}
    if int(metrics.get("page_faults") or 0) > 100000 or int(metrics.get("involuntary_context_switches") or 0) > 180000:
        print(json.dumps({"event": "cooldown", "seconds": 60, "label": run.get("label")}, sort_keys=True), flush=True)
        time.sleep(60)


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
