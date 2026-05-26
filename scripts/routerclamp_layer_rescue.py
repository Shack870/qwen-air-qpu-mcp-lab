from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    "source": "routerclamp-layer-rescue",
    "timeout_seconds": 900,
}

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
        "pass_regex": r"\b(no|not|none|does not|isn't|is not)\b.{0,120}\b(capital|city|government|country|nation|political)\b",
    },
    {
        "name": "prime",
        "prompt": chat("Write a compact Python function that checks whether n is prime."),
        "n_predict": 96,
        "pass_regex": r"def\s+\w+.*(%|sqrt|range).*return",
    },
]


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "layer_rescue_start"}), flush=True)
        candidates = [
            ("baseline_ser3_1", "3,1", None, None),
            ("ser1_5_last8", "1,5", 0, 8),
            ("ser1_5_first8", "1,5", 8, 0),
            ("ser1_5_first8_last8", "1,5", 8, 8),
            ("ser1_8_last8", "1,8", 0, 8),
            ("ser1_8_first8_last8", "1,8", 8, 8),
            ("ser1_8_first12_last12", "1,8", 12, 12),
        ]

        summaries = []
        for label, ser, full_first, full_last in candidates:
            speed = run_config(case(f"speed_{label}", ser, full_first, full_last))
            print(json.dumps({"event": "speed", **summarize(speed)}, sort_keys=True), flush=True)

            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(
                    case(
                        f"quality_{q['name']}_{label}",
                        ser,
                        full_first,
                        full_last,
                        prompt=q["prompt"],
                        n_predict=q["n_predict"],
                    )
                )
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                regex_pass = bool(re.search(q["pass_regex"], answer, flags=re.I | re.S))
                gibberish = looks_like_gibberish(answer)
                passed = regex_pass and not gibberish
                row = {
                    "quality": q["name"],
                    "passed": passed,
                    "regex_pass": regex_pass,
                    "gibberish": gibberish,
                    "answer": answer[:240],
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
                "full_first": full_first,
                "full_last": full_last,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "quality_passes": sum(1 for row in quality_rows if row["passed"]),
                "quality_total": len(quality_rows),
                "quality_rows": quality_rows,
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)
            maybe_cool(speed)

        print(json.dumps({"event": "layer_rescue_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ser: str, full_first: int | None, full_last: int | None, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE, label=label, smart_expert_reduction=ser, **overrides)
    if full_first is not None:
        cfg["env_ser_full_first"] = full_first
    if full_last is not None:
        cfg["env_ser_full_last"] = full_last
    return cfg


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
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"`+-=*/_%<>#") for ch in answer)
    words = answer.split()
    very_short_words = sum(1 for word in words if len(word) == 1)
    return (
        alpha / max(1, len(answer)) < 0.45
        or weird > 8
        or very_short_words / max(1, len(words)) > 0.35
        or bool(re.search(r"[A-Za-z]{12,}[A-Z][a-z]{0,2}[A-Z]", answer))
    )


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
    if int(metrics.get("swaps") or 0) > 0:
        print(json.dumps({"event": "cooldown", "seconds": 20, "label": run.get("label")}, sort_keys=True), flush=True)
        time.sleep(20)


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


def resume_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,stat,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ && $2 ~ /^T/ {print $1}' | xargs -r kill -CONT"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    main()
