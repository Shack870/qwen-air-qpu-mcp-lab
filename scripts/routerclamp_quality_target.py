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


PROMPT_CONTINUE = chat("Continue this comma-separated list of Mars facts: red planet, thin atmosphere,")

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
    "source": "routerclamp-target-quality",
    "timeout_seconds": 900,
}

QUALITY_PROMPTS: list[dict[str, Any]] = [
    {"name": "serbia", "prompt": chat("What is the capital of Serbia? Answer with one short sentence."), "n_predict": 48, "pass_regex": r"\bBelgrade\b"},
    {"name": "mars", "prompt": chat("What is the capital of Mars? Answer factually in one short sentence."), "n_predict": 64, "pass_regex": r"\b(no|not|none|does not|isn't|is not)\b.{0,100}\b(capital|city|government|country|nation|political)\b"},
    {"name": "prime", "prompt": chat("Write a compact Python function that checks whether n is prime."), "n_predict": 128, "pass_regex": r"def\s+\w+.*(%|sqrt|range).*return"},
    {"name": "moe", "prompt": chat("Explain a mixture-of-experts model in two concise sentences."), "n_predict": 96, "pass_regex": r"\b(expert|experts)\b.*\b(router|routing|select|selected|activates?|gate|gating)\b"},
    {"name": "logic", "prompt": chat("If all bloops are razzes and no razzes are blue, are any bloops blue? Answer yes or no and why."), "n_predict": 64, "pass_regex": r"\b(no|not)\b.*\bblue\b"},
]


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "target_quality_start"}), flush=True)
    candidates = [
        ("baseline_raw_ser3_1", 2304, 104, "3,1", []),
        ("raw_b2304_ub104_ser1_5", 2304, 104, "1,5", []),
        ("raw_b2368_ub96_ser1_5", 2368, 96, "1,5", []),
        ("raw_b2304_ub104_ser1_6", 2304, 104, "1,6", []),
    ]
    summaries = []
    for label, batch, ubatch, ser, extra_args in candidates:
        speed = run_config(case(f"target_speed_{label}", batch, ubatch, ser, extra_args))
        print(json.dumps({"event": "speed", **summarize(speed)}, sort_keys=True), flush=True)
        passes = 0
        details = []
        for q in QUALITY_PROMPTS:
            run = run_config(
                dict(
                    BASE,
                    label=f"target_quality_{q['name']}_{label}",
                    batch_size=batch,
                    ubatch_size=ubatch,
                    smart_expert_reduction=ser,
                    extra_args=extra_args,
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                )
            )
            answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
            ok = bool(re.search(q["pass_regex"], answer, flags=re.I | re.S))
            passes += int(ok)
            details.append({"quality": q["name"], "passed": ok, "answer": answer[:180], "gen_tps": run.get("gen_tps")})
            print(
                json.dumps(
                    {"event": "quality", "candidate": label, "quality": q["name"], "passed": ok, "answer": answer[:180], **summarize(run)},
                    sort_keys=True,
                ),
                flush=True,
            )
            if not ok and passes == 0 and q["name"] in {"serbia", "mars"}:
                print(json.dumps({"event": "candidate_early_quality_warning", "candidate": label}), flush=True)
            maybe_cool(run)
        summary = {"candidate": label, "speed_gen_tps": speed.get("gen_tps"), "passes": passes, "total": len(QUALITY_PROMPTS), "details": details}
        summaries.append(summary)
        print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)
    print(json.dumps({"event": "target_quality_done", "summaries": summaries}, sort_keys=True), flush=True)
    resume_codex_gpu_helper()


def case(label: str, batch: int, ubatch: int, ser: str, extra_args: list[str]) -> dict[str, Any]:
    return dict(BASE, label=label, batch_size=batch, ubatch_size=ubatch, smart_expert_reduction=ser, extra_args=extra_args)


def generated_answer(text: str) -> str:
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    return re.sub(r"\s+", " ", text.strip())


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
    if int(metrics.get("page_faults") or 0) > 50000 or int(metrics.get("involuntary_context_switches") or 0) > 140000:
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
