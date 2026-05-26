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
    "source": "routerclamp-rescue",
    "timeout_seconds": 900,
}

QUICK_QUALITY = [
    {"name": "serbia", "prompt": chat("What is the capital of Serbia? Answer with one short sentence."), "n_predict": 48, "pass_regex": r"\bBelgrade\b"},
    {"name": "mars", "prompt": chat("What is the capital of Mars? Answer factually in one short sentence."), "n_predict": 64, "pass_regex": r"\b(no|not|none|does not|isn't|is not)\b.{0,100}\b(capital|city|government|country|nation|political)\b"},
]


def main() -> None:
    stop_codex_gpu_helper()
    print(json.dumps({"event": "rescue_start"}), flush=True)
    emit(run_config(config("rescue_warmup_discard", "3,1", n_predict=32)), "warmup_discard")
    time.sleep(45)
    results = []
    for candidate in candidates():
        speed = run_config(config(candidate["label"], candidate["ser"], **candidate["overrides"]))
        emit(speed, "speed")
        if float(speed.get("gen_tps") or 0.0) < 11.0:
            results.append({"candidate": candidate["label"], "speed": speed.get("gen_tps"), "quality": "skipped_slow"})
            maybe_cool(speed)
            continue
        quality_rows = []
        for q in QUICK_QUALITY:
            qcfg = config(
                f"rescue_quality_{q['name']}_{candidate['label']}",
                candidate["ser"],
                **candidate["overrides"],
                prompt=q["prompt"],
                n_predict=q["n_predict"],
            )
            run = run_config(qcfg)
            answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
            passed = bool(re.search(q["pass_regex"], answer, re.I | re.S)) and not looks_like_gibberish(answer)
            row = {"quality": q["name"], "passed": passed, "answer": answer[:180], "gen_tps": run.get("gen_tps")}
            quality_rows.append(row)
            print(json.dumps({"event": "quality", "candidate": candidate["label"], **row, **summarize(run)}, sort_keys=True), flush=True)
            maybe_cool(run)
            if not passed:
                break
        result = {
            "candidate": candidate["label"],
            "speed_gen_tps": speed.get("gen_tps"),
            "passes": sum(1 for row in quality_rows if row["passed"]),
            "total": len(quality_rows),
            "quality_rows": quality_rows,
        }
        results.append(result)
        print(json.dumps({"event": "candidate_summary", **result}, sort_keys=True), flush=True)
        maybe_cool(speed)
    print(json.dumps({"event": "rescue_done", "results": results}, sort_keys=True), flush=True)
    resume_codex_gpu_helper()


def candidates() -> list[dict[str, Any]]:
    return [
        {"label": "baseline_ser3_1", "ser": "3,1", "overrides": {}},
        {"label": "baseline_ser3_1_repack", "ser": "3,1", "overrides": {"extra_args": ["--run-time-repack"]}},
        {"label": "baseline_np2_ser3_1", "ser": "3,1", "overrides": {"batch_size": 2560, "ubatch_size": 96, "extra_args": ["-np", "2", "-ns", "2", "-pps"]}},
        {"label": "baseline_np2_ser3_1_repack", "ser": "3,1", "overrides": {"batch_size": 2560, "ubatch_size": 96, "extra_args": ["-np", "2", "-ns", "2", "-pps", "--run-time-repack"]}},
        {"label": "kv_k6_v4_ser3_1", "ser": "3,1", "overrides": {"cache_type_v": "q4_1"}},
        {"label": "kv_k6_v4_np2_ser3_1", "ser": "3,1", "overrides": {"batch_size": 2560, "ubatch_size": 96, "cache_type_v": "q4_1", "extra_args": ["-np", "2", "-ns", "2", "-pps"]}},
        {"label": "edge_ser1_8_repack", "ser": "1,8", "overrides": {"extra_args": ["--run-time-repack"]}},
        {"label": "edge_ser1_7_repack", "ser": "1,7", "overrides": {"extra_args": ["--run-time-repack"]}},
        {"label": "edge_ser1_6_repack", "ser": "1,6", "overrides": {"extra_args": ["--run-time-repack"]}},
        {"label": "edge_ser1_8_q8v", "ser": "1,8", "overrides": {"cache_type_v": "q8_0"}},
        {"label": "edge_ser1_8_k6v4", "ser": "1,8", "overrides": {"cache_type_v": "q4_1"}},
        {"label": "ser3_1_no_ignore_eos", "ser": "3,1", "overrides": {"ignore_eos": False}},
    ]


def config(label: str, ser: str, **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, smart_expert_reduction=ser, **overrides)


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
    words = answer.split()
    singletons = sum(1 for word in words if len(word) == 1)
    return alpha / max(1, len(answer)) < 0.45 or singletons / max(1, len(words)) > 0.35


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
