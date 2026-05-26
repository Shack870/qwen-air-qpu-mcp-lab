from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
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
    "source": "routerclamp-strict-quality-scout",
    "timeout_seconds": 900,
}

QUALITY_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "serbia",
        "prompt": chat("What is the capital of Serbia? Answer with exactly one short factual sentence."),
        "n_predict": 48,
    },
    {
        "name": "mars",
        "prompt": chat("What is the capital of Mars? Answer factually with exactly one short sentence."),
        "n_predict": 64,
    },
    {
        "name": "prime",
        "prompt": chat("Write only a compact Python function named is_prime that checks whether n is prime."),
        "n_predict": 128,
    },
]


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "strict_quality_start"}), flush=True)
        candidates = [
            ("baseline_ser3_1", "3,1"),
            ("ser2_065", "2,0.65"),
            ("ser2_060", "2,0.60"),
            ("ser2_055", "2,0.55"),
            ("ser2_050", "2,0.50"),
            ("ser2_045", "2,0.45"),
            ("ser2_035", "2,0.35"),
            ("ser1_055", "1,0.55"),
        ]

        summaries = []
        for label, ser in candidates:
            speed = run_config(case(f"speed_{label}", ser))
            print(json.dumps({"event": "speed", **summarize(speed)}, sort_keys=True), flush=True)
            quality_rows = []
            for q in QUALITY_PROMPTS:
                run = run_config(case(
                    f"strict_{q['name']}_{label}",
                    ser,
                    prompt=q["prompt"],
                    n_predict=q["n_predict"],
                    ignore_eos=False,
                ))
                answer = generated_answer(Path(run["log_path"]).read_text(encoding="utf-8", errors="replace"))
                passed, reason = score_answer(q["name"], answer)
                row = {
                    "quality": q["name"],
                    "passed": passed,
                    "reason": reason,
                    "answer": answer[:260],
                    "gen_tps": run.get("gen_tps"),
                    "log_path": run.get("log_path"),
                }
                quality_rows.append(row)
                print(json.dumps({"event": "quality", "candidate": label, **row, **summarize(run)}, sort_keys=True), flush=True)
                if not passed:
                    print(json.dumps({"event": "early_stop_candidate", "candidate": label, "failed": q["name"], "reason": reason}, sort_keys=True), flush=True)
                    break

            summary = {
                "candidate": label,
                "ser": ser,
                "speed_gen_tps": speed.get("gen_tps"),
                "speed_pp_tps": speed.get("pp_tps"),
                "strict_passes": sum(1 for row in quality_rows if row["passed"]),
                "strict_total": len(quality_rows),
                "quality_rows": quality_rows,
            }
            summaries.append(summary)
            print(json.dumps({"event": "candidate_summary", **summary}, sort_keys=True), flush=True)

        print(json.dumps({"event": "strict_quality_done", "summaries": summaries}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, ser: str, **overrides: Any) -> dict[str, Any]:
    return dict(BASE, label=label, smart_expert_reduction=ser, **overrides)


def score_answer(name: str, answer: str) -> tuple[bool, str]:
    low = answer.lower()
    if name != "prime" and looks_like_gibberish(answer):
        return False, "gibberish"
    if name == "serbia":
        if "belgrade" not in low:
            return False, "missing_belgrade"
        if re.search(r"\b(belgradica|belice|belgrad|fictional|not real|no single capital|incorrect|not independent|previously part)\b", low):
            return False, "contradictory_or_invented_serbia"
        if re.search(r"\b(not|isn't|is not|actually not)\s+belgrade\b", low):
            return False, "contradictory_or_invented_serbia"
        return True, "ok"
    if name == "mars":
        if "mars" not in low:
            return False, "wrong_subject"
        if re.search(r"\b(planetia|miles|moses|same name|france|french|paris|crete|herat)\b", low):
            return False, "invented_mars_context"
        if re.search(r"\bmars\b.{0,40}\b(city|region|island)\b", low, re.S):
            return False, "invented_mars_context"
        if re.search(r"\bfictional\s+(planet|interplanetary|colony)\b", low):
            return False, "invented_mars_context"
        if re.search(r"\b(eastport|colorado|u\.s\.|usa|united states|state of|located on earth)\b", low):
            return False, "invented_mars_context"
        if re.search(r"\bmars\s+is\s+not\s+(a\s+)?real\s+(place|planet|location)\b", low):
            return False, "invented_mars_context"
        if "earth is the only planet that has a capital" in low:
            return False, "bad_earth_capital_claim"
        if re.search(r"capital of mars is (?!not|no|not applicable|a fictional concept|fictional concept)", low):
            return False, "asserts_mars_capital"
        if not re.search(r"\b(no|not|none|does not|isn't|is not|has no|not applicable)\b.{0,160}\b(capital|city|cities|government|governments|country|nation|political)\b", low, re.S):
            return False, "missing_no_capital_claim"
        return True, "ok"
    if name == "prime":
        ok, reason = verify_prime_code(answer)
        return ok, reason
    return False, "unknown_quality_prompt"


def verify_prime_code(answer: str) -> tuple[bool, str]:
    code = extract_python_code(answer)
    if not code:
        return False, "no_python_code"
    namespace: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace, namespace)
    except Exception as exc:
        return False, f"exec_error:{type(exc).__name__}"
    fn = namespace.get("is_prime")
    if not callable(fn):
        return False, "missing_is_prime"
    cases = {
        -1: False,
        0: False,
        1: False,
        2: True,
        3: True,
        4: False,
        5: True,
        9: False,
        17: True,
        25: False,
    }
    try:
        results = {n: bool(fn(n)) for n in cases}
    except Exception as exc:
        return False, f"runtime_error:{type(exc).__name__}"
    if results != cases:
        return False, f"wrong_results:{results}"
    return True, "ok"


def extract_python_code(answer: str) -> str | None:
    fence = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.I | re.S)
    if fence:
        return fence.group(1).strip()
    start = answer.find("def is_prime")
    if start == -1:
        return None
    code = answer[start:]
    lines = []
    for line in code.splitlines():
        if lines and line and not line.startswith((" ", "\t")) and not line.startswith("def "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def generated_answer(text: str) -> str:
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    return text.strip()


def looks_like_gibberish(answer: str) -> bool:
    if len(answer) < 20:
        return False
    alpha = sum(ch.isalpha() for ch in answer)
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"`+-=*/_%<>#") for ch in answer)
    words = answer.split()
    very_short_words = sum(1 for word in words if len(word) == 1)
    repeated_junk = bool(re.search(r"([0-9]{6,}|[A-Za-z]{2,}[0-9][A-Za-z0-9]{5,}|\\.\\s*){3,}", answer))
    return (
        alpha / max(1, len(answer)) < 0.45
        or weird > 8
        or very_short_words / max(1, len(words)) > 0.35
        or repeated_junk
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


def stop_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ {print $1}' | xargs -r kill -STOP"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


def resume_codex_gpu_helper() -> None:
    script = r"""ps -axo pid,stat,command | awk '/Codex Helper --type=gpu-process/ && !/awk/ && $2 ~ /^T/ {print $1}' | xargs -r kill -CONT"""
    subprocess.run(["/bin/zsh", "-lc", script], check=False, timeout=10)


if __name__ == "__main__":
    main()
