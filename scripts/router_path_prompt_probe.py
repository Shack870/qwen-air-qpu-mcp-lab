from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qpu_mcp_lab.bench import run_config
from routerclamp_strict_quality_scout import generated_answer, resume_codex_gpu_helper, stop_codex_gpu_helper, summarize


def chat(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "ctx_size": 16384,
    "batch_size": 2496,
    "ubatch_size": 128,
    "threads": 4,
    "threads_batch": 4,
    "cache_type_k": "q6_0",
    "cache_type_v": "q6_0",
    "smart_expert_reduction": "3,1",
    "env_ser_cheap_ranges": "24:30",
    "env_ser_cheap_min": 2,
    "env_ser_cheap_thresh": 1.0,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "router-path-prompt-probe",
    "timeout_seconds": 900,
}


PROMPTS = [
    (
        "baseline_continue",
        "<|im_start|>user\nContinue this comma-separated list of Mars facts: red planet, thin atmosphere,<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "true_short_phrases",
        chat("List true Mars facts as short comma-separated phrases. Keep every phrase factual and concise."),
    ),
    (
        "csv_mars",
        chat("Output one CSV row of true Mars facts, using short factual phrases only."),
    ),
    (
        "mars_colon",
        chat("Continue this factual pattern for Mars: Mars: red planet; Mars: thin atmosphere; Mars:"),
    ),
    (
        "no_capital_then_facts",
        chat("State that Mars has no capital, then list concise true Mars facts separated by commas."),
    ),
    (
        "astronomy_facts",
        chat("Continue this list of true astronomy facts: Mars is a planet, Mars has two moons, Mars"),
    ),
    (
        "wiki_style",
        chat("Write a compact factual encyclopedia-style sentence chain about Mars, with no fiction."),
    ),
]


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "router_path_prompt_probe_start"}), flush=True)
        cases = []
        for n_predict in (128, 256):
            for name, prompt in PROMPTS:
                cases.append(case(f"path_{name}_n{n_predict}", prompt, n_predict))
        # One high-batch check on the best baseline shape.
        cases.append(case("path_baseline_b2560_n256", PROMPTS[0][1], 256, batch_size=2560))

        runs = []
        for cfg in cases:
            run = run_config(cfg)
            detail = inspect_answer(Path(run["log_path"]))
            run["answer_detail"] = detail
            runs.append(run)
            print(json.dumps({"event": "run", **summarize(run), "answer_detail": detail}, sort_keys=True), flush=True)

        leaders = [run for run in runs if run.get("exit_code") == 0 and run.get("gen_tps") is not None]
        leaders.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
        print("=== ROUTER PATH PROMPT PROBE ===", flush=True)
        for run in leaders:
            cfg = json.loads(run["config_json"])
            detail = run.get("answer_detail") or {}
            print(
                f"{run['label']:<32} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"n={cfg.get('n_predict')} b={cfg.get('batch_size')} ub={cfg.get('ubatch_size')} "
                f"facts={detail.get('mentions_mars')} fiction={detail.get('fiction_red_flag')} "
                f"gib={detail.get('gibberish')} faults={run.get('metrics', {}).get('page_faults')}",
                flush=True,
            )
            print(f"  sample: {detail.get('sample')}", flush=True)
        print(json.dumps({"event": "router_path_prompt_probe_done", "best": summarize(leaders[0] if leaders else None)}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, prompt: str, n_predict: int, **overrides: Any) -> dict[str, Any]:
    cfg = dict(BASE)
    cfg.update(label=label, prompt=prompt, n_predict=n_predict)
    cfg.update(overrides)
    return cfg


def inspect_answer(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    answer = generated_answer(text)
    low = answer.lower()
    fiction_red_flag = bool(re.search(r"\bfictional|eastport|colorado|not a real|not real\b", low))
    gibberish = looks_like_gibberish(answer)
    return {
        "mentions_mars": "mars" in low,
        "fiction_red_flag": fiction_red_flag,
        "gibberish": gibberish,
        "sample": re.sub(r"\s+", " ", answer[:260]).strip(),
    }


def looks_like_gibberish(answer: str) -> bool:
    if len(answer) < 20:
        return False
    alpha = sum(ch.isalpha() for ch in answer)
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"`+-=*/_%<>#") for ch in answer)
    words = answer.split()
    very_short = sum(1 for word in words if len(word) == 1)
    return alpha / max(1, len(answer)) < 0.40 or weird > 12 or (words and very_short / len(words) > 0.40)


if __name__ == "__main__":
    main()
