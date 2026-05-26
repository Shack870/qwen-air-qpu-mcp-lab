from __future__ import annotations

import json
from typing import Any

from qpu_mcp_lab.bench import run_config


PROMPT_CHAT = (
    "<|im_start|>user\n"
    "List concise facts about Mars as comma-separated phrases."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PROMPT_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PROMPT_RAW_SEED = "Mars facts: red planet, thin atmosphere,"


BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "ctx_size": 16384,
    "threads": 4,
    "threads_batch": 4,
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "source": "record-lane-launchd",
    "timeout_seconds": 720,
}


def main() -> None:
    print(json.dumps({"event": "record_lane_sweep_start"}), flush=True)
    warm = dict(
        BASE,
        label="launchd_warmup_q6_q6_b1792_ub96_n32",
        batch_size=1792,
        ubatch_size=96,
        cache_type_k="q6_0",
        cache_type_v="q6_0",
        n_predict=32,
        prompt=PROMPT_CONTINUE,
    )
    emit(run_config(warm))

    prompt_probe_configs = [
        ("prompt_chat", PROMPT_CHAT),
        ("prompt_continue", PROMPT_CONTINUE),
        ("prompt_raw_seed", PROMPT_RAW_SEED),
    ]
    prompt_results = []
    for label, prompt in prompt_probe_configs:
        run = run_config(
            dict(
                BASE,
                label=f"launchd_{label}_q6_q6_b1792_ub96",
                batch_size=1792,
                ubatch_size=96,
                cache_type_k="q6_0",
                cache_type_v="q6_0",
                prompt=prompt,
            )
        )
        emit(run)
        prompt_results.append((run.get("gen_tps") or 0.0, prompt, label))

    prompt_results.sort(reverse=True, key=lambda item: item[0])
    best_prompt = prompt_results[0][1]
    best_prompt_label = prompt_results[0][2]
    print(
        json.dumps(
            {
                "event": "best_prompt_selected",
                "label": best_prompt_label,
                "gen_tps": prompt_results[0][0],
            }
        ),
        flush=True,
    )

    frontier = [
        ("q6_q6_b1792_ub96_repeat", 1792, 96, "q6_0", "q6_0", []),
        ("q6_q6_b1792_ub128", 1792, 128, "q6_0", "q6_0", []),
        ("q6_q6_b2304_ub96", 2304, 96, "q6_0", "q6_0", []),
        ("q6_q6_b3072_ub96", 3072, 96, "q6_0", "q6_0", []),
        ("q4_q4_b1792_ub96", 1792, 96, "q4_1", "q4_1", []),
        ("k6_v4_b1792_ub96", 1792, 96, "q6_0", "q4_1", []),
        ("k6_v4_b2048_ub96", 2048, 96, "q6_0", "q4_1", []),
        ("ngram_mod16_q6_q6_b1792_ub96", 1792, 96, "q6_0", "q6_0", ["--spec-type", "ngram-mod", "--draft-max", "16", "--draft-min", "1"]),
        ("mallocnano0_q6_q6_b1792_ub96", 1792, 96, "q6_0", "q6_0", []),
    ]
    for label, batch, ubatch, ctk, ctv, extra_args in frontier:
        cfg = dict(
            BASE,
            label=f"launchd_{label}",
            batch_size=batch,
            ubatch_size=ubatch,
            cache_type_k=ctk,
            cache_type_v=ctv,
            prompt=best_prompt,
            extra_args=extra_args,
        )
        if label.startswith("mallocnano0"):
            cfg["env_malloc_nano_zone"] = "0"
        emit(run_config(cfg))

    print(json.dumps({"event": "record_lane_sweep_done"}), flush=True)


def emit(run: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "event": "run",
                "label": run.get("label"),
                "gen_tps": run.get("gen_tps"),
                "pp_tps": run.get("pp_tps"),
                "rss": run.get("peak_rss_bytes"),
                "exit_code": run.get("exit_code"),
                "log_path": run.get("log_path"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
