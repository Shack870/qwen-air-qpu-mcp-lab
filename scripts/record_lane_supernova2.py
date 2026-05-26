from __future__ import annotations

import json
from typing import Any

from qpu_mcp_lab.bench import run_config


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
    "threads": 4,
    "threads_batch": 4,
    "smart_expert_reduction": "3,1",
    "n_predict": 128,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "source": "record-lane-supernova2",
    "timeout_seconds": 720,
}


def main() -> None:
    print(json.dumps({"event": "supernova2_start"}), flush=True)
    cases = [
        ("q6_q6_b2048_ub96", 2048, 96, "q6_0", "q6_0"),
        ("q6_q6_b2048_ub128", 2048, 128, "q6_0", "q6_0"),
        ("q6_q6_b2176_ub96", 2176, 96, "q6_0", "q6_0"),
        ("q6_q6_b2176_ub112", 2176, 112, "q6_0", "q6_0"),
        ("q6_q6_b2304_ub80", 2304, 80, "q6_0", "q6_0"),
        ("q6_q6_b2304_ub96_repeat", 2304, 96, "q6_0", "q6_0"),
        ("q6_q6_b2304_ub112", 2304, 112, "q6_0", "q6_0"),
        ("q6_q6_b2304_ub128", 2304, 128, "q6_0", "q6_0"),
        ("q6_q6_b2432_ub96", 2432, 96, "q6_0", "q6_0"),
        ("q6_q6_b2432_ub112", 2432, 112, "q6_0", "q6_0"),
        ("q6_q6_b2560_ub96", 2560, 96, "q6_0", "q6_0"),
        ("q6_q6_b2560_ub112", 2560, 112, "q6_0", "q6_0"),
        ("q6_q6_b2816_ub96", 2816, 96, "q6_0", "q6_0"),
        ("q6_q6_b3072_ub96_repeat", 3072, 96, "q6_0", "q6_0"),
        ("q6_q6_b3072_ub112", 3072, 112, "q6_0", "q6_0"),
        ("q6_q6_b3328_ub96", 3328, 96, "q6_0", "q6_0"),
        ("k6_v4_b2048_ub96_repeat", 2048, 96, "q6_0", "q4_1"),
        ("k6_v4_b2304_ub96", 2304, 96, "q6_0", "q4_1"),
        ("k6_v4_b2560_ub96", 2560, 96, "q6_0", "q4_1"),
    ]
    for label, batch, ubatch, ctk, ctv in cases:
        run = run_config(
            dict(
                BASE,
                label=f"supernova2_{label}",
                batch_size=batch,
                ubatch_size=ubatch,
                cache_type_k=ctk,
                cache_type_v=ctv,
            )
        )
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
    print(json.dumps({"event": "supernova2_done"}), flush=True)


if __name__ == "__main__":
    main()
