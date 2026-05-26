from __future__ import annotations

import json
import os
import re
import subprocess
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
from routerclamp_strict_quality_scout import resume_codex_gpu_helper, stop_codex_gpu_helper, summarize


CPU_BIN = os.environ.get(
    "QPU_MCP_LAB_LLAMA_BIN",
    str(Path("~/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli").expanduser()),
)
METAL_BIN = os.environ.get(
    "QPU_MCP_LAB_METAL_LLAMA_BIN",
    str(Path("~/src/ik_llama.cpp/build-air-metal-omp/bin/llama-cli").expanduser()),
)

PROMPT = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

BASE: dict[str, Any] = {
    "prompt_key": "mars_fact_list",
    "prompt": PROMPT,
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
    "n_predict": 32,
    "temp": 0.0,
    "prewarm_model": False,
    "env_omp_dynamic": "FALSE",
    "env_omp_wait_policy": "ACTIVE",
    "source": "metal-hybrid-sidequest",
    "timeout_seconds": 240,
}


def main() -> None:
    stop_codex_gpu_helper()
    try:
        print(json.dumps({"event": "metal_hybrid_sidequest_start"}), flush=True)
        candidates = [
            case("cpu_control_lean_n32", CPU_BIN, []),
            case("metal_ngl0_n32", METAL_BIN, ["-ngl", "0"]),
            case("metal_ngl1_cpu_moe_n32", METAL_BIN, ["-ngl", "1", "--cpu-moe"]),
            case("metal_ngl2_cpu_moe_n32", METAL_BIN, ["-ngl", "2", "--cpu-moe"]),
            case("metal_ngl4_cpu_moe_n32", METAL_BIN, ["-ngl", "4", "--cpu-moe"]),
            case("metal_ngl1_all_n32", METAL_BIN, ["-ngl", "1"]),
            case("metal_ngl1_cpu_moe_no_fa_n32", METAL_BIN, ["-ngl", "1", "--cpu-moe", "-no-fa"]),
        ]

        runs = []
        for cfg in candidates:
            run = run_config(cfg)
            detail = inspect_log(Path(run["log_path"]))
            run["metal_detail"] = detail
            runs.append(run)
            print(json.dumps({"event": "run", **summarize(run), "metal_detail": detail}, sort_keys=True), flush=True)

        leaders = [
            run
            for run in runs
            if run.get("exit_code") == 0 and run.get("gen_tps") is not None
        ]
        leaders.sort(key=lambda run: float(run["gen_tps"]), reverse=True)
        print("=== METAL HYBRID SIDEQUEST ===", flush=True)
        for run in leaders:
            cfg = json.loads(run["config_json"])
            detail = run.get("metal_detail") or {}
            print(
                f"{run['label']:<32} gen={run.get('gen_tps')} pp={run.get('pp_tps')} "
                f"bin={Path(cfg.get('llama_bin_override') or '').parent.name} args={cfg.get('extra_args')} "
                f"offload={detail.get('offloaded_layers')} metal_buf={detail.get('metal_buffer_mib')} "
                f"metal_kv={detail.get('metal_kv_mib')} skipped={detail.get('skipped_kernels')} "
                f"faults={run.get('metrics', {}).get('page_faults')}",
                flush=True,
            )
        print(json.dumps({"event": "metal_hybrid_sidequest_done", "best": summarize(leaders[0] if leaders else None)}, sort_keys=True), flush=True)
    finally:
        resume_codex_gpu_helper()


def case(label: str, bin_path: str, extra_args: list[str]) -> dict[str, Any]:
    return dict(BASE, label=label, llama_bin_override=bin_path, extra_args=extra_args)


def inspect_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    detail: dict[str, Any] = {
        "skipped_kernels": len(re.findall(r"ggml_metal_init: skipping", text)),
        "metal_device": bool(re.search(r"Metal: using device|ggml_metal_init: GPU name", text)),
    }
    match = re.search(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers", text)
    if match:
        detail["offloaded_layers"] = f"{match.group(1)}/{match.group(2)}"
    match = re.search(r"Metal buffer size\s*=\s*([0-9.]+)\s*MiB", text)
    if match:
        detail["metal_buffer_mib"] = float(match.group(1))
    match = re.search(r"Metal KV buffer size\s*=\s*([0-9.]+)\s*MiB", text)
    if match:
        detail["metal_kv_mib"] = float(match.group(1))
    match = re.search(r"Metal compute buffer size\s*=\s*([0-9.]+)\s*MiB", text)
    if match:
        detail["metal_compute_mib"] = float(match.group(1))
    return detail


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        resume_codex_gpu_helper()
        raise
