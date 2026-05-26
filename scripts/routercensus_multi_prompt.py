from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from routerclamp_strict_quality_scout import resume_codex_gpu_helper, stop_codex_gpu_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_mcp_lab.config import llama_bin, model_path

LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
BIN = llama_bin()
MODEL = model_path()


def chat(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


PROMPTS: list[dict[str, Any]] = [
    {
        "name": "mars_no_capital",
        "prompt": chat("What is the capital of Mars? Answer factually with exactly one short sentence."),
        "n_predict": 48,
    },
    {
        "name": "serbia_capital",
        "prompt": chat("What is the capital of Serbia? Answer with exactly one short factual sentence."),
        "n_predict": 48,
    },
    {
        "name": "prime_code",
        "prompt": chat("Write only a compact Python function named is_prime that checks whether n is prime."),
        "n_predict": 80,
    },
    {
        "name": "mars_facts",
        "prompt": chat("Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"),
        "n_predict": 64,
    },
    {
        "name": "code_rewrite",
        "prompt": chat("Rewrite this JavaScript as concise Python: function add(a,b){ return a + b }"),
        "n_predict": 64,
    },
    {
        "name": "summary",
        "prompt": chat("Summarize in one sentence: Mars is the fourth planet from the Sun. It has a thin atmosphere, polar ice caps, and two small moons."),
        "n_predict": 64,
    },
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = LOG_DIR / f"{stamp}-routercensus-multiprompt"
    out_dir.mkdir()

    stop_codex_gpu_helper()
    try:
        runs = []
        for prompt in PROMPTS:
            run = run_prompt(out_dir, prompt)
            runs.append(run)
            print(json.dumps({"event": "census_run", **run}, sort_keys=True), flush=True)
            time.sleep(5)

        analysis = analyze_runs(runs)
        json_path = DATA_DIR / f"{stamp}-routercensus-multiprompt.json"
        md_path = DATA_DIR / f"{stamp}-routercensus-multiprompt.md"
        json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(render_markdown(analysis), encoding="utf-8")
        print(json.dumps({"event": "census_done", "json": str(json_path), "markdown": str(md_path)}, sort_keys=True), flush=True)
        print(render_scoreboard(analysis), flush=True)
    finally:
        resume_codex_gpu_helper()


def run_prompt(out_dir: Path, prompt: dict[str, Any]) -> dict[str, Any]:
    log_path = out_dir / f"{prompt['name']}.log"
    census_path = out_dir / f"{prompt['name']}.census"
    env = os.environ.copy()
    env.update(
        {
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_WAIT_POLICY": "ACTIVE",
            "OMP_DYNAMIC": "FALSE",
            "LLAMA_SER_CHEAP_RANGES": "24:30",
            "LLAMA_SER_CHEAP_MIN": "2",
            "LLAMA_SER_CHEAP_THRESH": "1.0",
            "LLAMA_ROUTER_CENSUS": str(census_path),
            "LLAMA_ROUTER_CENSUS_TOPN": "8",
            "LLAMA_ROUTER_CENSUS_MAX_ROWS": "1",
        }
    )
    cmd = [
        "/usr/bin/time", "-l",
        "caffeinate", "-dimsu",
        str(BIN),
        "-m", str(MODEL),
        "-c", "16384",
        "-b", "2456",
        "-ub", "144",
        "-t", "4",
        "-tb", "4",
        "--cache-type-k", "q6_0",
        "--cache-type-v", "q6_0",
        "-fa", "1",
        "-ser", "3,1",
        "-n", str(prompt["n_predict"]),
        "--temp", "0.0",
        "--ignore-eos",
        "--no-display-prompt",
        "-p", prompt["prompt"],
    ]
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=900, check=False)
    wall = time.perf_counter() - start
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": prompt["name"],
        "exit_code": proc.returncode,
        "log_path": str(log_path),
        "census_path": str(census_path),
        "gen_tps": parse_tps(text, prompt_eval=False),
        "pp_tps": parse_tps(text, prompt_eval=True),
        "wall_seconds": round(wall, 3),
        "page_faults": parse_int(text, r"([0-9]+)\s+page faults"),
        "context_switches": parse_int(text, r"([0-9]+)\s+involuntary context switches"),
        "census_lines": sum(1 for _ in census_path.open("r", encoding="utf-8", errors="replace")) if census_path.exists() else 0,
        "sample": generated_answer(text)[:220],
    }


def parse_tps(text: str, *, prompt_eval: bool) -> float | None:
    if prompt_eval:
        pat = re.compile(r"prompt eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\s*\)", re.I)
    else:
        pat = re.compile(r"(?<!prompt )eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\s*\)", re.I)
    matches = list(pat.finditer(text))
    return float(matches[-1].group(1)) if matches else None


def parse_int(text: str, pattern: str) -> int | None:
    matches = list(re.finditer(pattern, text, flags=re.I))
    return int(matches[-1].group(1)) if matches else None


def generated_answer(text: str) -> str:
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    return re.sub(r"\s+", " ", text).strip()


LINE_RE = re.compile(
    r"layer=(?P<layer>\d+).*?min=(?P<min>\d+)\s+thresh=(?P<thresh>[0-9.eE+-]+)\s+top=(?P<top>[^ ]+)\s+active=(?P<active>\d+)\s+selected_mass=(?P<mass>[0-9.eE+-]+)"
)


def analyze_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_prompt = {}
    aggregate_rows = []
    for run in runs:
        rows = parse_census(Path(run["census_path"]))
        per_prompt[run["name"]] = summarize_rows(rows)
        aggregate_rows.extend(rows)
    aggregate = summarize_rows(aggregate_rows)
    return {
        "runs": runs,
        "aggregate": aggregate,
        "per_prompt": per_prompt,
        "candidate_bands": candidate_bands(aggregate),
    }


def parse_census(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        top = []
        for item in match.group("top").split(","):
            sid, sval = item.split(":", 1)
            top.append((int(sid), float(sval)))
        rows.append(
            {
                "layer": int(match.group("layer")),
                "active": int(match.group("active")),
                "selected_mass": float(match.group("mass")),
                "top": top,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)
    layers = {}
    for layer, layer_rows in sorted(by_layer.items()):
        top1_probs = [row["top"][0][1] for row in layer_rows if row["top"]]
        top2_masses = [sum(prob for _, prob in row["top"][:2]) for row in layer_rows]
        top3_masses = [sum(prob for _, prob in row["top"][:3]) for row in layer_rows]
        top8_masses = [sum(prob for _, prob in row["top"][:8]) for row in layer_rows]
        selected_masses = [row["selected_mass"] for row in layer_rows]
        hot = Counter(eid for row in layer_rows for eid, _ in row["top"][:3] if eid >= 0)
        layers[str(layer)] = {
            "rows": len(layer_rows),
            "top1_prob_mean": mean(top1_probs),
            "top2_mass_mean": mean(top2_masses),
            "top3_mass_mean": mean(top3_masses),
            "top8_mass_mean": mean(top8_masses),
            "selected_mass_mean": mean(selected_masses),
            "top1_unique": len({row["top"][0][0] for row in layer_rows if row["top"]}),
            "top3_unique": len({eid for row in layer_rows for eid, _ in row["top"][:3] if eid >= 0}),
            "dominant_top3": hot.most_common(8),
            "top2_to_top3_ratio": mean([safe_div(a, b) for a, b in zip(top2_masses, top3_masses)]),
        }
    return {"row_count": len(rows), "layers": layers}


def candidate_bands(summary: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for layer_s, info in summary["layers"].items():
        layer = int(layer_s)
        # High ratio means top-2 is close to top-3. Low unique count means less
        # expert wandering. This is a heuristic, not truth.
        score = (
            1.5 * (info["top2_to_top3_ratio"] or 0)
            + 0.5 * (info["top2_mass_mean"] or 0)
            - 0.02 * info["top3_unique"]
        )
        scored.append((score, layer, info))
    best = sorted(scored, reverse=True)[:16]
    worst = sorted(scored)[:16]
    return {
        "top2_candidates": [{"layer": layer, "score": score, **brief(info)} for score, layer, info in best],
        "protect_candidates": [{"layer": layer, "score": score, **brief(info)} for score, layer, info in worst],
        "contiguous_top2_bands": contiguous_bands([layer for _, layer, _ in best]),
    }


def brief(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "top2_to_top3_ratio": info["top2_to_top3_ratio"],
        "top2_mass_mean": info["top2_mass_mean"],
        "top3_unique": info["top3_unique"],
        "top1_unique": info["top1_unique"],
    }


def contiguous_bands(layers: list[int]) -> list[str]:
    if not layers:
        return []
    layers = sorted(set(layers))
    bands = []
    start = prev = layers[0]
    for layer in layers[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        bands.append(f"{start}:{prev + 1}")
        start = prev = layer
    bands.append(f"{start}:{prev + 1}")
    return bands


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def render_scoreboard(analysis: dict[str, Any]) -> str:
    lines = ["=== ROUTER CENSUS MULTI-PROMPT ==="]
    for run in analysis["runs"]:
        lines.append(
            f"{run['name']:<16} gen={run['gen_tps']} pp={run['pp_tps']} lines={run['census_lines']} "
            f"faults={run['page_faults']} ctxsw={run['context_switches']}"
        )
    lines.append("")
    lines.append("Top-2 candidate bands: " + ", ".join(analysis["candidate_bands"]["contiguous_top2_bands"]))
    lines.append("Best top-2 layers:")
    for item in analysis["candidate_bands"]["top2_candidates"][:10]:
        lines.append(
            f"  L{item['layer']:02d} score={item['score']:.3f} ratio={item['top2_to_top3_ratio']} "
            f"mass2={item['top2_mass_mean']} uniq3={item['top3_unique']}"
        )
    lines.append("Protect layers:")
    for item in analysis["candidate_bands"]["protect_candidates"][:10]:
        lines.append(
            f"  L{item['layer']:02d} score={item['score']:.3f} ratio={item['top2_to_top3_ratio']} "
            f"mass2={item['top2_mass_mean']} uniq3={item['top3_unique']}"
        )
    return "\n".join(lines)


def render_markdown(analysis: dict[str, Any]) -> str:
    return "# Router Census Multi-Prompt\n\n```text\n" + render_scoreboard(analysis) + "\n```\n"


if __name__ == "__main__":
    main()
