from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
RUNS_PATH = ROOT / "data" / "public_runs.csv"
FIGURE_BASE = "https://huggingface.co/spaces/Shack870/qwen-air-qpu-dashboard/resolve/main/figures"


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RUNS_PATH)
    numeric_cols = [
        "gen_tps",
        "pp_tps",
        "ctx_size",
        "batch_size",
        "ubatch_size",
        "threads",
        "threads_batch",
        "n_predict",
        "temp",
        "peak_rss_gib",
        "page_faults",
        "swaps",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["cache_type_k", "cache_type_v", "smart_expert_reduction", "env_ser_cheap_ranges", "lane"]:
        df[col] = df[col].fillna("").astype(str)
    return df


RUNS = load_runs()


MILESTONES = pd.DataFrame(
    [
        {"stage": "Out-of-box", "gen_tps": 0.09, "kind": "baseline"},
        {"stage": "Classical frontier", "gen_tps": 6.49, "kind": "systems"},
        {"stage": "QPU-informed jump", "gen_tps": 13.12, "kind": "hybrid quantum"},
        {"stage": "Clean-room Codex-off", "gen_tps": 13.91, "kind": "validation"},
        {"stage": "Strict record", "gen_tps": 14.03, "kind": "quality gated"},
        {"stage": "Rejected speed edge", "gen_tps": 16.53, "kind": "incoherent"},
    ]
)

QPU_PRESET = pd.DataFrame(
    [
        {"bit": "x0", "choice": "batch >= 2304", "why_it_mattered": "Large batches unlocked the high-throughput record family."},
        {"bit": "x1", "choice": "ubatch around 96-144", "why_it_mattered": "Moderate ubatch balanced throughput and memory pressure."},
        {"bit": "x2", "choice": "q6_0/q6_0 KV", "why_it_mattered": "q6 KV stayed coherent where lower-quality KV often degraded."},
        {"bit": "x3", "choice": "SER 3,1 base lane", "why_it_mattered": "A stable routed-compute setting before narrower layer experiments."},
        {"bit": "x4", "choice": "cheap layer band 24:30", "why_it_mattered": "Targeted expert reduction beat broad expert cuts."},
        {"bit": "x5", "choice": "temperature 0.0", "why_it_mattered": "Deterministic prompts made quality gates easier to compare."},
        {"bit": "x6", "choice": "16k context", "why_it_mattered": "Matched the record-family long-context target."},
        {"bit": "x7", "choice": "no swaps / low page faults", "why_it_mattered": "Prevented speed wins that were only page-cache accidents."},
    ]
)

QPU_CANDIDATES = {
    "11011110": {"candidate": "qpu_top1_b2304_ub96_repeat2", "gen_tps": 13.12, "status": "breakthrough", "notes": "First clear QPU-informed jump over the 6.49 classical frontier."},
    "11111110": {"candidate": "near_b2456_ub144", "gen_tps": 14.03, "status": "strict record family", "notes": "Local refinement around the QPU-revealed ridge."},
    "11010110": {"candidate": "record-family neighbor", "gen_tps": 13.6, "status": "worth testing", "notes": "Likely good throughput, needs quality gate."},
    "10010010": {"candidate": "conservative fallback", "gen_tps": 9.5, "status": "safe but slower", "notes": "Useful control lane."},
    "11100000": {"candidate": "speed-only edge", "gen_tps": 16.5, "status": "reject unless quality passes", "notes": "Fast neighbors often crossed the coherence boundary."},
}


DISPLAY_COLS = [
    "id",
    "lane",
    "gen_tps",
    "pp_tps",
    "source",
    "label",
    "ctx_size",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "smart_expert_reduction",
    "env_ser_cheap_ranges",
    "quality_flag",
    "page_faults",
    "swaps",
    "peak_rss_gib",
]


def fmt_df(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    out = df.loc[:, [c for c in DISPLAY_COLS if c in df.columns]].head(limit).copy()
    for col in ["gen_tps", "pp_tps", "peak_rss_gib"]:
        if col in out:
            out[col] = out[col].round(3)
    return out


def leaderboard(lane: str, min_tps: float, strict_only: bool, limit: int) -> pd.DataFrame:
    df = RUNS.copy()
    if lane != "all":
        df = df[df["lane"] == lane]
    df = df[df["gen_tps"].fillna(0) >= min_tps]
    if strict_only:
        df = df[(df["quality_flag"].fillna("") != "") | (df["lane"] == "strict_quality_record")]
    df = df.sort_values(["gen_tps", "id"], ascending=[False, False])
    return fmt_df(df, int(limit))


def estimate_config(
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    threads: int,
    threads_batch: int,
    kv_mode: str,
    smart_expert_reduction: str,
    cheap_ranges: str,
    n_predict: int,
    temp: float,
) -> tuple[str, pd.DataFrame]:
    cache_k, cache_v = kv_mode.split("/")
    df = RUNS.copy()
    df = df[df["exit_code"] == 0]
    if cache_k:
        df = df[df["cache_type_k"] == cache_k]
    if cache_v:
        df = df[df["cache_type_v"] == cache_v]
    if df.empty:
        return "No comparable runs found for that KV mode.", pd.DataFrame()

    cheap_ranges = (cheap_ranges or "").strip()
    ser = (smart_expert_reduction or "").strip()
    features = pd.DataFrame(
        {
            "ctx": (df["ctx_size"].fillna(ctx_size) - ctx_size) / max(ctx_size, 1),
            "batch": (df["batch_size"].fillna(batch_size) - batch_size) / max(batch_size, 1),
            "ubatch": (df["ubatch_size"].fillna(ubatch_size) - ubatch_size) / max(ubatch_size, 1),
            "threads": (df["threads"].fillna(threads) - threads) / max(threads, 1),
            "threads_batch": (df["threads_batch"].fillna(threads_batch) - threads_batch) / max(threads_batch, 1),
            "n_predict": (df["n_predict"].fillna(n_predict) - n_predict) / max(n_predict, 1),
            "temp": (df["temp"].fillna(temp) - temp) / 1.0,
        }
    )
    dist = np.sqrt((features**2).sum(axis=1))
    dist += np.where(df["smart_expert_reduction"] == ser, 0.0, 0.55)
    dist += np.where(df["env_ser_cheap_ranges"] == cheap_ranges, 0.0, 0.35)
    df = df.assign(distance=dist)
    nearest = df.sort_values(["distance", "gen_tps"], ascending=[True, False]).head(12).copy()
    weights = 1.0 / (nearest["distance"].to_numpy() + 0.08)
    pred = float(np.average(nearest["gen_tps"].to_numpy(), weights=weights))

    risky_neighbors = nearest["lane"].isin(["speed_only_rejected"]).mean()
    strict_neighbors = nearest["lane"].isin(["strict_quality_record", "record_family", "prompt_example"]).mean()
    quality = "medium"
    if risky_neighbors >= 0.35 or ser.startswith("1,"):
        quality = "high risk"
    elif strict_neighbors >= 0.5 and pred <= 14.2:
        quality = "better"

    confidence = "low"
    if nearest["distance"].min() < 0.05:
        confidence = "high"
    elif nearest["distance"].min() < 0.35:
        confidence = "medium"

    summary = f"""
### Estimated outcome

- Estimated generation speed: **{pred:.2f} tok/s**
- Quality risk: **{quality}**
- Confidence: **{confidence}**, based on nearest logged benchmark neighbors

This is an empirical nearest-neighbor estimate from the public run log. It is not
an inference benchmark and it does not run Qwen. The MacBook remains the judge.
"""
    return summary, fmt_df(nearest, 12)


def source_summary() -> pd.DataFrame:
    out = (
        RUNS.groupby("source", dropna=False)
        .agg(n=("id", "count"), max_gen_tps=("gen_tps", "max"), avg_gen_tps=("gen_tps", "mean"))
        .reset_index()
        .sort_values("max_gen_tps", ascending=False)
        .head(30)
    )
    out["max_gen_tps"] = out["max_gen_tps"].round(3)
    out["avg_gen_tps"] = out["avg_gen_tps"].round(3)
    return out


def qpu_candidate_demo(bitstring: str, shots: int) -> tuple[str, pd.DataFrame]:
    bits = "".join(ch for ch in bitstring if ch in "01")
    if len(bits) < len(QPU_PRESET):
        bits = bits.ljust(len(QPU_PRESET), "0")
    bits = bits[: len(QPU_PRESET)]
    selected = QPU_PRESET.copy()
    selected["selected"] = [bit == "1" for bit in bits]

    known = QPU_CANDIDATES.get(bits)
    if known:
        gen = known["gen_tps"]
        status = known["status"]
        candidate = known["candidate"]
        notes = known["notes"]
    else:
        score = sum(bit == "1" for bit in bits)
        gen = 7.0 + 0.75 * score
        if bits[2] == "1":
            gen += 1.0
        if bits[4] == "1":
            gen += 1.2
        if bits[:3] == "111" and bits[4] == "0":
            status = "speed-biased unknown"
        elif score >= 5:
            status = "plausible frontier candidate"
        else:
            status = "control candidate"
        candidate = "synthetic decoded config"
        notes = "Illustrative decode from the public bit encoding, not a real QPU job result."

    summary = f"""
### Candidate decode

- Bitstring: `{bits}`
- Candidate: **{candidate}**
- Estimated / observed gen speed: **{gen:.2f} tok/s**
- Status: **{status}**
- Shots requested in this demo: **{shots}**

{notes}

In the actual lab, QPU samples were decoded into concrete `llama.cpp` configs and
then tested locally. The quantum side changed which candidates Codex tested next;
the MacBook remained the judge.
"""
    return summary, selected


def img_tag(name: str, alt: str) -> str:
    return f'<img src="{FIGURE_BASE}/{name}" alt="{alt}" style="width:100%;height:auto;border:1px solid #d9e1e8;border-radius:8px;background:#fff;" />'


with gr.Blocks(title="Qwen Air QPU/MCP Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Qwen Air QPU/MCP Lab

Quantum-enhanced autoresearch for CPU-only Qwen3 MoE inference on a 2017 Intel
MacBook Air with 8GB RAM.

The QPU did not run the model. IBM Quantum sampled compact candidate search
spaces inside a Codex-driven loop; the MacBook judged each candidate with real
inference.

[GitHub](https://github.com/Shack870/qwen-air-qpu-mcp-lab) ·
[Collection](https://huggingface.co/collections/Shack870/qwen-air-qpu-mcp-lab-6a174dd8d752afe40a429846) ·
[Dataset artifacts](https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab) ·
[Paper draft](https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab/blob/main/paper/quantum_enhanced_legacy_moe_inference.md)
"""
    )

    with gr.Row():
        gr.HTML(img_tag("throughput_progression.svg", "Throughput progression"))
        gr.HTML(img_tag("qpu_jump.svg", "QPU-informed jump"))

    gr.Markdown("## Milestones")
    gr.Dataframe(MILESTONES, interactive=False, wrap=True)

    with gr.Tab("Leaderboard"):
        gr.Markdown(
            "Explore the sanitized public run log. Speed-only rejected rows are useful boundary data, not claimed quality results."
        )
        with gr.Row():
            lane = gr.Dropdown(
                ["all"] + sorted(RUNS["lane"].dropna().unique().tolist()),
                value="all",
                label="Lane",
            )
            min_tps = gr.Slider(0, 17, value=0, step=0.25, label="Minimum gen tok/s")
            strict_only = gr.Checkbox(False, label="Quality-flagged rows only")
            limit = gr.Slider(10, 100, value=30, step=10, label="Rows")
        board = gr.Dataframe(
            leaderboard("all", 0, False, 30),
            interactive=False,
            wrap=True,
            label="Public leaderboard",
        )
        for control in [lane, min_tps, strict_only, limit]:
            control.change(leaderboard, [lane, min_tps, strict_only, limit], board)

    with gr.Tab("Config Explorer"):
        gr.Markdown(
            "Estimate a candidate config from nearest logged runs. This is a search guide, not a benchmark."
        )
        with gr.Row():
            ctx_size = gr.Dropdown([128, 256, 512, 1024, 2048, 4096, 8192, 16384], value=16384, label="Context")
            batch_size = gr.Slider(512, 4096, value=2456, step=8, label="Batch")
            ubatch_size = gr.Slider(16, 256, value=144, step=8, label="Ubatch")
        with gr.Row():
            threads = gr.Slider(1, 4, value=4, step=1, label="Threads")
            threads_batch = gr.Slider(1, 4, value=4, step=1, label="Batch threads")
            kv_mode = gr.Dropdown(["q6_0/q6_0", "q4_0/q4_0", "q8_0/q8_0", "q6_0/q4_0"], value="q6_0/q6_0", label="KV mode")
        with gr.Row():
            smart_expert_reduction = gr.Dropdown(["3,1", "1,5", "1,1", "2,1", ""], value="3,1", label="SER")
            cheap_ranges = gr.Dropdown(["24:30", "", "24:28", "26:30", "18:24"], value="24:30", label="Cheap layer ranges")
            n_predict = gr.Slider(32, 256, value=128, step=32, label="Predicted tokens")
            temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
        estimate_button = gr.Button("Estimate from run log", variant="primary")
        estimate_md = gr.Markdown()
        nearest = gr.Dataframe(interactive=False, wrap=True, label="Nearest logged runs")
        estimate_button.click(
            estimate_config,
            [
                ctx_size,
                batch_size,
                ubatch_size,
                threads,
                threads_batch,
                kv_mode,
                smart_expert_reduction,
                cheap_ranges,
                n_predict,
                temp,
            ],
            [estimate_md, nearest],
        )

    with gr.Tab("QPU Candidate Demo"):
        gr.Markdown(
            """
This tab illustrates the quantum bridge in the project. Compact binary choices
encode candidate configuration decisions. A QPU samples bitstrings from the
compressed search problem; Codex decodes the bitstrings into concrete benchmark
configs; the MacBook tests them with real inference.

The examples below are an educational view of the public encoding and known
candidate neighborhood, not live QPU execution.
"""
        )
        with gr.Row():
            bitstring = gr.Textbox("11011110", label="Candidate bitstring")
            shots = gr.Slider(64, 1024, value=256, step=64, label="QPU shots")
        qpu_button = gr.Button("Decode candidate", variant="primary")
        qpu_md = gr.Markdown()
        qpu_table = gr.Dataframe(interactive=False, wrap=True, label="Bit choices")
        qpu_button.click(qpu_candidate_demo, [bitstring, shots], [qpu_md, qpu_table])
        gr.Markdown(
            """
Known examples:

- `11011110`: first QPU-informed 13.12 tok/s breakthrough neighborhood
- `11111110`: strict-record family neighborhood after local refinement
- `11100000`: speed-edge warning neighborhood
"""
        )

    with gr.Tab("Source Summary"):
        gr.Dataframe(source_summary(), interactive=False, wrap=True, label="Top run sources by max gen tok/s")
        gr.HTML(img_tag("quality_boundary.svg", "Quality boundary"))
        gr.HTML(img_tag("prompt_examples.svg", "Prompt examples"))


if __name__ == "__main__":
    demo.launch()
