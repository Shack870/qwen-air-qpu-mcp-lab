#!/usr/bin/env python3
"""Generate paper data snapshots and SVG figures from the local Qwen Air lab DB.

This intentionally uses only the Python standard library so the paper artifacts
can be regenerated on the old Mac without installing a plotting stack.
"""

from __future__ import annotations

import csv
import html
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "qpu_lab.sqlite"
OUT = ROOT / "paper"
FIG = OUT / "figures"
DATA = OUT / "data"


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return list(con.execute(sql, args))
    finally:
        con.close()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_doc(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18222f; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .subtitle {{ font-size: 12px; fill: #506070; }}
    .axis {{ stroke: #b9c3cf; stroke-width: 1; }}
    .grid {{ stroke: #e4e9ef; stroke-width: 1; }}
    .label {{ font-size: 12px; fill: #29384a; }}
    .small {{ font-size: 11px; fill: #5c6978; }}
    .value {{ font-size: 12px; font-weight: 700; fill: #111827; }}
    .note {{ font-size: 11px; fill: #667085; }}
  </style>
{body}
</svg>
"""


def wrap_label(label: str, max_chars: int = 16) -> list[str]:
    words = label.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:3]


def bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict],
    *,
    value_key: str = "gen_tps",
    label_key: str = "label",
    color_key: str | None = None,
    max_y: float | None = None,
    footnote: str | None = None,
) -> None:
    width, height = 1100, 650
    ml, mr, mt, mb = 86, 42, 82, 150
    plot_w, plot_h = width - ml - mr, height - mt - mb
    values = [float(r[value_key]) for r in rows]
    max_val = max_y or max(values) * 1.15
    bar_gap = 20
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / len(rows)
    colors = {
        "baseline": "#64748b",
        "manual": "#2563eb",
        "external": "#7c3aed",
        "quantum": "#0891b2",
        "record": "#16a34a",
        "unsafe": "#dc2626",
        "example": "#d97706",
    }
    body = [
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text class="title" x="{ml}" y="36">{html.escape(title)}</text>',
        f'  <text class="subtitle" x="{ml}" y="58">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, int(max_val) + 1, 2):
        y = mt + plot_h - (tick / max_val) * plot_h
        body.append(f'  <line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}"/>')
        body.append(f'  <text class="small" text-anchor="end" x="{ml-10}" y="{y+4:.1f}">{tick}</text>')
    body.append(f'  <line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}"/>')
    body.append(f'  <line class="axis" x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}"/>')
    body.append(f'  <text class="small" text-anchor="middle" transform="translate(24 {mt+plot_h/2:.1f}) rotate(-90)">generation tokens/sec</text>')
    for i, row in enumerate(rows):
        x = ml + i * (bar_w + bar_gap)
        value = float(row[value_key])
        y = mt + plot_h - (value / max_val) * plot_h
        h = plot_h - (y - mt)
        color = colors.get(str(row.get(color_key or "kind", "")), "#2563eb")
        body.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        body.append(f'  <text class="value" text-anchor="middle" x="{x+bar_w/2:.1f}" y="{y-8:.1f}">{value:.2f}</text>')
        for li, line in enumerate(wrap_label(str(row[label_key]), 17)):
            body.append(
                f'  <text class="label" text-anchor="middle" x="{x+bar_w/2:.1f}" y="{mt+plot_h+24+li*15:.1f}">{html.escape(line)}</text>'
            )
    if footnote:
        body.append(f'  <text class="note" x="{ml}" y="{height-26}">{html.escape(footnote)}</text>')
    path.write_text(svg_doc(width, height, "\n".join(body)), encoding="utf-8")


def line_chart(path: Path, title: str, subtitle: str, rows: list[dict]) -> None:
    width, height = 1100, 650
    ml, mr, mt, mb = 90, 45, 82, 145
    plot_w, plot_h = width - ml - mr, height - mt - mb
    max_val = 18.0
    body = [
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text class="title" x="{ml}" y="36">{html.escape(title)}</text>',
        f'  <text class="subtitle" x="{ml}" y="58">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, 19, 2):
        y = mt + plot_h - (tick / max_val) * plot_h
        body.append(f'  <line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}"/>')
        body.append(f'  <text class="small" text-anchor="end" x="{ml-10}" y="{y+4:.1f}">{tick}</text>')
    body.append(f'  <line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}"/>')
    body.append(f'  <line class="axis" x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}"/>')
    points = []
    for i, row in enumerate(rows):
        x = ml + (i / (len(rows) - 1)) * plot_w
        y = mt + plot_h - (float(row["gen_tps"]) / max_val) * plot_h
        points.append((x, y))
    body.append(
        '  <polyline points="{}" fill="none" stroke="#0f766e" stroke-width="4" stroke-linejoin="round"/>'.format(
            " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        )
    )
    for i, (row, (x, y)) in enumerate(zip(rows, points)):
        kind = row.get("kind", "")
        color = "#dc2626" if kind == "unsafe" else "#0f766e"
        body.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        body.append(f'  <text class="value" text-anchor="middle" x="{x:.1f}" y="{y-13:.1f}">{float(row["gen_tps"]):.2f}</text>')
        for li, line in enumerate(wrap_label(str(row["label"]), 15)):
            body.append(f'  <text class="label" text-anchor="middle" x="{x:.1f}" y="{mt+plot_h+24+li*15:.1f}">{html.escape(line)}</text>')
    body.append(f'  <text class="note" x="{ml}" y="{height-25}">Milestones combine logged DB rows with notebook milestones explicitly marked in paper/data/milestones.csv.</text>')
    path.write_text(svg_doc(width, height, "\n".join(body)), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    db_rows = q(
        """
        select id, ts, label, source, gen_tps, pp_tps, total_ms, peak_rss_bytes,
               quality_flag, prompt_key, log_path, config_json, metrics_json, stdout_tail
        from runs
        where id in (635, 814, 1490, 1666, 1667, 1668, 1669, 1670, 1671)
        order by id
        """
    )
    selected = []
    for r in db_rows:
        cfg = json.loads(r["config_json"])
        metrics = json.loads(r["metrics_json"] or "{}")
        selected.append(
            {
                "id": r["id"],
                "timestamp": r["ts"],
                "label": r["label"],
                "source": r["source"],
                "gen_tps": f'{float(r["gen_tps"]):.2f}',
                "pp_tps": f'{float(r["pp_tps"]):.2f}' if r["pp_tps"] is not None else "",
                "ctx": cfg.get("ctx_size", ""),
                "batch": cfg.get("batch_size", ""),
                "ubatch": cfg.get("ubatch_size", ""),
                "ser": cfg.get("smart_expert_reduction", ""),
                "cheap_ranges": cfg.get("env_ser_cheap_ranges", ""),
                "kv": f'{cfg.get("cache_type_k", "")}/{cfg.get("cache_type_v", "")}',
                "quality_flag": r["quality_flag"] or "",
                "page_faults": metrics.get("page_faults", ""),
                "swaps": metrics.get("swaps", ""),
                "rss_gib": f'{(r["peak_rss_bytes"] or 0) / (1024**3):.2f}',
                "log_path": r["log_path"],
            }
        )
    write_csv(
        DATA / "selected_runs.csv",
        selected,
        [
            "id",
            "timestamp",
            "label",
            "source",
            "gen_tps",
            "pp_tps",
            "ctx",
            "batch",
            "ubatch",
            "ser",
            "cheap_ranges",
            "kv",
            "quality_flag",
            "page_faults",
            "swaps",
            "rss_gib",
            "log_path",
        ],
    )

    milestones = [
        {"label": "Out-of-box", "gen_tps": 0.09, "kind": "baseline", "evidence": "lab notebook"},
        {"label": "Early stable", "gen_tps": 2.40, "kind": "manual", "evidence": "lab notebook"},
        {"label": "Manual frontier", "gen_tps": 6.49, "kind": "manual", "evidence": "lab notebook"},
        {"label": "Pi 5 report", "gen_tps": 7.50, "kind": "external", "evidence": "Reddit 7-8 t/s midpoint"},
        {"label": "QPU boom", "gen_tps": 13.12, "kind": "quantum", "evidence": "run 635"},
        {"label": "Clean-room", "gen_tps": 13.91, "kind": "record", "evidence": "codex-off scoreboard"},
        {"label": "Strict Mars", "gen_tps": 14.03, "kind": "record", "evidence": "run 1490"},
        {"label": "Corrupted edge", "gen_tps": 16.53, "kind": "unsafe", "evidence": "run 814"},
    ]
    write_csv(DATA / "milestones.csv", milestones, ["label", "gen_tps", "kind", "evidence"])
    line_chart(
        FIG / "throughput_progression.svg",
        "Qwen Air Throughput Progression",
        "From first proof-of-life to quantum-guided strict record on a 2017 Intel MacBook Air",
        milestones,
    )

    qpu_focus = [
        {"label": "Manual frontier", "gen_tps": 6.49, "kind": "manual"},
        {"label": "Pi 5 report midpoint", "gen_tps": 7.50, "kind": "external"},
        {"label": "QPU first boom", "gen_tps": 13.12, "kind": "quantum"},
        {"label": "Clean-room", "gen_tps": 13.91, "kind": "record"},
        {"label": "Strict Mars record", "gen_tps": 14.03, "kind": "record"},
    ]
    bar_chart(
        FIG / "qpu_jump.svg",
        "Quantum-Guided Search Jump",
        "The QPU sampled compact QUBO candidates; the MacBook benchmark remained the judge",
        qpu_focus,
        max_y=16,
        footnote="Pi comparison uses the 7-8 tok/s Reddit-reported 8GB Pi 5 range midpoint.",
    )

    boundary = [
        {"label": "Strict-quality Mars", "gen_tps": 14.03, "kind": "record"},
        {"label": "Speed-only corrupted", "gen_tps": 16.53, "kind": "unsafe"},
    ]
    bar_chart(
        FIG / "quality_boundary.svg",
        "Speed/Quality Boundary",
        "A faster decode lane existed, but the paper record is the quality-gated 14.03 tok/s run",
        boundary,
        max_y=18,
        footnote="Run 814 reached 16.53 tok/s but emitted incoherent text; run 1490 is the strict-pass record.",
    )

    examples = [
        {"label": "Eagle poem", "gen_tps": 9.69, "kind": "example"},
        {"label": "Gingerbread", "gen_tps": 10.25, "kind": "example"},
        {"label": "UFO barber", "gen_tps": 10.21, "kind": "example"},
        {"label": "Largest state", "gen_tps": 5.66, "kind": "example"},
        {"label": "Yellowstone", "gen_tps": 8.91, "kind": "example"},
        {"label": "State/country", "gen_tps": 11.20, "kind": "example"},
    ]
    bar_chart(
        FIG / "prompt_examples.svg",
        "Prompt Example Throughput",
        "Representative responses under the same strict 16k-context configuration",
        examples,
        max_y=14,
        footnote="Short outputs are dominated by prompt/setup overhead; long creative prompts clipped at fixed token caps.",
    )

    source_summary_rows = q(
        """
        select source, count(*) as n, max(gen_tps) as max_gen, avg(gen_tps) as avg_gen
        from runs
        where gen_tps is not null
        group by source
        order by max(gen_tps) desc
        limit 30
        """
    )
    source_summary = [
        {
            "source": r["source"],
            "n": r["n"],
            "max_gen_tps": f'{float(r["max_gen"]):.2f}',
            "avg_gen_tps": f'{float(r["avg_gen"]):.2f}',
        }
        for r in source_summary_rows
    ]
    write_csv(DATA / "source_summary_top30.csv", source_summary, ["source", "n", "max_gen_tps", "avg_gen_tps"])


if __name__ == "__main__":
    main()
