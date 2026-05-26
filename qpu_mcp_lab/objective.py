from __future__ import annotations

import math
import json
from typing import Any

from .config import safe_memory_gb


def score_run(run: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Score a run without hiding the tradeoffs.

    Tokens/sec is the headline, but the MacBook Air experiment cares about
    stability. This first objective penalizes failures, huge RSS, and slow
    end-to-end latency. We can tune weights after we have more rows.
    """

    gen_tps = float(run.get("gen_tps") or 0.0)
    pp_tps = float(run.get("pp_tps") or 0.0)
    exit_code = int(run.get("exit_code") if run.get("exit_code") is not None else 0)
    total_ms = float(run.get("total_ms") or 0.0)
    peak_rss = float(run.get("peak_rss_bytes") or 0.0)
    metrics = run.get("metrics") or {}
    cfg = _run_config(run)
    safe_bytes = safe_memory_gb() * 1024**3

    failure_penalty = 8.0 if exit_code != 0 or gen_tps <= 0 else 0.0
    expert_quality_penalty = _expert_quality_penalty(cfg)
    rss_gb_over = max(0.0, (peak_rss - safe_bytes) / 1024**3)
    rss_penalty = 0.75 * rss_gb_over * rss_gb_over
    latency_penalty = 0.10 * math.log1p(total_ms / 1000.0) if total_ms > 0 else 0.0
    page_fault_penalty = 0.15 * math.log1p(float(metrics.get("page_faults") or 0.0))
    swap_penalty = 2.0 * math.log1p(float(metrics.get("swaps") or 0.0))
    involuntary_switch_penalty = 0.03 * math.log1p(float(metrics.get("involuntary_context_switches") or 0.0))
    output_quality_penalty = _output_quality_penalty(str(run.get("stdout_tail") or ""))
    prompt_bonus = 0.05 * math.log1p(pp_tps)

    score = (
        gen_tps
        + prompt_bonus
        - failure_penalty
        - expert_quality_penalty
        - rss_penalty
        - latency_penalty
        - page_fault_penalty
        - swap_penalty
        - involuntary_switch_penalty
        - output_quality_penalty
    )
    components = {
        "gen_tps": gen_tps,
        "prompt_bonus": prompt_bonus,
        "failure_penalty": failure_penalty,
        "expert_quality_penalty": expert_quality_penalty,
        "rss_penalty": rss_penalty,
        "latency_penalty": latency_penalty,
        "page_fault_penalty": page_fault_penalty,
        "swap_penalty": swap_penalty,
        "involuntary_switch_penalty": involuntary_switch_penalty,
        "output_quality_penalty": output_quality_penalty,
        "score": score,
    }
    return score, components


def _run_config(run: dict[str, Any]) -> dict[str, Any]:
    cfg = run.get("config")
    if isinstance(cfg, dict):
        return cfg
    raw = run.get("config_json")
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _expert_quality_penalty(cfg: dict[str, Any]) -> float:
    ser = str(cfg.get("smart_expert_reduction") or "")
    cheap_ranges = str(cfg.get("env_ser_cheap_ranges") or "")
    if cheap_ranges:
        return 0.0
    if ser.startswith("1,"):
        return 10.0
    if ser.startswith("2,"):
        return 5.0
    return 0.0


def _output_quality_penalty(stdout_tail: str) -> float:
    """Catch the speed-only gibberish rows before they teach the optimizer.

    This is intentionally conservative. Strict task-specific quality checks
    remain in the experiment scripts; the objective just rejects outputs that
    are visibly not natural language or code.
    """

    if not stdout_tail:
        return 0.0

    text = stdout_tail
    if "sampling order:" in text:
        text = text.split("sampling order:", 1)[-1]
    if "llama_print_timings:" in text:
        text = text.split("llama_print_timings:", 1)[0]
    if "\n\n" in text:
        text = text.split("\n\n", 1)[-1]
    text = text.strip()
    if not text:
        return 0.0

    alpha = sum(ch.isalpha() for ch in text)
    weird = sum((not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"`+-=*/_%<>#") for ch in text)
    words = text.split()
    very_short_words = sum(1 for word in words if len(word) == 1)

    if "percouplespo" in text.lower():
        return 10.0
    if alpha / max(1, len(text)) < 0.35:
        return 8.0
    if weird > 12:
        return 8.0
    if words and very_short_words / len(words) > 0.40:
        return 6.0
    return 0.0


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    score_a, components_a = score_run(a)
    score_b, components_b = score_run(b)
    return {
        "a_id": a.get("id"),
        "b_id": b.get("id"),
        "score_delta_a_minus_b": score_a - score_b,
        "gen_tps_delta_a_minus_b": float(a.get("gen_tps") or 0.0) - float(b.get("gen_tps") or 0.0),
        "a_components": components_a,
        "b_components": components_b,
    }
