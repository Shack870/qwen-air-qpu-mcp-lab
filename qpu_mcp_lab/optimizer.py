from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bench import BenchConfig
from .db import all_successful_runs, best_runs
from .objective import score_run


@dataclass(frozen=True)
class BinaryVar:
    name: str
    off_value: Any
    on_value: Any


DEFAULT_VARS = [
    BinaryVar("batch_high", 1792, 2304),
    BinaryVar("ubatch_high", 96, 128),
    BinaryVar("threads_4", 2, 4),
    BinaryVar("kv_q6", "q4_1", "q6_0"),
    BinaryVar("ser_aggressive", "3,1", "2,0.85"),
    BinaryVar("ctx_16k", 8192, 16384),
    BinaryVar("np2_lane", False, True),
    BinaryVar("cheap24_top2", False, True),
]


def propose_classical_candidates(limit: int = 12) -> list[dict[str, Any]]:
    """Generate conservative next configs around the best observed frontier."""

    best = best_runs(1)
    base = BenchConfig()
    if best:
        cfg = best[0].get("config") or {}
        base = _config_from_observed(cfg, base)

    candidates: list[BenchConfig] = []
    for batch, ubatch, ctk, ctv, ser in itertools.product(
        _near(base.batch_size, [1536, 1792, 2048, 2304, 2560, 2816]),
        _near(base.ubatch_size, [64, 80, 96, 112, 128, 160]),
        ["q6_0", "q4_1", "iq4_nl", "q8_0"],
        ["q6_0", "q4_1", "iq4_nl", "q8_0"],
        ["3,1", "4,1", "2,0.85"],
    ):
        if ctk != ctv:
            continue
        if ser == "2,0.85" and ctk != "q6_0":
            continue
        cfg = base.model_copy(
            update={
                "label": f"classical_b{batch}_ub{ubatch}_{ctk}_ser{ser.replace(',', '-')}",
                "batch_size": batch,
                "ubatch_size": ubatch,
                "cache_type_k": ctk,
                "cache_type_v": ctv,
                "smart_expert_reduction": ser,
                "source": "optimizer-classical",
            }
        )
        candidates.append(cfg)

    # Small deterministic shuffle to avoid running near-duplicates first every time.
    rng = random.Random(1729)
    rng.shuffle(candidates)
    return [cfg.model_dump() for cfg in candidates[:limit]]


def build_qubo(limit_runs: int = 200) -> dict[str, Any]:
    """Fit a tiny quadratic surrogate and return a QUBO dictionary.

    This is intentionally compact: it is the bridge object that can be sent to
    a simulator or IBM sampler later. The MacBook benchmark remains the judge.
    """

    runs = all_successful_runs()[-limit_runs:]
    variables = DEFAULT_VARS
    n = len(variables)
    if len(runs) < max(8, n + 2):
        # Heuristic QUBO: favor the known good current defaults, lightly penalize
        # aggressive expert undervolting until real data earns it.
        q = np.zeros((n, n), dtype=float)
        for i, var in enumerate(variables):
            q[i, i] = {
                "batch_high": -0.10,
                "ubatch_high": -0.05,
                "threads_4": -0.20,
                "kv_q6": -0.45,
                "ser_aggressive": 0.20,
                "ctx_16k": -0.25,
            }.get(var.name, 0.0)
        return _qubo_payload(q, variables, offset=0.0, note="heuristic; import or run more data for fitted QUBO")

    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for run in runs:
        x = _encode_run(run, variables)
        if x is None:
            continue
        x_rows.append(_quadratic_features(x))
        score, _ = score_run(run)
        y_vals.append(float(score))

    if len(x_rows) < max(8, n + 2):
        q = np.zeros((n, n), dtype=float)
        return _qubo_payload(q, variables, offset=0.0, note="not enough encodable rows")

    x_mat = np.array(x_rows, dtype=float)
    y_vec = np.array(y_vals, dtype=float)
    ridge = 1e-3 * np.eye(x_mat.shape[1])
    beta = np.linalg.solve(x_mat.T @ x_mat + ridge, x_mat.T @ y_vec)
    offset = float(beta[0])
    q = np.zeros((n, n), dtype=float)
    cursor = 1
    for i in range(n):
        # Convert maximization of predicted t/s into minimization QUBO.
        q[i, i] = -float(beta[cursor])
        cursor += 1
    for i in range(n):
        for j in range(i + 1, n):
            q[i, j] = q[j, i] = -float(beta[cursor]) / 2.0
            cursor += 1
    return _qubo_payload(q, variables, offset=-offset, note=f"fitted from {len(x_rows)} rows")


def sample_qubo_local(qubo_payload: dict[str, Any], top_k: int = 8, max_random: int = 20000) -> list[dict[str, Any]]:
    q = np.array(qubo_payload["qubo"], dtype=float)
    n = q.shape[0]
    samples: list[tuple[float, tuple[int, ...]]] = []
    if n <= 20:
        iterable = itertools.product([0, 1], repeat=n)
    else:
        rng = random.Random(4242)
        iterable = (tuple(rng.randint(0, 1) for _ in range(n)) for _ in range(max_random))
    for bits in iterable:
        x = np.array(bits, dtype=float)
        energy = float(x @ q @ x + float(qubo_payload.get("offset", 0.0)))
        samples.append((energy, tuple(int(v) for v in bits)))
    samples.sort(key=lambda item: item[0])
    result = []
    seen = set()
    for energy, bits in samples:
        if bits in seen:
            continue
        seen.add(bits)
        result.append({"energy": energy, "bits": list(bits), "config": bits_to_config(bits, qubo_payload)})
        if len(result) >= top_k:
            break
    return result


def bits_to_config(bits: tuple[int, ...] | list[int], qubo_payload: dict[str, Any]) -> dict[str, Any]:
    cfg = BenchConfig().model_dump()
    for bit, var in zip(bits, qubo_payload["variables"], strict=False):
        cfg.update(_var_to_config(var["name"], var["on_value"] if bit else var["off_value"]))
    cfg["label"] = "qubo_" + "".join(str(int(b)) for b in bits)
    cfg["source"] = "optimizer-qubo"
    return cfg


def _near(value: int, options: list[int]) -> list[int]:
    return sorted(options, key=lambda x: (abs(x - value), x))[:4]


def _config_from_observed(observed: dict[str, Any], default: BenchConfig) -> BenchConfig:
    updates: dict[str, Any] = {}
    mapping = {
        "batch": "batch_size",
        "ubatch": "ubatch_size",
        "ctk": "cache_type_k",
        "ctv": "cache_type_v",
        "extra": "extra",
    }
    for src, dest in mapping.items():
        if src in observed and observed[src] not in ("", None):
            value = observed[src]
            if dest in {"batch_size", "ubatch_size"}:
                try:
                    value = int(value)
                except ValueError:
                    continue
            elif dest == "extra":
                continue
            updates[dest] = value
    return default.model_copy(update=updates)


def _encode_run(run: dict[str, Any], variables: list[BinaryVar]) -> list[int] | None:
    cfg = run.get("config") or {}
    encoded = []
    for var in variables:
        observed = _observed_var_value(cfg, var.name)
        if observed is None:
            return None
        encoded.append(1 if observed == var.on_value else 0)
    return encoded


def _observed_var_value(cfg: dict[str, Any], name: str) -> Any:
    if name == "batch_high":
        batch = _int_from_cfg(cfg, "batch", "batch_size")
        if batch is None:
            batch, _ = _batch_ubatch_from_label(cfg)
        if batch is None:
            return None
        return 2304 if batch >= 2048 else 1792
    if name == "ubatch_high":
        ubatch = _int_from_cfg(cfg, "ubatch", "ubatch_size")
        if ubatch is None:
            _, ubatch = _batch_ubatch_from_label(cfg)
        if ubatch is None:
            return None
        return 128 if ubatch >= 112 else 96
    if name == "threads_4":
        threads = _int_from_cfg(cfg, "threads", "t")
        return 4 if threads is None or threads >= 4 else 2
    if name == "kv_q6":
        ctk = cfg.get("ctk") or cfg.get("cache_type_k")
        if ctk is None and "q6" in str(cfg.get("label", "")):
            ctk = "q6_0"
        if ctk is None:
            return None
        return "q6_0" if ctk == "q6_0" else "q4_1"
    if name == "ser_aggressive":
        extra = str(cfg.get("extra", ""))
        ser = str(cfg.get("smart_expert_reduction", ""))
        return "2,0.85" if "2,0.85" in extra or ser == "2,0.85" else "3,1"
    if name == "ctx_16k":
        ctx = _int_from_cfg(cfg, "ctx_size", "ctx", "c")
        if ctx is None:
            ctx = 16384
        return 16384 if ctx >= 12000 else 8192
    if name == "np2_lane":
        extra = cfg.get("extra_args") or cfg.get("extra") or []
        if isinstance(extra, str):
            extra_s = extra
        else:
            extra_s = " ".join(str(item) for item in extra)
        return bool("-np" in extra_s and "-pps" in extra_s)
    if name == "cheap24_top2":
        ranges = str(cfg.get("env_ser_cheap_ranges") or "")
        cheap_min = cfg.get("env_ser_cheap_min")
        cheap_thresh = cfg.get("env_ser_cheap_thresh")
        try:
            cheap_min_i = int(cheap_min)
        except (TypeError, ValueError):
            cheap_min_i = None
        try:
            cheap_thresh_f = float(cheap_thresh)
        except (TypeError, ValueError):
            cheap_thresh_f = None
        return ranges == "24:30" and cheap_min_i == 2 and cheap_thresh_f == 1.0
    return None


def _int_from_cfg(cfg: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in cfg or cfg[key] in ("", None):
            continue
        try:
            return int(float(cfg[key]))
        except (TypeError, ValueError):
            continue
    return None


def _batch_ubatch_from_label(cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    import re

    label = str(cfg.get("label", ""))
    match = re.search(r"b([0-9]+)_ub([0-9]+)", label)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _quadratic_features(bits: list[int]) -> list[float]:
    features = [1.0]
    features.extend(float(bit) for bit in bits)
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            features.append(float(bits[i] * bits[j]))
    return features


def _var_to_config(name: str, value: Any) -> dict[str, Any]:
    if name == "batch_high":
        return {"batch_size": int(value)}
    if name == "ubatch_high":
        return {"ubatch_size": int(value)}
    if name == "threads_4":
        return {"threads": int(value), "threads_batch": int(value)}
    if name == "kv_q6":
        return {"cache_type_k": value, "cache_type_v": value}
    if name == "ser_aggressive":
        return {"smart_expert_reduction": value}
    if name == "ctx_16k":
        return {"ctx_size": int(value)}
    if name == "np2_lane":
        return {"extra_args": ["-np", "2", "-ns", "2", "-pps"] if value else []}
    if name == "cheap24_top2":
        if value:
            return {
                "env_ser_cheap_ranges": "24:30",
                "env_ser_cheap_min": 2,
                "env_ser_cheap_thresh": 1.0,
            }
        return {
            "env_ser_cheap_ranges": None,
            "env_ser_cheap_min": None,
            "env_ser_cheap_thresh": None,
        }
    return {}


def _qubo_payload(q: np.ndarray, variables: list[BinaryVar], offset: float, note: str) -> dict[str, Any]:
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": offset,
        "note": note,
        "variables": [
            {"name": var.name, "off_value": var.off_value, "on_value": var.on_value} for var in variables
        ],
        "qubo": q.tolist(),
    }
