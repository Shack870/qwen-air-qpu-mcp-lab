from __future__ import annotations

import math
from typing import Any

from qiskit.quantum_info import Statevector

from . import db
from .bench import BenchConfig
from .quantum import _qaoa_circuit_from_qubo, get_job_result, submit_qaoa_angle_sweep_job, submit_qaoa_job

PROMPT_MONKEY = (
    'Who is the author of the story "The Monkey\'s Paw"? What is the title of the story that '
    'features a character named "The Monkey\'s Paw"? What is the significance of the title '
    '"The Monkey\'s Paw"?'
)

PROMPT_MARS_CONTINUE = (
    "<|im_start|>user\n"
    "Continue this comma-separated list of Mars facts: red planet, thin atmosphere,"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_micro_frontier_qubo() -> dict[str, Any]:
    """QUBO focused on the current live frontier, not the whole universe."""

    names = [
        "batch_1920",
        "batch_2048",
        "batch_2176",
        "ubatch_80",
        "ubatch_96",
        "ubatch_104",
        "kv_k6_v4",
        "kv_q4_q4",
    ]
    n = len(names)
    q = [[0.0] * n for _ in range(n)]
    diag = {
        0: -0.30,
        1: -0.80,
        2: -0.25,
        3: -0.15,
        4: -0.75,
        5: -0.20,
        6: -0.95,
        7: -0.30,
    }
    for i, value in diag.items():
        q[i][i] = value
    for group in ([0, 1, 2], [3, 4, 5], [6, 7]):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                q[left][right] = q[right][left] = 1.5
    pairs = {
        (1, 4): -0.90,
        (1, 6): -0.90,
        (4, 6): -0.65,
        (0, 4): -0.25,
        (2, 4): -0.20,
        (1, 5): -0.15,
        (3, 6): -0.05,
        (7, 4): -0.10,
    }
    for (i, j), value in pairs.items():
        q[i][j] = q[j][i] = value / 2.0
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": 0.0,
        "decoder": "micro_frontier_v1",
        "note": "micro-QUBO around K6/V4 b2048 ub96 frontier",
        "variables": [{"name": name, "off_value": 0, "on_value": 1} for name in names],
        "qubo": q,
    }


def build_record_lane_qubo() -> dict[str, Any]:
    """QUBO focused on the hot record lane that broke 12 t/s."""

    names = [
        "batch_1792",
        "batch_2048",
        "batch_2304",
        "batch_2560",
        "batch_2816",
        "batch_3072",
        "batch_3328",
        "ubatch_80",
        "ubatch_96",
        "ubatch_112",
        "ubatch_128",
        "kv_q6_q6",
        "kv_k6_v4",
        "kv_q4_q4",
    ]
    n = len(names)
    q = [[0.0] * n for _ in range(n)]
    diag = {
        0: -0.65,
        1: -0.85,
        2: -1.35,
        3: -0.90,
        4: -0.80,
        5: -1.05,
        6: -0.55,
        7: -0.35,
        8: -1.20,
        9: -0.65,
        10: -0.90,
        11: -1.20,
        12: -0.90,
        13: -0.55,
    }
    for i, value in diag.items():
        q[i][i] = value
    for group in ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13]):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                q[left][right] = q[right][left] = 1.8
    pairs = {
        (2, 8): -1.00,   # b2304/ub96 was the 12.95 record
        (2, 11): -0.90,
        (5, 8): -0.75,   # b3072/ub96 was also strong
        (5, 11): -0.65,
        (0, 10): -0.65,  # b1792/ub128 reached 7.8-11.4 depending heat state
        (1, 8): -0.45,
        (1, 12): -0.75,  # K6/V4 b2048/ub96 reached 12.1
        (3, 8): -0.35,
        (4, 8): -0.30,
        (6, 8): -0.20,
        (8, 11): -0.70,
        (8, 12): -0.45,
        (10, 11): -0.25,
    }
    for (i, j), value in pairs.items():
        q[i][j] = q[j][i] = value / 2.0
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": 0.0,
        "decoder": "record_lane_v1",
        "note": "hot record-lane QUBO after 12.95 t/s breakthrough",
        "variables": [{"name": name, "off_value": 0, "on_value": 1} for name in names],
        "qubo": q,
    }


def build_ridge_v2_qubo() -> dict[str, Any]:
    """QUBO for the narrow b2304/ub96/q6 ridge after the 13.12 t/s run."""

    names = [
        "batch_2176",
        "batch_2240",
        "batch_2304",
        "batch_2368",
        "batch_2432",
        "batch_2560",
        "ubatch_88",
        "ubatch_96",
        "ubatch_104",
        "ubatch_112",
        "kv_q6_q6",
        "kv_k6_v4",
    ]
    n = len(names)
    q = [[0.0] * n for _ in range(n)]
    diag = {
        0: -0.70,
        1: -1.05,
        2: -1.45,
        3: -1.05,
        4: -0.85,
        5: -0.65,
        6: -0.70,
        7: -1.35,
        8: -0.70,
        9: -0.50,
        10: -1.30,
        11: -0.45,
    }
    for i, value in diag.items():
        q[i][i] = value
    for group in ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11]):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                q[left][right] = q[right][left] = 1.9
    pairs = {
        (2, 7): -1.20,  # b2304/ub96 is the current record ridge
        (2, 10): -1.00,
        (1, 7): -0.70,
        (3, 7): -0.70,
        (2, 6): -0.45,
        (2, 8): -0.45,
        (4, 7): -0.35,
        (0, 7): -0.25,
        (5, 7): -0.20,
        (7, 10): -0.90,
        (1, 10): -0.35,
        (3, 10): -0.35,
    }
    for (i, j), value in pairs.items():
        q[i][j] = q[j][i] = value / 2.0
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": 0.0,
        "decoder": "ridge_v2",
        "note": "narrow ridge after 13.12 t/s: explore b2240/b2368 and ub88/ub104 around b2304/ub96",
        "variables": [{"name": name, "off_value": 0, "on_value": 1} for name in names],
        "qubo": q,
    }


def build_moonshot_ridge_v3_qubo() -> dict[str, Any]:
    """QUBO for the post-13.9 t/s ridge, including the surprising -np lane."""

    names = [
        "batch_2304",
        "batch_2368",
        "batch_2432",
        "batch_2560",
        "batch_2688",
        "ubatch_88",
        "ubatch_96",
        "ubatch_104",
        "ubatch_112",
        "parallel_1",
        "parallel_2",
        "parallel_4",
        "kv_q6_q6",
        "ctx_checkpoints_off",
    ]
    n = len(names)
    q = [[0.0] * n for _ in range(n)]
    diag = {
        0: -1.00,  # b2304 remains stable and strong
        1: -1.35,  # b2368 produced the first 13.81 raw record
        2: -0.65,
        3: -1.20,  # b2560 won only with -np2 so far
        4: -0.35,
        5: -0.90,
        6: -1.10,
        7: -1.45,  # ub104 produced 13.83 raw
        8: -0.55,
        9: -1.15,
        10: -0.95, # -np 2 was strong at n=96 but less convincing at n=128
        11: -0.55,
        12: -1.30,
        13: 0.35,  # ctx checkpoint off lost under clean state
    }
    for i, value in diag.items():
        q[i][i] = value
    for group in ([0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                q[left][right] = q[right][left] = 2.0
    pairs = {
        (0, 7): -1.05,  # b2304/ub104 raw record
        (1, 6): -1.00,  # b2368/ub96 raw record
        (3, 6): -0.80,
        (3, 10): -0.55, # b2560/np2 record was not stable at 128 tokens
        (6, 10): -0.35,
        (7, 10): -0.25,
        (0, 9): -0.65,
        (1, 9): -0.55,
        (0, 10): -0.10,
        (1, 10): -0.10,
        (11, 6): -0.10,
        (12, 6): -0.65,
        (12, 7): -0.75,
        (13, 0): 0.20,
        (13, 1): 0.20,
        (13, 3): 0.20,
    }
    for (i, j), value in pairs.items():
        q[i][j] = q[j][i] = value / 2.0
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": 0.0,
        "decoder": "moonshot_ridge_v3",
        "note": "post-13.9 ridge: b2368 raw, b2304/ub104 raw, and b2560/-np2 memory-shape lane",
        "variables": [{"name": name, "off_value": 0, "on_value": 1} for name in names],
        "qubo": q,
    }


def build_impl_flags_qubo() -> dict[str, Any]:
    """QUBO for implementation/runtime switches around the b2304/ub96 ridge."""

    names = [
        "no_cont_batching",
        "ctx_checkpoints_off",
        "cache_ram_off",
        "merge_qkv",
        "merge_up_gate",
        "grouped_routing",
        "async_scheduler",
        "attn_max_512",
        "k_hadamard",
        "v_hadamard",
    ]
    n = len(names)
    q = [[0.0] * n for _ in range(n)]
    diag = {
        0: -0.10,
        1: -0.60,
        2: -0.10,
        3: 2.00,
        4: 1.60,
        5: 0.10,
        6: -0.05,
        7: 0.10,
        8: 0.25,
        9: 0.25,
    }
    for i, value in diag.items():
        q[i][i] = value
    pairs = {
        (0, 1): -0.20,
        (3, 4): 0.80,
        (3, 5): 0.20,
        (4, 5): 0.20,
        (6, 7): -0.15,
        (8, 9): 0.65,
        (2, 8): 0.20,
        (2, 9): 0.20,
    }
    for (i, j), value in pairs.items():
        q[i][j] = q[j][i] = value / 2.0
    return {
        "type": "qubo",
        "sense": "minimize",
        "offset": 0.0,
        "decoder": "impl_flags_v1",
        "note": "implementation flag combinations around b2304/ub96/q6/q6/-ser 3,1",
        "variables": [{"name": name, "off_value": 0, "on_value": 1} for name in names],
        "qubo": q,
    }


def submit_micro_frontier_job(
    backend_name: str,
    shots: int = 256,
    gamma: float | None = None,
    beta: float | None = None,
    auto_angle: bool = True,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_micro_frontier_qubo()
    chosen = {"gamma": gamma if gamma is not None else 1.0, "beta": beta if beta is not None else 0.7}
    if auto_angle:
        sweep = sweep_angles(payload)
        chosen = {"gamma": sweep[0]["gamma"], "beta": sweep[0]["beta"]}
    result = submit_qaoa_job(
        payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=chosen["gamma"],
        beta=chosen["beta"],
    )
    result["angle_source"] = "local_statevector_sweep" if auto_angle else "manual"
    return result


def submit_record_lane_job(
    backend_name: str,
    shots: int = 256,
    gamma: float | None = None,
    beta: float | None = None,
    auto_angle: bool = True,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_record_lane_qubo()
    chosen = {"gamma": gamma if gamma is not None else 1.0, "beta": beta if beta is not None else 0.7}
    if auto_angle:
        sweep = sweep_angles(payload)
        chosen = {"gamma": sweep[0]["gamma"], "beta": sweep[0]["beta"]}
    result = submit_qaoa_job(
        payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=chosen["gamma"],
        beta=chosen["beta"],
    )
    result["angle_source"] = "local_statevector_sweep" if auto_angle else "manual"
    return result


def submit_ridge_v2_job(
    backend_name: str,
    shots: int = 256,
    gamma: float | None = None,
    beta: float | None = None,
    auto_angle: bool = True,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_ridge_v2_qubo()
    chosen = {"gamma": gamma if gamma is not None else 1.0, "beta": beta if beta is not None else 0.7}
    if auto_angle:
        sweep = sweep_angles(payload)
        chosen = {"gamma": sweep[0]["gamma"], "beta": sweep[0]["beta"]}
    result = submit_qaoa_job(
        payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=chosen["gamma"],
        beta=chosen["beta"],
    )
    result["angle_source"] = "local_statevector_sweep" if auto_angle else "manual"
    return result


def submit_moonshot_ridge_v3_job(
    backend_name: str,
    shots: int = 256,
    gamma: float | None = None,
    beta: float | None = None,
    auto_angle: bool = True,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_moonshot_ridge_v3_qubo()
    chosen = {"gamma": gamma if gamma is not None else 1.0, "beta": beta if beta is not None else 0.7}
    if auto_angle:
        chosen = {"gamma": 1.0, "beta": 0.7}
    result = submit_qaoa_job(
        payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=chosen["gamma"],
        beta=chosen["beta"],
    )
    result["angle_source"] = "fixed_moonshot_default" if auto_angle else "manual"
    return result


def submit_moonshot_ridge_v3_angle_sweep_job(
    backend_name: str,
    shots_per_angle: int = 64,
    angle_count: int = 4,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_moonshot_ridge_v3_qubo()
    fixed_angles = [
        {"gamma": 0.50, "beta": 0.40},
        {"gamma": 0.75, "beta": 0.60},
        {"gamma": 1.00, "beta": 0.70},
        {"gamma": 1.25, "beta": 0.80},
        {"gamma": 1.50, "beta": 1.00},
    ]
    angles = fixed_angles[:angle_count]
    result = submit_qaoa_angle_sweep_job(
        payload,
        backend_name=backend_name,
        angles=angles,
        shots_per_angle=shots_per_angle,
        allow_real_qpu=allow_real_qpu,
    )
    result["angle_source"] = "fixed_moonshot_grid"
    return result


def submit_impl_flags_job(
    backend_name: str,
    shots: int = 256,
    gamma: float | None = None,
    beta: float | None = None,
    auto_angle: bool = True,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_impl_flags_qubo()
    chosen = {"gamma": gamma if gamma is not None else 1.0, "beta": beta if beta is not None else 0.7}
    if auto_angle:
        sweep = sweep_angles(payload)
        chosen = {"gamma": sweep[0]["gamma"], "beta": sweep[0]["beta"]}
    result = submit_qaoa_job(
        payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=chosen["gamma"],
        beta=chosen["beta"],
    )
    result["angle_source"] = "local_statevector_sweep" if auto_angle else "manual"
    return result


def submit_impl_flags_angle_sweep_job(
    backend_name: str,
    shots_per_angle: int = 64,
    angle_count: int = 4,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    payload = build_impl_flags_qubo()
    angles = [
        {"gamma": float(row["gamma"]), "beta": float(row["beta"])}
        for row in sweep_angles(payload)[:angle_count]
    ]
    result = submit_qaoa_angle_sweep_job(
        payload,
        backend_name=backend_name,
        angles=angles,
        shots_per_angle=shots_per_angle,
        allow_real_qpu=allow_real_qpu,
    )
    result["angle_source"] = "local_statevector_sweep_top_k"
    return result


def sweep_angles(payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = payload or build_micro_frontier_qubo()
    gammas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    betas = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for gamma in gammas:
        for beta in betas:
            rows.append(_evaluate_angles(payload, gamma, beta))
    rows.sort(key=lambda row: (row["expected_energy"], -row["top_probability"]))
    return rows


def decode_job_candidates(job_id: str, top_k: int = 12) -> dict[str, Any]:
    result = get_job_result(job_id)
    counts = result.get("counts") or {}
    job = db.get_quantum_job(job_id)
    payload = (job or {}).get("payload") or build_micro_frontier_qubo()
    rows = counts_to_candidates(job_id, counts, payload, top_k=top_k)
    return {"job_id": job_id, "done": result.get("done"), "candidates": rows}


def counts_to_candidates(
    job_id: str,
    counts: dict[str, int],
    payload: dict[str, Any],
    top_k: int = 12,
) -> list[dict[str, Any]]:
    counts = _flatten_angle_sweep_counts(counts)
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for bitstring, shot_count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        for orientation, bits in (("direct", bitstring), ("reverse", bitstring[::-1])):
            if payload.get("decoder") == "record_lane_v1":
                cfg = decode_record_lane(bits)
                canonical_bits = encode_record_lane_config(cfg)
            elif payload.get("decoder") == "ridge_v2":
                cfg = decode_ridge_v2(bits)
                canonical_bits = encode_ridge_v2_config(cfg)
            elif payload.get("decoder") == "moonshot_ridge_v3":
                cfg = decode_moonshot_ridge_v3(bits)
                canonical_bits = encode_moonshot_ridge_v3_config(cfg)
            elif payload.get("decoder") == "impl_flags_v1":
                cfg = decode_impl_flags(bits)
                canonical_bits = encode_impl_flags_config(cfg)
            else:
                cfg = decode_micro_frontier(bits)
                canonical_bits = encode_micro_frontier_config(cfg)
            key = (
                cfg["batch_size"],
                cfg["ubatch_size"],
                cfg["cache_type_k"],
                cfg["cache_type_v"],
                cfg["smart_expert_reduction"],
                tuple(cfg.get("extra_args") or []),
            )
            energy = qubo_energy(canonical_bits, payload)
            if key in candidates:
                candidates[key]["shot_count"] += shot_count
                candidates[key]["samples"].append(
                    {"bitstring": bitstring, "orientation": orientation, "count": shot_count, "decoded_bits": bits}
                )
                candidates[key]["energy"] = min(float(candidates[key]["energy"]), energy)
                continue
            row = {
                "bitstring": bitstring,
                "orientation": orientation,
                "decoded_bits": bits,
                "shot_count": shot_count,
                "energy": energy,
                "canonical_bits": "".join(str(bit) for bit in canonical_bits),
                "config": cfg,
                "samples": [
                    {"bitstring": bitstring, "orientation": orientation, "count": shot_count, "decoded_bits": bits}
                ],
            }
            candidates[key] = row
            db.insert_quantum_candidate(
                job_id=job_id,
                bitstring=bitstring,
                orientation=orientation,
                shot_count=shot_count,
                energy=energy,
                config=cfg,
            )
    rows = list(candidates.values())
    rows.sort(key=lambda row: (float(row["energy"]), -int(row["shot_count"])))
    return rows[:top_k]


def _flatten_angle_sweep_counts(counts: dict[str, Any]) -> dict[str, int]:
    if not counts:
        return {}
    if all(isinstance(value, dict) for value in counts.values()):
        flat: dict[str, int] = {}
        for sub_counts in counts.values():
            for bitstring, count in sub_counts.items():
                flat[str(bitstring)] = flat.get(str(bitstring), 0) + int(count)
        return flat
    return {str(bitstring): int(count) for bitstring, count in counts.items()}


def decode_record_lane(bits: str) -> dict[str, Any]:
    b = [int(char) for char in bits]
    batch_options = [1792, 2048, 2304, 2560, 2816, 3072, 3328]
    ubatch_options = [80, 96, 112, 128]
    batch = _choose_one_hot(b, range(0, 7), batch_options, default=2304)
    ubatch = _choose_one_hot(b, range(7, 11), ubatch_options, default=96)

    cache_type_k = "q6_0"
    cache_type_v = "q6_0"
    kv_choice = _choose_one_hot(b, range(11, 14), ["q6_q6", "k6_v4", "q4_q4"], default="q6_q6")
    if kv_choice == "k6_v4":
        cache_type_k, cache_type_v = "q6_0", "q4_1"
    elif kv_choice == "q4_q4":
        cache_type_k, cache_type_v = "q4_1", "q4_1"

    return BenchConfig(
        label=f"qpu_record_{bits}",
        ctx_size=16384,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=4,
        threads_batch=4,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        smart_expert_reduction="3,1",
        n_predict=128,
        prompt_key="mars_fact_list",
        prompt=PROMPT_MARS_CONTINUE,
        source="qpu-record-lane",
        timeout_seconds=600,
        no_warmup=False,
    ).model_dump()


def decode_ridge_v2(bits: str) -> dict[str, Any]:
    b = [int(char) for char in bits]
    batch_options = [2176, 2240, 2304, 2368, 2432, 2560]
    ubatch_options = [88, 96, 104, 112]
    batch = _choose_one_hot(b, range(0, 6), batch_options, default=2304)
    ubatch = _choose_one_hot(b, range(6, 10), ubatch_options, default=96)

    cache_type_k = "q6_0"
    cache_type_v = "q6_0"
    kv_choice = _choose_one_hot(b, range(10, 12), ["q6_q6", "k6_v4"], default="q6_q6")
    if kv_choice == "k6_v4":
        cache_type_k, cache_type_v = "q6_0", "q4_1"

    return BenchConfig(
        label=f"qpu_ridge2_{bits}",
        ctx_size=16384,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=4,
        threads_batch=4,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        smart_expert_reduction="3,1",
        n_predict=128,
        prompt_key="mars_fact_list",
        prompt=PROMPT_MARS_CONTINUE,
        source="qpu-ridge-v2",
        timeout_seconds=600,
        no_warmup=False,
    ).model_dump()


def decode_moonshot_ridge_v3(bits: str) -> dict[str, Any]:
    b = [int(char) for char in bits]
    batch_options = [2304, 2368, 2432, 2560, 2688]
    ubatch_options = [88, 96, 104, 112]
    parallel_options = [1, 2, 4]
    batch = _choose_one_hot(b, range(0, 5), batch_options, default=2368)
    ubatch = _choose_one_hot(b, range(5, 9), ubatch_options, default=104)
    parallel = _choose_one_hot(b, range(9, 12), parallel_options, default=1)

    extra_args: list[str] = []
    label_parts = [f"b{batch}", f"ub{ubatch}"]
    n_predict = 128
    if parallel > 1:
        extra_args += ["-np", str(parallel), "-ns", str(parallel), "-pps"]
        label_parts.append(f"np{parallel}")
        n_predict = 128
    if b[13]:
        extra_args += ["--ctx-checkpoints", "0"]
        label_parts.append("ckpt0")

    return BenchConfig(
        label=f"qpu_moonshot3_{'_'.join(label_parts)}_{bits}",
        ctx_size=16384,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=4,
        threads_batch=4,
        cache_type_k="q6_0",
        cache_type_v="q6_0",
        smart_expert_reduction="3,1",
        n_predict=n_predict,
        prompt_key="mars_fact_list",
        prompt=PROMPT_MARS_CONTINUE,
        source="qpu-moonshot-ridge-v3",
        timeout_seconds=720,
        no_warmup=False,
        extra_args=extra_args,
    ).model_dump()


def decode_micro_frontier(bits: str) -> dict[str, Any]:
    b = [int(char) for char in bits]
    # Tolerant decoding: if a one-hot group conflicts, choose the safer known center.
    batch = 1792
    if _only(b, 0, [1, 2]):
        batch = 1920
    elif b[1]:
        batch = 2048
    elif _only(b, 2, [0, 1]):
        batch = 2176

    ubatch = 64
    if _only(b, 3, [4, 5]):
        ubatch = 80
    elif b[4]:
        ubatch = 96
    elif _only(b, 5, [3, 4]):
        ubatch = 104

    cache_type_k = "q6_0"
    cache_type_v = "q6_0"
    if b[6] and not b[7]:
        cache_type_k, cache_type_v = "q6_0", "q4_1"
    elif b[7] and not b[6]:
        cache_type_k, cache_type_v = "q4_1", "q4_1"
    elif b[6] and b[7]:
        cache_type_k, cache_type_v = "q6_0", "q4_1"

    return BenchConfig(
        label=f"qpu_micro_{bits}",
        ctx_size=16384,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=4,
        threads_batch=4,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        smart_expert_reduction="3,1",
        n_predict=128,
        prompt_key="mars_capital",
        prompt=PROMPT_MONKEY,
        source="qpu-micro-frontier",
        timeout_seconds=600,
        no_warmup=False,
    ).model_dump()


def decode_impl_flags(bits: str) -> dict[str, Any]:
    b = [int(char) for char in bits]
    extra_args: list[str] = []
    label_parts: list[str] = []
    if b[0]:
        extra_args.append("-nocb")
        label_parts.append("nocb")
    if b[1]:
        extra_args += ["--ctx-checkpoints", "0"]
        label_parts.append("ckpt0")
    if b[2]:
        extra_args += ["-cram", "0"]
        label_parts.append("cram0")
    if b[3]:
        extra_args.append("-mqkv")
        label_parts.append("mqkv")
    if b[4]:
        extra_args.append("-muge")
        label_parts.append("muge")
    if b[5]:
        extra_args.append("-ger")
        label_parts.append("ger")
    if b[6]:
        extra_args.append("-sas")
        label_parts.append("sas")
    if b[7]:
        extra_args += ["-amb", "512"]
        label_parts.append("amb512")
    if b[8]:
        extra_args.append("-khad")
        label_parts.append("khad")
    if b[9]:
        extra_args.append("-vhad")
        label_parts.append("vhad")
    label_tail = "_".join(label_parts) if label_parts else "baseline"
    return BenchConfig(
        label=f"qpu_impl_{label_tail}_{bits}",
        ctx_size=16384,
        batch_size=2304,
        ubatch_size=96,
        threads=4,
        threads_batch=4,
        cache_type_k="q6_0",
        cache_type_v="q6_0",
        smart_expert_reduction="3,1",
        n_predict=128,
        prompt_key="mars_fact_list",
        prompt=PROMPT_MARS_CONTINUE,
        source="qpu-impl-flags",
        timeout_seconds=420,
        no_warmup=False,
        extra_args=extra_args,
    ).model_dump()


def encode_record_lane_config(cfg: dict[str, Any]) -> list[int]:
    bits = [0] * 14
    batch_values = [1792, 2048, 2304, 2560, 2816, 3072, 3328]
    ubatch_values = [80, 96, 112, 128]
    batch = int(cfg.get("batch_size", 2304))
    ubatch = int(cfg.get("ubatch_size", 96))
    ctk = cfg.get("cache_type_k")
    ctv = cfg.get("cache_type_v")

    if batch in batch_values:
        bits[batch_values.index(batch)] = 1
    if ubatch in ubatch_values:
        bits[7 + ubatch_values.index(ubatch)] = 1
    if ctk == "q6_0" and ctv == "q6_0":
        bits[11] = 1
    elif ctk == "q6_0" and ctv == "q4_1":
        bits[12] = 1
    elif ctk == "q4_1" and ctv == "q4_1":
        bits[13] = 1
    return bits


def encode_ridge_v2_config(cfg: dict[str, Any]) -> list[int]:
    bits = [0] * 12
    batch_values = [2176, 2240, 2304, 2368, 2432, 2560]
    ubatch_values = [88, 96, 104, 112]
    batch = int(cfg.get("batch_size", 2304))
    ubatch = int(cfg.get("ubatch_size", 96))
    ctk = cfg.get("cache_type_k")
    ctv = cfg.get("cache_type_v")

    if batch in batch_values:
        bits[batch_values.index(batch)] = 1
    if ubatch in ubatch_values:
        bits[6 + ubatch_values.index(ubatch)] = 1
    if ctk == "q6_0" and ctv == "q6_0":
        bits[10] = 1
    elif ctk == "q6_0" and ctv == "q4_1":
        bits[11] = 1
    return bits


def encode_moonshot_ridge_v3_config(cfg: dict[str, Any]) -> list[int]:
    bits = [0] * 14
    batch_values = [2304, 2368, 2432, 2560, 2688]
    ubatch_values = [88, 96, 104, 112]
    parallel_values = [1, 2, 4]
    batch = int(cfg.get("batch_size", 2368))
    ubatch = int(cfg.get("ubatch_size", 104))
    args = list(cfg.get("extra_args") or [])
    parallel = 1
    if "-np" in args:
        try:
            parallel = int(args[args.index("-np") + 1])
        except (IndexError, ValueError):
            parallel = 1

    if batch in batch_values:
        bits[batch_values.index(batch)] = 1
    if ubatch in ubatch_values:
        bits[5 + ubatch_values.index(ubatch)] = 1
    if parallel in parallel_values:
        bits[9 + parallel_values.index(parallel)] = 1
    bits[12] = 1
    if "--ctx-checkpoints" in args and "0" in args:
        bits[13] = 1
    return bits


def qubo_energy(bits: list[int], payload: dict[str, Any]) -> float:
    q = payload["qubo"]
    total = float(payload.get("offset", 0.0))
    for i, xi in enumerate(bits):
        for j, xj in enumerate(bits):
            total += float(xi) * float(q[i][j]) * float(xj)
    return total


def encode_micro_frontier_config(cfg: dict[str, Any]) -> list[int]:
    bits = [0] * 8
    batch = int(cfg.get("batch_size", 1792))
    ubatch = int(cfg.get("ubatch_size", 64))
    ctk = cfg.get("cache_type_k")
    ctv = cfg.get("cache_type_v")

    if batch == 1920:
        bits[0] = 1
    elif batch == 2048:
        bits[1] = 1
    elif batch == 2176:
        bits[2] = 1

    if ubatch == 80:
        bits[3] = 1
    elif ubatch == 96:
        bits[4] = 1
    elif ubatch == 104:
        bits[5] = 1

    if ctk == "q6_0" and ctv == "q4_1":
        bits[6] = 1
    elif ctk == "q4_1" and ctv == "q4_1":
        bits[7] = 1
    return bits


def encode_impl_flags_config(cfg: dict[str, Any]) -> list[int]:
    args = list(cfg.get("extra_args") or [])
    joined = " ".join(args)
    return [
        1 if "-nocb" in args or "--no-cont-batching" in args else 0,
        1 if "--ctx-checkpoints" in args and "0" in args else 0,
        1 if "-cram" in args and "0" in args else 0,
        1 if "-mqkv" in args or "--merge-qkv" in args else 0,
        1 if "-muge" in args or "--merge-up-gate-experts" in args else 0,
        1 if "-ger" in args or "--grouped-expert-routing" in args else 0,
        1 if "-sas" in args or "--scheduler_async" in args else 0,
        1 if "-amb 512" in joined or "--attention-max-batch 512" in joined else 0,
        1 if "-khad" in args or "--k-cache-hadamard" in args else 0,
        1 if "-vhad" in args or "--v-cache-hadamard" in args else 0,
    ]


def _evaluate_angles(payload: dict[str, Any], gamma: float, beta: float) -> dict[str, Any]:
    circuit = _qaoa_circuit_from_qubo(payload, gamma=gamma, beta=beta)
    circuit.remove_final_measurements(inplace=True)
    state = Statevector.from_instruction(circuit)
    probs = state.probabilities_dict()
    expected = 0.0
    best_bitstring = None
    best_prob = -1.0
    for bitstring, prob in probs.items():
        energy = qubo_energy([int(char) for char in bitstring], payload)
        expected += float(prob) * energy
        if prob > best_prob:
            best_prob = float(prob)
            best_bitstring = bitstring
    return {
        "gamma": gamma,
        "beta": beta,
        "expected_energy": expected,
        "top_bitstring": best_bitstring,
        "top_probability": best_prob,
        "entropy": -sum(float(p) * math.log(float(p) + 1e-12) for p in probs.values()),
    }


def _only(bits: list[int], index: int, others: list[int]) -> bool:
    return bool(bits[index]) and not any(bits[other] for other in others)


def _choose_one_hot(bits: list[int], indices: range, values: list[Any], default: Any) -> Any:
    chosen = [pos for pos, bit_index in enumerate(indices) if bits[bit_index]]
    if len(chosen) != 1:
        return default
    return values[chosen[0]]
