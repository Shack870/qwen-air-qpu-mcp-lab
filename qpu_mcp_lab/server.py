from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import db
from .bench import import_many, run_config
from .objective import compare_runs, score_run
from .optimizer import build_qubo, propose_classical_candidates, sample_qubo_local
from .quantum import (
    credential_status,
    estimate_runtime_cost,
    get_job_result,
    list_backends,
    sample_candidates,
    submit_qaoa_job,
)
from .qpu_strategy import decode_job_candidates, submit_micro_frontier_job, sweep_angles

mcp = FastMCP("qwen-air-qpu-lab")


@mcp.tool()
def bench_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run a validated llama.cpp benchmark config and log it to SQLite."""

    return run_config(config)


@mcp.tool()
def bench_get_best_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Return the best successful runs by generation tokens/sec."""

    return db.best_runs(limit=limit)


@mcp.tool()
def bench_import_summaries(paths: list[str]) -> dict[str, Any]:
    """Import existing TSV summaries from the previous MacBook experiments."""

    count = import_many([Path(path) for path in paths])
    return {"imported_rows": count, "total_rows": db.count_runs()}


@mcp.tool()
def objective_score_run(run: dict[str, Any]) -> dict[str, Any]:
    """Score a run using the current stability-aware objective."""

    score, components = score_run(run)
    return {"score": score, "components": components}


@mcp.tool()
def objective_compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two run dictionaries under the current objective."""

    return compare_runs(a, b)


@mcp.tool()
def optimizer_propose_classical_candidates(limit: int = 12) -> list[dict[str, Any]]:
    """Propose safe next benchmark configs from the classical optimizer."""

    return propose_classical_candidates(limit=limit)


@mcp.tool()
def optimizer_build_qubo(limit_runs: int = 200) -> dict[str, Any]:
    """Build a compact QUBO surrogate from logged benchmark data."""

    return build_qubo(limit_runs=limit_runs)


@mcp.tool()
def optimizer_sample_qubo_local(qubo_payload: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    """Sample QUBO candidates locally before spending IBM QPU time."""

    return sample_qubo_local(qubo_payload, top_k=top_k)


@mcp.tool()
def quantum_credential_status() -> dict[str, Any]:
    """Check whether IBM credentials are configured without exposing secrets."""

    return credential_status()


@mcp.tool()
def quantum_list_backends(operational_only: bool = True, simulator: bool | None = None) -> dict[str, Any]:
    """List IBM Quantum backends if credentials are configured."""

    return list_backends(operational_only=operational_only, simulator=simulator)


@mcp.tool()
def quantum_estimate_runtime_cost(num_qubits: int, shots: int = 1024, circuit_count: int = 1) -> dict[str, Any]:
    """Return a local preflight estimate shape for an IBM runtime job."""

    return estimate_runtime_cost(num_qubits=num_qubits, shots=shots, circuit_count=circuit_count)


@mcp.tool()
def quantum_sample_candidates(qubo_payload: dict[str, Any], backend: str = "local", top_k: int = 8) -> dict[str, Any]:
    """Sample candidate configs from a QUBO using local backend for now."""

    return sample_candidates(qubo_payload=qubo_payload, backend=backend, top_k=top_k)


@mcp.tool()
def quantum_submit_qaoa_job(
    qubo_payload: dict[str, Any],
    backend_name: str,
    shots: int = 256,
    allow_real_qpu: bool = False,
    gamma: float = 1.0,
    beta: float = 0.7,
) -> dict[str, Any]:
    """Guarded QAOA submission boundary for IBM Quantum smoke tests."""

    return submit_qaoa_job(
        qubo_payload=qubo_payload,
        backend_name=backend_name,
        shots=shots,
        allow_real_qpu=allow_real_qpu,
        gamma=gamma,
        beta=beta,
    )


@mcp.tool()
def quantum_get_job_result(job_id: str) -> dict[str, Any]:
    """Fetch a known IBM Runtime job result without exposing credentials."""

    return get_job_result(job_id)


@mcp.tool()
def quantum_sweep_qaoa_angles(limit: int = 10) -> list[dict[str, Any]]:
    """Locally simulate QAOA angles for the current frontier before using QPU time."""

    return sweep_angles()[:limit]


@mcp.tool()
def quantum_submit_micro_frontier_job(
    backend_name: str,
    shots: int = 256,
    allow_real_qpu: bool = False,
    auto_angle: bool = True,
    gamma: float = 1.0,
    beta: float = 0.7,
) -> dict[str, Any]:
    """Submit the current frontier QUBO to IBM, with local angle preselection."""

    return submit_micro_frontier_job(
        backend_name=backend_name,
        shots=shots,
        gamma=gamma,
        beta=beta,
        auto_angle=auto_angle,
        allow_real_qpu=allow_real_qpu,
    )


@mcp.tool()
def quantum_decode_job_candidates(job_id: str, top_k: int = 12) -> dict[str, Any]:
    """Decode IBM bitstring samples into benchmark configs and persist them."""

    return decode_job_candidates(job_id, top_k=top_k)


def main() -> None:
    db.init_db()
    mcp.run()


if __name__ == "__main__":
    main()
