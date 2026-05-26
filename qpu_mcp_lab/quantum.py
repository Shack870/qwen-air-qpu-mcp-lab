from __future__ import annotations

import os
import subprocess
from typing import Any

from . import db
from .optimizer import sample_qubo_local

KEYCHAIN_TOKEN_SERVICE = "ibm_quantum_api_key"
KEYCHAIN_INSTANCE_SERVICE = "ibm_quantum_instance_crn"


def credential_status() -> dict[str, Any]:
    token = _get_token()
    instance = _get_instance()
    return {
        "has_token": bool(token),
        "has_instance": bool(instance),
        "token_source": _source_for("IBM_QUANTUM_API_KEY", KEYCHAIN_TOKEN_SERVICE),
        "instance_source": _source_for("IBM_QUANTUM_INSTANCE", KEYCHAIN_INSTANCE_SERVICE),
    }


def list_backends(operational_only: bool = True, simulator: bool | None = None) -> dict[str, Any]:
    token = _get_token()
    if not token:
        return {"ok": False, "error": "IBM Quantum API key is not configured", "backends": []}
    try:
        service = _runtime_service(token=token, instance=_get_instance())
        kwargs: dict[str, Any] = {}
        if operational_only:
            kwargs["operational"] = True
        if simulator is not None:
            kwargs["simulator"] = simulator
        backends = service.backends(**kwargs)
        rows = []
        for backend in backends:
            rows.append(
                {
                    "name": getattr(backend, "name", None) or str(backend),
                    "num_qubits": getattr(backend, "num_qubits", None),
                    "simulator": bool(getattr(backend, "simulator", False)),
                    "status": _safe_status(backend),
                }
            )
        return {"ok": True, "backends": rows}
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc), "backends": []}


def estimate_runtime_cost(num_qubits: int, shots: int = 1024, circuit_count: int = 1) -> dict[str, Any]:
    return {
        "num_qubits": num_qubits,
        "shots": shots,
        "circuit_count": circuit_count,
        "qpu_time_estimate_seconds": None,
        "note": (
            "This harness cannot know the real IBM plan cost or queue time locally. "
            "Use this as a preflight only, and keep Open Plan jobs tiny."
        ),
    }


def sample_candidates(qubo_payload: dict[str, Any], backend: str = "local", top_k: int = 8) -> dict[str, Any]:
    if backend != "local":
        return {
            "ok": False,
            "error": "Real QPU submission is intentionally gated. Use submit_qaoa_job with allow_real_qpu=true.",
        }
    return {"ok": True, "backend": "local-enumeration", "samples": sample_qubo_local(qubo_payload, top_k=top_k)}


def submit_qaoa_job(
    qubo_payload: dict[str, Any],
    backend_name: str,
    shots: int = 256,
    allow_real_qpu: bool = False,
    gamma: float = 1.0,
    beta: float = 0.7,
) -> dict[str, Any]:
    """Submit a tiny fixed-angle p=1 QAOA-style sampler job.

    The adapter is intentionally conservative: it validates credentials and job
    shape, then returns a dry-run payload unless the caller explicitly allows
    a real QPU submission.
    """

    n = len(qubo_payload.get("variables", []))
    if n <= 0 or n > 16:
        return {"ok": False, "error": "QPU smoke tests are capped at 1..16 binary variables"}
    if shots < 1 or shots > 1024:
        return {"ok": False, "error": "shots must be between 1 and 1024 for this harness"}
    if not allow_real_qpu:
        return {
            "ok": True,
            "dry_run": True,
            "backend_name": backend_name,
            "shots": shots,
            "num_qubits": n,
            "gamma": gamma,
            "beta": beta,
            "message": "Dry run only. Pass allow_real_qpu=true after credential and budget checks.",
        }
    token = _get_token()
    if not token:
        return {"ok": False, "error": "IBM Quantum API key is not configured"}
    try:
        service = _runtime_service(token=token, instance=_get_instance())
        backend = service.backend(backend_name)
        circuit = _qaoa_circuit_from_qubo(qubo_payload, gamma=gamma, beta=beta)
        try:
            from qiskit import transpile

            circuit = transpile(circuit, backend=backend, optimization_level=1)
        except Exception:
            pass
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=backend)
        job = sampler.run([circuit], shots=shots)
        job_id = job.job_id()
        db.upsert_quantum_job(
            job_id=job_id,
            payload=qubo_payload,
            backend_name=backend_name,
            shots=shots,
            num_qubits=n,
            gamma=gamma,
            beta=beta,
            status="SUBMITTED",
        )
        return {
            "ok": True,
            "dry_run": False,
            "job_id": job_id,
            "backend_name": backend_name,
            "shots": shots,
            "num_qubits": n,
            "gamma": gamma,
            "beta": beta,
        }
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}


def submit_qaoa_angle_sweep_job(
    qubo_payload: dict[str, Any],
    backend_name: str,
    angles: list[dict[str, float]],
    shots_per_angle: int = 64,
    allow_real_qpu: bool = False,
) -> dict[str, Any]:
    """Submit several fixed-angle QAOA circuits as one sampler workload.

    This keeps QPU use compact: one queue item explores several gamma/beta
    settings, then the decoder can aggregate all sampled bitstrings.
    """

    n = len(qubo_payload.get("variables", []))
    if n <= 0 or n > 16:
        return {"ok": False, "error": "QPU smoke tests are capped at 1..16 binary variables"}
    if not angles or len(angles) > 12:
        return {"ok": False, "error": "angle sweep must contain 1..12 angle pairs"}
    if shots_per_angle < 1 or shots_per_angle > 512:
        return {"ok": False, "error": "shots_per_angle must be between 1 and 512"}
    clean_angles = [
        {"gamma": float(row["gamma"]), "beta": float(row["beta"])}
        for row in angles
        if "gamma" in row and "beta" in row
    ]
    if len(clean_angles) != len(angles):
        return {"ok": False, "error": "each angle must include gamma and beta"}
    payload = dict(qubo_payload)
    payload["angle_sweep"] = clean_angles
    payload["counts_format"] = "by_pub_index"
    if not allow_real_qpu:
        return {
            "ok": True,
            "dry_run": True,
            "backend_name": backend_name,
            "shots_per_angle": shots_per_angle,
            "total_shots": shots_per_angle * len(clean_angles),
            "num_qubits": n,
            "angles": clean_angles,
            "message": "Dry run only. Pass allow_real_qpu=true after credential and budget checks.",
        }
    token = _get_token()
    if not token:
        return {"ok": False, "error": "IBM Quantum API key is not configured"}
    try:
        service = _runtime_service(token=token, instance=_get_instance())
        backend = service.backend(backend_name)
        circuits = [
            _qaoa_circuit_from_qubo(qubo_payload, gamma=row["gamma"], beta=row["beta"])
            for row in clean_angles
        ]
        for index, circuit in enumerate(circuits):
            circuit.metadata = {"angle_index": index, **clean_angles[index]}
        try:
            from qiskit import transpile

            circuits = transpile(circuits, backend=backend, optimization_level=1)
        except Exception:
            pass
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=backend)
        job = sampler.run(circuits, shots=shots_per_angle)
        job_id = job.job_id()
        db.upsert_quantum_job(
            job_id=job_id,
            payload=payload,
            backend_name=backend_name,
            shots=shots_per_angle * len(clean_angles),
            num_qubits=n,
            gamma=None,
            beta=None,
            status="SUBMITTED",
            notes=f"angle_sweep; {len(clean_angles)} circuits x {shots_per_angle} shots",
        )
        return {
            "ok": True,
            "dry_run": False,
            "job_id": job_id,
            "backend_name": backend_name,
            "shots_per_angle": shots_per_angle,
            "total_shots": shots_per_angle * len(clean_angles),
            "num_qubits": n,
            "angles": clean_angles,
        }
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}


def get_job_result(job_id: str, refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = cached_job_result(job_id)
        if cached.get("ok") and cached.get("done"):
            return cached
    token = _get_token()
    if not token:
        return {"ok": False, "error": "IBM Quantum API key is not configured"}
    try:
        service = _runtime_service(token=token, instance=_get_instance())
        job = service.job(job_id)
        status = str(job.status())
        if "DONE" not in status.upper() and "COMPLETED" not in status.upper():
            db.upsert_quantum_job(job_id=job_id, payload={}, status=status)
            return {"ok": True, "done": False, "job_id": job_id, "status": status}
        result = job.result()
        counts = _extract_counts(result)
        db.upsert_quantum_job(job_id=job_id, payload={}, status=status, counts=counts)
        return {
            "ok": True,
            "done": True,
            "job_id": job_id,
            "status": status,
            "counts": counts,
        }
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}


def cached_job_result(job_id: str) -> dict[str, Any]:
    job = db.get_quantum_job(job_id)
    if not job:
        return {"ok": False, "error": "job is not in local database", "job_id": job_id}
    counts = job.get("counts")
    status = str(job.get("status") or "")
    done = bool(counts) and ("DONE" in status.upper() or "COMPLETED" in status.upper())
    return {
        "ok": True,
        "done": done,
        "cached": True,
        "job_id": job_id,
        "status": status,
        "counts": counts,
    }


def _qaoa_circuit_from_qubo(qubo_payload: dict[str, Any], gamma: float, beta: float):
    from qiskit import QuantumCircuit

    q = qubo_payload.get("qubo")
    n = len(qubo_payload.get("variables", []))
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    # Convert symmetric QUBO x^T Q x into Ising h_i Z_i + J_ij Z_i Z_j terms.
    h = [0.0 for _ in range(n)]
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        h[i] += -float(q[i][i]) / 2.0
        for j in range(i + 1, n):
            qij = float(q[i][j])
            if abs(qij) < 1e-12:
                continue
            h[i] += -qij / 2.0
            h[j] += -qij / 2.0
            pairs.append((i, j, qij / 2.0))

    for i, hi in enumerate(h):
        if abs(hi) > 1e-12:
            qc.rz(2.0 * gamma * hi, i)
    for i, j, Jij in pairs:
        qc.rzz(2.0 * gamma * Jij, i, j)
    for i in range(n):
        qc.rx(2.0 * beta, i)
    qc.measure(range(n), range(n))
    return qc


def _extract_counts(result: Any) -> dict[str, int] | None:
    try:
        if len(result) > 1:
            rows: dict[str, dict[str, int]] = {}
            for index, pub_result in enumerate(result):
                counts = _extract_counts_from_pub(pub_result)
                if counts is not None:
                    rows[str(index)] = counts
            if rows:
                return rows  # type: ignore[return-value]
    except Exception:
        pass
    try:
        counts = _extract_counts_from_pub(result[0])
        if counts is not None:
            return counts
    except Exception:
        pass
    try:
        first = result[0]
        data = first.data
        for name in ("c", "meas", "measure"):
            register = getattr(data, name, None)
            if register is not None and hasattr(register, "get_counts"):
                counts = register.get_counts()
                return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        pass
    try:
        counts = result.get_counts()
        return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return None


def _extract_counts_from_pub(pub_result: Any) -> dict[str, int] | None:
    data = getattr(pub_result, "data", None)
    if data is None:
        return None
    for name in ("c", "meas", "measure"):
        register = getattr(data, name, None)
        if register is not None and hasattr(register, "get_counts"):
            counts = register.get_counts()
            return {str(k): int(v) for k, v in counts.items()}
    return None


def _runtime_service(token: str, instance: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    attempts = [
        {"token": token, "instance": instance},
        {"channel": "ibm_quantum_platform", "token": token},
        {"channel": "ibm_quantum_platform", "token": token, "instance": instance},
        # Kept as a compatibility fallback for older qiskit-ibm-runtime builds.
        {"channel": "ibm_quantum", "token": token},
        {"channel": "ibm_quantum", "token": token, "instance": instance},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        kwargs = {k: v for k, v in kwargs.items() if v not in (None, "")}
        try:
            return QiskitRuntimeService(**kwargs)
        except Exception as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _get_token() -> str | None:
    return os.environ.get("IBM_QUANTUM_API_KEY") or _keychain_read(KEYCHAIN_TOKEN_SERVICE)


def _get_instance() -> str | None:
    return os.environ.get("IBM_QUANTUM_INSTANCE") or _keychain_read(KEYCHAIN_INSTANCE_SERVICE)


def _keychain_read(service: str) -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", service],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _source_for(env_name: str, service: str) -> str | None:
    if os.environ.get(env_name):
        return "environment"
    if _keychain_read(service):
        return "macOS keychain"
    return None


def _safe_status(backend: Any) -> dict[str, Any] | None:
    try:
        status = backend.status()
    except Exception:
        return None
    return {
        "operational": getattr(status, "operational", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "status_msg": getattr(status, "status_msg", None),
    }


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    token = _get_token()
    if token:
        text = text.replace(token, "[redacted-token]")
    return text
