from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH, ensure_dirs


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                external_key TEXT UNIQUE,
                config_json TEXT NOT NULL,
                command_json TEXT NOT NULL DEFAULT '[]',
                prompt_key TEXT,
                prompt_hash TEXT,
                model_path TEXT,
                model_fingerprint TEXT,
                llama_bin TEXT,
                llama_commit TEXT,
                exit_code INTEGER,
                pp_tps REAL,
                gen_tps REAL,
                total_ms REAL,
                peak_rss_bytes INTEGER,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                log_path TEXT,
                stdout_tail TEXT,
                quality_flag TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                ts TEXT NOT NULL,
                score REAL NOT NULL,
                components_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_gen_tps ON runs(gen_tps DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_source ON runs(source);
            CREATE INDEX IF NOT EXISTS idx_scores_run_id ON scores(run_id);

            CREATE TABLE IF NOT EXISTS quantum_jobs (
                job_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                backend_name TEXT,
                shots INTEGER,
                num_qubits INTEGER,
                gamma REAL,
                beta REAL,
                status TEXT,
                payload_json TEXT NOT NULL,
                counts_json TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS quantum_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES quantum_jobs(job_id) ON DELETE CASCADE,
                bitstring TEXT NOT NULL,
                orientation TEXT NOT NULL,
                shot_count INTEGER NOT NULL,
                energy REAL,
                config_json TEXT NOT NULL,
                run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
                UNIQUE(job_id, bitstring, orientation, config_json)
            );

            CREATE INDEX IF NOT EXISTS idx_quantum_candidates_job_id ON quantum_candidates(job_id);
            """
        )
        _ensure_column(con, "runs", "metrics_json", "TEXT NOT NULL DEFAULT '{}'")


def _ensure_column(con: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in rows):
        return
    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in list(d):
        if key in d and isinstance(d[key], str):
            if key.endswith("_json"):
                try:
                    d[key.removesuffix("_json")] = json.loads(d[key])
                except json.JSONDecodeError:
                    pass
    return d


def insert_run(run: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    payload = dict(run)
    payload.setdefault("ts", now_iso())
    payload.setdefault("source", "unknown")
    payload.setdefault("label", "run")
    payload.setdefault("config_json", "{}")
    payload.setdefault("command_json", "[]")
    fields = [
        "ts",
        "source",
        "label",
        "external_key",
        "config_json",
        "command_json",
        "prompt_key",
        "prompt_hash",
        "model_path",
        "model_fingerprint",
        "llama_bin",
        "llama_commit",
        "exit_code",
        "pp_tps",
        "gen_tps",
        "total_ms",
        "peak_rss_bytes",
        "metrics_json",
        "log_path",
        "stdout_tail",
        "quality_flag",
        "notes",
    ]
    placeholders = ",".join("?" for _ in fields)
    sql = f"INSERT OR IGNORE INTO runs ({','.join(fields)}) VALUES ({placeholders})"
    values = [payload.get(field) for field in fields]
    with connect(db_path) as con:
        cur = con.execute(sql, values)
        if cur.lastrowid:
            return int(cur.lastrowid)
        external_key = payload.get("external_key")
        if external_key:
            row = con.execute("SELECT id FROM runs WHERE external_key=?", (external_key,)).fetchone()
            if row:
                return int(row["id"])
        raise RuntimeError("Run insert was ignored but no existing row could be found")


def insert_score(run_id: int, score: float, components: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as con:
        cur = con.execute(
            "INSERT INTO scores (run_id, ts, score, components_json) VALUES (?, ?, ?, ?)",
            (run_id, now_iso(), float(score), json.dumps(components, sort_keys=True)),
        )
        return int(cur.lastrowid)


def list_runs(limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT runs.*, scores.score AS latest_score
            FROM runs
            LEFT JOIN scores ON scores.id = (
                SELECT id FROM scores s WHERE s.run_id = runs.id ORDER BY id DESC LIMIT 1
            )
            ORDER BY runs.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def best_runs(limit: int = 10, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT runs.*, scores.score AS latest_score
            FROM runs
            LEFT JOIN scores ON scores.id = (
                SELECT id FROM scores s WHERE s.run_id = runs.id ORDER BY id DESC LIMIT 1
            )
            WHERE runs.gen_tps IS NOT NULL AND COALESCE(runs.exit_code, 0) = 0
            ORDER BY runs.gen_tps DESC, runs.pp_tps DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def all_successful_runs(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT * FROM runs
            WHERE gen_tps IS NOT NULL AND COALESCE(exit_code, 0) = 0
            ORDER BY id ASC
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def count_runs(db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as con:
        row = con.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return int(row["n"])


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def insert_many_runs(runs: Iterable[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    count = 0
    for run in runs:
        insert_run(run, db_path)
        count += 1
    return count


def upsert_quantum_job(
    job_id: str,
    payload: dict[str, Any],
    backend_name: str | None = None,
    shots: int | None = None,
    num_qubits: int | None = None,
    gamma: float | None = None,
    beta: float | None = None,
    status: str | None = None,
    counts: dict[str, int] | None = None,
    notes: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    init_db(db_path)
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO quantum_jobs (
                job_id, ts, backend_name, shots, num_qubits, gamma, beta,
                status, payload_json, counts_json, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                backend_name = COALESCE(excluded.backend_name, quantum_jobs.backend_name),
                shots = COALESCE(excluded.shots, quantum_jobs.shots),
                num_qubits = COALESCE(excluded.num_qubits, quantum_jobs.num_qubits),
                gamma = COALESCE(excluded.gamma, quantum_jobs.gamma),
                beta = COALESCE(excluded.beta, quantum_jobs.beta),
                status = COALESCE(excluded.status, quantum_jobs.status),
                payload_json = CASE
                    WHEN excluded.payload_json != '{}' THEN excluded.payload_json
                    ELSE quantum_jobs.payload_json
                END,
                counts_json = COALESCE(excluded.counts_json, quantum_jobs.counts_json),
                notes = COALESCE(excluded.notes, quantum_jobs.notes)
            """,
            (
                job_id,
                now_iso(),
                backend_name,
                shots,
                num_qubits,
                gamma,
                beta,
                status,
                json_dumps(payload),
                json_dumps(counts) if counts is not None else None,
                notes,
            ),
        )


def get_quantum_job(job_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM quantum_jobs WHERE job_id=?", (job_id,)).fetchone()
    return row_to_dict(row) if row else None


def insert_quantum_candidate(
    job_id: str,
    bitstring: str,
    orientation: str,
    shot_count: int,
    config: dict[str, Any],
    energy: float | None = None,
    db_path: Path = DB_PATH,
) -> int:
    init_db(db_path)
    with connect(db_path) as con:
        cur = con.execute(
            """
            INSERT OR IGNORE INTO quantum_candidates (
                job_id, bitstring, orientation, shot_count, energy, config_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, bitstring, orientation, shot_count, energy, json_dumps(config)),
        )
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = con.execute(
            """
            SELECT id FROM quantum_candidates
            WHERE job_id=? AND bitstring=? AND orientation=? AND config_json=?
            """,
            (job_id, bitstring, orientation, json_dumps(config)),
        ).fetchone()
        return int(row["id"]) if row else 0


def list_quantum_jobs(limit: int = 10, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as con:
        rows = con.execute("SELECT * FROM quantum_jobs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [row_to_dict(row) for row in rows]
