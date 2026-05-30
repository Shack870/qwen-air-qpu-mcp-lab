from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DB = ROOT / "data" / "qpu_lab.sqlite"
PUBLIC_DB = ROOT / "paper" / "data" / "qpu_lab_public.sqlite"
HOME_RE = re.compile(r"/Users/[^/\s\"']+")
SECRET_RE = re.compile(r"(?i)(token|api[_-]?key|password|secret)([\"'\\s:=]+)([^\"'\\s,}]+)")


def redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = HOME_RE.sub("$HOME", value)
    value = SECRET_RE.sub(r"\1\2[REDACTED]", value)
    return value


def redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_json(value: str | None) -> str:
    if not value:
        return "{}"
    try:
        return json.dumps(redact_obj(json.loads(value)), sort_keys=True)
    except Exception:
        return json.dumps({"raw": redact_text(value)}, sort_keys=True)


def main() -> None:
    if not RAW_DB.exists():
        raise SystemExit(f"missing source DB: {RAW_DB}")
    PUBLIC_DB.parent.mkdir(parents=True, exist_ok=True)
    if PUBLIC_DB.exists():
        PUBLIC_DB.unlink()

    src = sqlite3.connect(RAW_DB)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(PUBLIC_DB)
    try:
        dst.executescript(
            """
            PRAGMA journal_mode=OFF;
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                external_key TEXT,
                config_json TEXT NOT NULL,
                prompt_key TEXT,
                prompt_hash TEXT,
                model_fingerprint TEXT,
                llama_commit TEXT,
                exit_code INTEGER,
                pp_tps REAL,
                gen_tps REAL,
                total_ms REAL,
                peak_rss_bytes INTEGER,
                quality_flag TEXT,
                notes TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE scores (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                score REAL NOT NULL,
                components_json TEXT NOT NULL
            );
            CREATE TABLE quantum_jobs (
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
            CREATE TABLE quantum_candidates (
                id INTEGER PRIMARY KEY,
                job_id TEXT NOT NULL,
                bitstring TEXT NOT NULL,
                orientation TEXT NOT NULL,
                shot_count INTEGER NOT NULL,
                energy REAL,
                config_json TEXT NOT NULL,
                run_id INTEGER
            );
            CREATE TABLE export_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        run_rows = src.execute(
            """
            SELECT id, ts, source, label, external_key, config_json, prompt_key,
                   prompt_hash, model_fingerprint, llama_commit, exit_code,
                   pp_tps, gen_tps, total_ms, peak_rss_bytes, quality_flag,
                   notes, metrics_json
            FROM runs
            ORDER BY id
            """
        ).fetchall()
        dst.executemany(
            """
            INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["id"],
                    r["ts"],
                    r["source"],
                    r["label"],
                    None,
                    redact_json(r["config_json"]),
                    redact_text(r["prompt_key"]),
                    redact_text(r["prompt_hash"]),
                    redact_text(r["model_fingerprint"]),
                    redact_text(r["llama_commit"]),
                    r["exit_code"],
                    r["pp_tps"],
                    r["gen_tps"],
                    r["total_ms"],
                    r["peak_rss_bytes"],
                    redact_text(r["quality_flag"]),
                    redact_text(r["notes"]),
                    redact_json(r["metrics_json"]),
                )
                for r in run_rows
            ],
        )

        score_rows = src.execute(
            "SELECT id, run_id, ts, score, components_json FROM scores ORDER BY id"
        ).fetchall()
        dst.executemany(
            "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
            [
                (
                    r["id"],
                    r["run_id"],
                    r["ts"],
                    r["score"],
                    redact_json(r["components_json"]),
                )
                for r in score_rows
            ],
        )

        job_rows = src.execute(
            """
            SELECT job_id, ts, backend_name, shots, num_qubits, gamma, beta,
                   status, payload_json, counts_json, notes
            FROM quantum_jobs
            ORDER BY ts, job_id
            """
        ).fetchall()
        dst.executemany(
            "INSERT INTO quantum_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    redact_text(r["job_id"]),
                    r["ts"],
                    redact_text(r["backend_name"]),
                    r["shots"],
                    r["num_qubits"],
                    r["gamma"],
                    r["beta"],
                    redact_text(r["status"]),
                    redact_json(r["payload_json"]),
                    redact_json(r["counts_json"]),
                    redact_text(r["notes"]),
                )
                for r in job_rows
            ],
        )

        candidate_rows = src.execute(
            """
            SELECT id, job_id, bitstring, orientation, shot_count, energy,
                   config_json, run_id
            FROM quantum_candidates
            ORDER BY id
            """
        ).fetchall()
        dst.executemany(
            "INSERT INTO quantum_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["id"],
                    redact_text(r["job_id"]),
                    redact_text(r["bitstring"]),
                    redact_text(r["orientation"]),
                    r["shot_count"],
                    r["energy"],
                    redact_json(r["config_json"]),
                    r["run_id"],
                )
                for r in candidate_rows
            ],
        )

        dst.executemany(
            "INSERT INTO export_metadata VALUES (?, ?)",
            [
                ("source", "sanitized export from data/qpu_lab.sqlite"),
                ("runs", str(len(run_rows))),
                ("scores", str(len(score_rows))),
                ("quantum_jobs", str(len(job_rows))),
                ("quantum_candidates", str(len(candidate_rows))),
                ("excluded_fields", "command_json, model_path, llama_bin, log_path, stdout_tail"),
            ],
        )
        dst.executescript(
            """
            CREATE INDEX idx_runs_gen_tps ON runs(gen_tps);
            CREATE INDEX idx_runs_source ON runs(source);
            CREATE INDEX idx_candidates_job ON quantum_candidates(job_id);
            VACUUM;
            """
        )
        print(f"wrote {PUBLIC_DB}")
        print(f"runs={len(run_rows)} scores={len(score_rows)} quantum_jobs={len(job_rows)} quantum_candidates={len(candidate_rows)}")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
