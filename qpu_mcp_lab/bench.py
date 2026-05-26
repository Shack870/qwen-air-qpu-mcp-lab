from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from . import db
from .config import DEFAULT_LLAMA_REPO, LOG_DIR, PROMPTS, ensure_dirs, llama_bin, model_path
from .objective import score_run

TIMING_PATTERNS = {
    "pp_tps": re.compile(
        r"prompt eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\s*\)",
        re.I,
    ),
    "gen_tps": re.compile(
        r"(?<!prompt )eval time\s*=.*?\(\s*[0-9.]+\s+ms per token,\s*([0-9.]+)\s+tokens per second\s*\)",
        re.I,
    ),
    "total_ms": re.compile(r"total time\s*=\s*([0-9.]+)\s*ms", re.I),
    "rss": re.compile(r"([0-9]+)\s+maximum resident set size", re.I),
    "page_reclaims": re.compile(r"([0-9]+)\s+page reclaims", re.I),
    "page_faults": re.compile(r"([0-9]+)\s+page faults", re.I),
    "swaps": re.compile(r"([0-9]+)\s+swaps", re.I),
    "voluntary_context_switches": re.compile(r"([0-9]+)\s+voluntary context switches", re.I),
    "involuntary_context_switches": re.compile(r"([0-9]+)\s+involuntary context switches", re.I),
    "user_seconds": re.compile(r"([0-9.]+)\s+user", re.I),
    "sys_seconds": re.compile(r"([0-9.]+)\s+sys", re.I),
}


class BenchConfig(BaseModel):
    label: str = "qpu_lab_run"
    ctx_size: int = Field(default=16384, ge=128, le=32768)
    batch_size: int = Field(default=1792, ge=1, le=8192)
    ubatch_size: int = Field(default=96, ge=1, le=1024)
    threads: int = Field(default=4, ge=1, le=8)
    threads_batch: int = Field(default=4, ge=1, le=8)
    cache_type_k: str = "q6_0"
    cache_type_v: str = "q6_0"
    cache_type_k_first: str | None = None
    cache_type_k_last: str | None = None
    cache_type_v_first: str | None = None
    cache_type_v_last: str | None = None
    flash_attn: bool = True
    smart_expert_reduction: str | None = "3,1"
    n_predict: int = Field(default=128, ge=1, le=1024)
    temp: float = Field(default=0.0, ge=0.0, le=2.0)
    prompt_key: str = "mars_capital"
    prompt: str | None = None
    ignore_eos: bool = True
    no_display_prompt: bool = True
    no_warmup: bool = False
    prewarm_model: bool = False
    prewarm_block_size: str = "32m"
    timeout_seconds: int = Field(default=420, ge=30, le=7200)
    env_veclib_threads: int = Field(default=1, ge=1, le=8)
    env_omp_wait_policy: str = "ACTIVE"
    env_omp_dynamic: str = "FALSE"
    env_omp_proc_bind: str | None = None
    env_omp_places: str | None = None
    env_malloc_nano_zone: str | None = None
    env_ser_full_first: int | None = Field(default=None, ge=0, le=48)
    env_ser_full_last: int | None = Field(default=None, ge=0, le=48)
    env_ser_full_ranges: str | None = None
    env_ser_cheap_ranges: str | None = None
    env_ser_cheap_min: int | None = Field(default=None, ge=0, le=8)
    env_ser_cheap_thresh: float | None = Field(default=None, gt=0.0, le=16.0)
    env_ser_cheap_max_ntokens: int | None = Field(default=None, ge=0, le=8192)
    env_ser_cheap2_ranges: str | None = None
    env_ser_cheap2_min: int | None = Field(default=None, ge=0, le=8)
    env_ser_cheap2_thresh: float | None = Field(default=None, gt=0.0, le=16.0)
    env_ser_cheap2_max_ntokens: int | None = Field(default=None, ge=0, le=8192)
    env_ser_adaptive_ranges: str | None = None
    env_ser_adaptive_third_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    llama_bin_override: str | None = None
    model_path_override: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    source: str = "mcp-bench"

    @field_validator("cache_type_k", "cache_type_v")
    @classmethod
    def cache_type_allowed(cls, value: str) -> str:
        allowed = {"f16", "q8_0", "q8_KV", "q6_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"}
        if value not in allowed:
            raise ValueError(f"cache type must be one of {sorted(allowed)}")
        return value

    @field_validator("cache_type_k_first", "cache_type_k_last", "cache_type_v_first", "cache_type_v_last")
    @classmethod
    def cache_type_layer_override_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"(f16|q8_0|q8_KV|q6_0|q4_0|q4_1|iq4_nl),[0-9]{1,2}", value):
            raise ValueError("layer cache override must look like 'q6_0,4'")
        return value

    @field_validator("prompt_key")
    @classmethod
    def prompt_key_allowed(cls, value: str) -> str:
        if value not in PROMPTS:
            raise ValueError(f"prompt_key must be one of {sorted(PROMPTS)}")
        return value

    @field_validator("smart_expert_reduction")
    @classmethod
    def smart_expert_reduction_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?){0,3}", value):
            raise ValueError("smart_expert_reduction must look like '3,1' or '2,0.85'")
        return value

    @field_validator("env_ser_full_ranges", "env_ser_cheap_ranges", "env_ser_cheap2_ranges", "env_ser_adaptive_ranges")
    @classmethod
    def env_ser_full_ranges_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"[0-9]{1,2}(:|-)[0-9]{1,2}(,[0-9]{1,2}(:|-)[0-9]{1,2})*", value):
            raise ValueError("SER range env values must look like '12:18' or '12:18,30:36'")
        for part in value.replace("-", ":").split(","):
            first, last = [int(piece) for piece in part.split(":", 1)]
            if not (0 <= first < last <= 48):
                raise ValueError("env_ser_full_ranges layers must satisfy 0 <= first < last <= 48")
        return value

    @field_validator("prewarm_block_size")
    @classmethod
    def prewarm_block_size_safe(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", value):
            raise ValueError("prewarm_block_size must look like '32m' or '1048576'")
        return value

    @field_validator("env_omp_proc_bind")
    @classmethod
    def env_omp_proc_bind_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        allowed = {"close", "spread", "master", "primary", "true", "false"}
        if value not in allowed:
            raise ValueError(f"env_omp_proc_bind must be one of {sorted(allowed)}")
        return value

    @field_validator("env_omp_places")
    @classmethod
    def env_omp_places_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        allowed = {"cores", "threads", "sockets"}
        if value not in allowed:
            raise ValueError(f"env_omp_places must be one of {sorted(allowed)}")
        return value

    @field_validator("env_malloc_nano_zone")
    @classmethod
    def env_malloc_nano_zone_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if value not in {"0", "1"}:
            raise ValueError("env_malloc_nano_zone must be '0' or '1'")
        return value

    @field_validator("extra_args")
    @classmethod
    def extra_args_safe(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", value):
                raise ValueError(f"unsafe extra arg: {value!r}")
        return values

    @field_validator("llama_bin_override", "model_path_override")
    @classmethod
    def path_override_safe(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"[A-Za-z0-9_./:+@%=-]+", value):
            raise ValueError(f"unsafe path override: {value!r}")
        return value


def prompt_text(cfg: BenchConfig) -> str:
    return cfg.prompt if cfg.prompt is not None else PROMPTS[cfg.prompt_key]


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    st = path.stat()
    return f"size={st.st_size}:mtime={int(st.st_mtime)}"


def llama_commit() -> str | None:
    if not DEFAULT_LLAMA_REPO.exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(DEFAULT_LLAMA_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def build_command(cfg: BenchConfig, bin_path: Path | None = None, model: Path | None = None) -> list[str]:
    bin_path = bin_path or llama_bin()
    model = model or model_path()
    cmd = [
        str(bin_path),
        "-m",
        str(model),
        "-c",
        str(cfg.ctx_size),
        "-b",
        str(cfg.batch_size),
        "-ub",
        str(cfg.ubatch_size),
        "-t",
        str(cfg.threads),
        "-tb",
        str(cfg.threads_batch),
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "-n",
        str(cfg.n_predict),
        "--temp",
        str(cfg.temp),
    ]
    if cfg.no_warmup:
        cmd.append("--no-warmup")
    if cfg.flash_attn:
        cmd += ["-fa", "1"]
    if cfg.cache_type_k_first:
        cmd += ["--cache-type-k-first", cfg.cache_type_k_first]
    if cfg.cache_type_k_last:
        cmd += ["--cache-type-k-last", cfg.cache_type_k_last]
    if cfg.cache_type_v_first:
        cmd += ["--cache-type-v-first", cfg.cache_type_v_first]
    if cfg.cache_type_v_last:
        cmd += ["--cache-type-v-last", cfg.cache_type_v_last]
    if cfg.smart_expert_reduction:
        cmd += ["-ser", cfg.smart_expert_reduction]
    if cfg.ignore_eos:
        cmd += ["--ignore-eos"]
    if cfg.no_display_prompt:
        cmd += ["--no-display-prompt"]
    cmd += cfg.extra_args
    cmd += ["-p", prompt_text(cfg)]
    return cmd


def parse_log(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, pattern in TIMING_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            parsed[key] = None
            continue
        match = matches[-1]
        if key == "rss":
            parsed["peak_rss_bytes"] = int(match.group(1))
        elif key in {
            "page_reclaims",
            "page_faults",
            "swaps",
            "voluntary_context_switches",
            "involuntary_context_switches",
        }:
            parsed[key] = int(match.group(1))
        else:
            parsed[key] = float(match.group(1))
    parsed.pop("rss", None)
    return parsed


def run_config(config: dict[str, Any] | BenchConfig) -> dict[str, Any]:
    cfg = config if isinstance(config, BenchConfig) else BenchConfig(**config)
    ensure_dirs()
    bin_path = Path(cfg.llama_bin_override).expanduser() if cfg.llama_bin_override else llama_bin()
    model = Path(cfg.model_path_override).expanduser() if cfg.model_path_override else model_path()
    cmd = build_command(cfg, bin_path, model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg.label)[:80]
    log_path = LOG_DIR / f"{stamp}-{safe_label}.log"
    env_pairs = [
        f"VECLIB_MAXIMUM_THREADS={cfg.env_veclib_threads}",
        f"OMP_WAIT_POLICY={cfg.env_omp_wait_policy}",
        f"OMP_DYNAMIC={cfg.env_omp_dynamic}",
    ]
    if cfg.env_omp_proc_bind:
        env_pairs.append(f"OMP_PROC_BIND={cfg.env_omp_proc_bind}")
    if cfg.env_omp_places:
        env_pairs.append(f"OMP_PLACES={cfg.env_omp_places}")
    if cfg.env_malloc_nano_zone:
        env_pairs.append(f"MallocNanoZone={cfg.env_malloc_nano_zone}")
    if cfg.env_ser_full_first is not None:
        env_pairs.append(f"LLAMA_SER_FULL_FIRST={cfg.env_ser_full_first}")
    if cfg.env_ser_full_last is not None:
        env_pairs.append(f"LLAMA_SER_FULL_LAST={cfg.env_ser_full_last}")
    if cfg.env_ser_full_ranges:
        env_pairs.append(f"LLAMA_SER_FULL_RANGES={cfg.env_ser_full_ranges}")
    if cfg.env_ser_cheap_ranges:
        env_pairs.append(f"LLAMA_SER_CHEAP_RANGES={cfg.env_ser_cheap_ranges}")
    if cfg.env_ser_cheap_min is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP_MIN={cfg.env_ser_cheap_min}")
    if cfg.env_ser_cheap_thresh is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP_THRESH={cfg.env_ser_cheap_thresh}")
    if cfg.env_ser_cheap_max_ntokens is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP_MAX_NTOKENS={cfg.env_ser_cheap_max_ntokens}")
    if cfg.env_ser_cheap2_ranges:
        env_pairs.append(f"LLAMA_SER_CHEAP2_RANGES={cfg.env_ser_cheap2_ranges}")
    if cfg.env_ser_cheap2_min is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP2_MIN={cfg.env_ser_cheap2_min}")
    if cfg.env_ser_cheap2_thresh is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP2_THRESH={cfg.env_ser_cheap2_thresh}")
    if cfg.env_ser_cheap2_max_ntokens is not None:
        env_pairs.append(f"LLAMA_SER_CHEAP2_MAX_NTOKENS={cfg.env_ser_cheap2_max_ntokens}")
    if cfg.env_ser_adaptive_ranges:
        env_pairs.append(f"LLAMA_SER_ADAPTIVE_RANGES={cfg.env_ser_adaptive_ranges}")
    if cfg.env_ser_adaptive_third_ratio is not None:
        env_pairs.append(f"LLAMA_SER_ADAPTIVE_THIRD_RATIO={cfg.env_ser_adaptive_third_ratio}")
    full_cmd = ["/usr/bin/time", "-l", "env", *env_pairs, "caffeinate", "-dimsu", *cmd]

    wall_start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        if cfg.prewarm_model:
            prewarm_cmd = [
                "/usr/bin/time",
                "-l",
                "dd",
                f"if={model}",
                "of=/dev/null",
                f"bs={cfg.prewarm_block_size}",
            ]
            log.write("$ " + " ".join(prewarm_cmd) + "\n\n")
            log.flush()
            subprocess.run(
                prewarm_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(60, min(cfg.timeout_seconds, 600)),
                check=False,
            )
            log.write("\n")
        log.write("$ " + " ".join(full_cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                full_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=cfg.timeout_seconds,
                check=False,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {cfg.timeout_seconds} seconds\n")
            exit_code = 124
    wall_seconds = time.perf_counter() - wall_start

    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(text)
    metrics = {
        key: parsed.get(key)
        for key in (
            "page_reclaims",
            "page_faults",
            "swaps",
            "voluntary_context_switches",
            "involuntary_context_switches",
            "user_seconds",
            "sys_seconds",
        )
        if parsed.get(key) is not None
    }
    metrics["wall_seconds"] = round(wall_seconds, 3)
    if parsed.get("gen_tps") and parsed.get("total_ms"):
        metrics["tokens_per_wall_second"] = round(float(parsed["gen_tps"]), 4)
    tail = "\n".join(text.splitlines()[-40:])
    ptext = prompt_text(cfg)
    run = {
        "ts": db.now_iso(),
        "source": cfg.source,
        "label": cfg.label,
        "external_key": f"{cfg.source}:{log_path}",
        "config_json": db.json_dumps(cfg.model_dump()),
        "command_json": db.json_dumps(full_cmd),
        "prompt_key": cfg.prompt_key,
        "prompt_hash": prompt_hash(ptext),
        "model_path": str(model),
        "model_fingerprint": file_fingerprint(model),
        "llama_bin": str(bin_path),
        "llama_commit": llama_commit(),
        "exit_code": exit_code,
        "pp_tps": parsed.get("pp_tps"),
        "gen_tps": parsed.get("gen_tps"),
        "total_ms": parsed.get("total_ms"),
        "peak_rss_bytes": parsed.get("peak_rss_bytes"),
        "metrics_json": db.json_dumps(metrics),
        "log_path": str(log_path),
        "stdout_tail": tail,
        "quality_flag": None if exit_code == 0 else "failed",
        "notes": None,
    }
    run_id = db.insert_run(run)
    run["id"] = run_id
    run["metrics"] = metrics
    score, components = score_run(run)
    db.insert_score(run_id, score, components)
    run["score"] = score
    run["score_components"] = components
    return run


def import_summary_tsv(path: Path) -> int:
    imported = 0
    source = path.parent.name
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("label") or row.get("LABEL") or row.get("kind") or path.stem
            log_path = row.get("log") or row.get("LOG") or ""
            gen_tps = _float(row.get("gen_tps") or row.get("tg_tps") or row.get("GEN_TPS") or row.get("TG_TPS"))
            pp_tps = _float(row.get("pp_tps") or row.get("PP_TPS"))
            total_ms = _float(row.get("total_ms") or row.get("TOTAL_MS"))
            rss = _int(row.get("rss") or row.get("RSS"))
            exit_code = _int(row.get("status") or row.get("exit_code") or 0)
            config = {
                key: value
                for key, value in row.items()
                if key
                in {
                    "build",
                    "batch",
                    "ubatch",
                    "poll",
                    "fa",
                    "ctk",
                    "ctv",
                    "ngl",
                    "ncmoe",
                    "nkvo",
                    "nopo",
                    "dio",
                    "ot",
                    "reps",
                    "extra",
                    "label",
                    "kind",
                }
                and value not in (None, "")
            }
            run = {
                "ts": row.get("ts") or db.now_iso(),
                "source": f"import:{source}",
                "label": str(label),
                "external_key": f"import:{path}:{label}:{log_path}",
                "config_json": db.json_dumps(config),
                "command_json": "[]",
                "exit_code": exit_code,
                "pp_tps": pp_tps,
                "gen_tps": gen_tps,
                "total_ms": total_ms,
                "peak_rss_bytes": rss,
                "log_path": log_path or str(path),
                "notes": f"imported from {path}",
            }
            run_id = db.insert_run(run)
            run["id"] = run_id
            score, components = score_run(run)
            db.insert_score(run_id, score, components)
            imported += 1
    return imported


def import_many(paths: list[Path]) -> int:
    return sum(import_summary_tsv(path) for path in paths)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
