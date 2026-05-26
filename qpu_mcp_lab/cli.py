from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import db
from .bench import BenchConfig, import_many, run_config
from .optimizer import build_qubo, propose_classical_candidates, sample_qubo_local
from .quantum import credential_status, get_job_result, list_backends, sample_candidates, submit_qaoa_job
from .qpu_strategy import (
    decode_job_candidates,
    submit_micro_frontier_job,
    submit_impl_flags_angle_sweep_job,
    submit_impl_flags_job,
    submit_moonshot_ridge_v3_angle_sweep_job,
    submit_moonshot_ridge_v3_job,
    submit_record_lane_job,
    submit_ridge_v2_job,
    sweep_angles,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qpu-mcp-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")

    p_import = sub.add_parser("import-summaries")
    p_import.add_argument("paths", nargs="+")

    p_best = sub.add_parser("best")
    p_best.add_argument("--limit", type=int, default=10)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config-json", default="{}")

    p_propose = sub.add_parser("propose")
    p_propose.add_argument("--limit", type=int, default=12)

    p_qubo = sub.add_parser("build-qubo")
    p_qubo.add_argument("--limit-runs", type=int, default=200)

    p_sample = sub.add_parser("sample-qubo")
    p_sample.add_argument("--top-k", type=int, default=8)

    sub.add_parser("quantum-credentials")

    p_backends = sub.add_parser("quantum-backends")
    p_backends.add_argument("--include-simulators", action="store_true")

    p_submit = sub.add_parser("submit-qaoa")
    p_submit.add_argument("--backend", required=True)
    p_submit.add_argument("--shots", type=int, default=256)
    p_submit.add_argument("--allow-real-qpu", action="store_true")

    p_job = sub.add_parser("job-result")
    p_job.add_argument("job_id")
    p_job.add_argument("--refresh", action="store_true")

    p_sweep = sub.add_parser("sweep-qaoa-angles")
    p_sweep.add_argument("--limit", type=int, default=10)

    p_micro = sub.add_parser("submit-micro-frontier")
    p_micro.add_argument("--backend", required=True)
    p_micro.add_argument("--shots", type=int, default=256)
    p_micro.add_argument("--allow-real-qpu", action="store_true")
    p_micro.add_argument("--manual-angle", action="store_true")
    p_micro.add_argument("--gamma", type=float, default=1.0)
    p_micro.add_argument("--beta", type=float, default=0.7)

    p_record = sub.add_parser("submit-record-lane")
    p_record.add_argument("--backend", required=True)
    p_record.add_argument("--shots", type=int, default=256)
    p_record.add_argument("--allow-real-qpu", action="store_true")
    p_record.add_argument("--manual-angle", action="store_true")
    p_record.add_argument("--gamma", type=float, default=1.0)
    p_record.add_argument("--beta", type=float, default=0.7)

    p_ridge = sub.add_parser("submit-ridge-v2")
    p_ridge.add_argument("--backend", required=True)
    p_ridge.add_argument("--shots", type=int, default=256)
    p_ridge.add_argument("--allow-real-qpu", action="store_true")
    p_ridge.add_argument("--manual-angle", action="store_true")
    p_ridge.add_argument("--gamma", type=float, default=1.0)
    p_ridge.add_argument("--beta", type=float, default=0.7)

    p_moonshot = sub.add_parser("submit-moonshot-ridge-v3")
    p_moonshot.add_argument("--backend", required=True)
    p_moonshot.add_argument("--shots", type=int, default=256)
    p_moonshot.add_argument("--allow-real-qpu", action="store_true")
    p_moonshot.add_argument("--manual-angle", action="store_true")
    p_moonshot.add_argument("--gamma", type=float, default=1.0)
    p_moonshot.add_argument("--beta", type=float, default=0.7)

    p_moonshot_sweep = sub.add_parser("submit-moonshot-ridge-v3-angle-sweep")
    p_moonshot_sweep.add_argument("--backend", required=True)
    p_moonshot_sweep.add_argument("--shots-per-angle", type=int, default=64)
    p_moonshot_sweep.add_argument("--angle-count", type=int, default=4)
    p_moonshot_sweep.add_argument("--allow-real-qpu", action="store_true")

    p_impl = sub.add_parser("submit-impl-flags")
    p_impl.add_argument("--backend", required=True)
    p_impl.add_argument("--shots", type=int, default=256)
    p_impl.add_argument("--allow-real-qpu", action="store_true")
    p_impl.add_argument("--manual-angle", action="store_true")
    p_impl.add_argument("--gamma", type=float, default=1.0)
    p_impl.add_argument("--beta", type=float, default=0.7)

    p_impl_sweep = sub.add_parser("submit-impl-flags-angle-sweep")
    p_impl_sweep.add_argument("--backend", required=True)
    p_impl_sweep.add_argument("--shots-per-angle", type=int, default=64)
    p_impl_sweep.add_argument("--angle-count", type=int, default=4)
    p_impl_sweep.add_argument("--allow-real-qpu", action="store_true")

    p_decode = sub.add_parser("decode-job-candidates")
    p_decode.add_argument("job_id")
    p_decode.add_argument("--top-k", type=int, default=12)

    p_jobs = sub.add_parser("quantum-jobs")
    p_jobs.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "init-db":
        db.init_db()
        print_json({"ok": True, "db": str(db.DB_PATH), "rows": db.count_runs()})
    elif args.cmd == "import-summaries":
        count = import_many([Path(path) for path in args.paths])
        print_json({"imported_rows": count, "total_rows": db.count_runs()})
    elif args.cmd == "best":
        print_json(db.best_runs(limit=args.limit))
    elif args.cmd == "run":
        cfg = BenchConfig(**json.loads(args.config_json)).model_dump()
        print_json(run_config(cfg))
    elif args.cmd == "propose":
        print_json(propose_classical_candidates(limit=args.limit))
    elif args.cmd == "build-qubo":
        print_json(build_qubo(limit_runs=args.limit_runs))
    elif args.cmd == "sample-qubo":
        qubo = build_qubo()
        print_json(sample_qubo_local(qubo, top_k=args.top_k))
    elif args.cmd == "quantum-credentials":
        print_json(credential_status())
    elif args.cmd == "quantum-backends":
        print_json(list_backends(simulator=None if args.include_simulators else False))
    elif args.cmd == "submit-qaoa":
        qubo = build_qubo()
        print_json(
            submit_qaoa_job(
                qubo_payload=qubo,
                backend_name=args.backend,
                shots=args.shots,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "job-result":
        print_json(get_job_result(args.job_id, refresh=args.refresh))
    elif args.cmd == "sweep-qaoa-angles":
        print_json(sweep_angles()[: args.limit])
    elif args.cmd == "submit-micro-frontier":
        print_json(
            submit_micro_frontier_job(
                backend_name=args.backend,
                shots=args.shots,
                gamma=args.gamma,
                beta=args.beta,
                auto_angle=not args.manual_angle,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-record-lane":
        print_json(
            submit_record_lane_job(
                backend_name=args.backend,
                shots=args.shots,
                gamma=args.gamma,
                beta=args.beta,
                auto_angle=not args.manual_angle,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-ridge-v2":
        print_json(
            submit_ridge_v2_job(
                backend_name=args.backend,
                shots=args.shots,
                gamma=args.gamma,
                beta=args.beta,
                auto_angle=not args.manual_angle,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-moonshot-ridge-v3":
        print_json(
            submit_moonshot_ridge_v3_job(
                backend_name=args.backend,
                shots=args.shots,
                gamma=args.gamma,
                beta=args.beta,
                auto_angle=not args.manual_angle,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-moonshot-ridge-v3-angle-sweep":
        print_json(
            submit_moonshot_ridge_v3_angle_sweep_job(
                backend_name=args.backend,
                shots_per_angle=args.shots_per_angle,
                angle_count=args.angle_count,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-impl-flags":
        print_json(
            submit_impl_flags_job(
                backend_name=args.backend,
                shots=args.shots,
                gamma=args.gamma,
                beta=args.beta,
                auto_angle=not args.manual_angle,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "submit-impl-flags-angle-sweep":
        print_json(
            submit_impl_flags_angle_sweep_job(
                backend_name=args.backend,
                shots_per_angle=args.shots_per_angle,
                angle_count=args.angle_count,
                allow_real_qpu=args.allow_real_qpu,
            )
        )
    elif args.cmd == "decode-job-candidates":
        print_json(decode_job_candidates(args.job_id, top_k=args.top_k))
    elif args.cmd == "quantum-jobs":
        print_json(db.list_quantum_jobs(limit=args.limit))
    else:
        parser.error(f"unknown command {args.cmd}")


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
