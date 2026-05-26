# Qwen Air QPU/MCP Lab

Quantum/classical optimization harness for CPU-only Qwen3 MoE inference on legacy
Mac hardware.

This repository contains the local experiment runner used in the 2017 Intel
MacBook Air Qwen3-30B-A3B work. The goal is reproducible validation, not a
turnkey model distribution: model weights and custom `llama.cpp` builds are
external inputs.

## What This Does

- Runs validated `llama.cpp` benchmark configs.
- Logs exact commands, timing output, resource counters, and scores to SQLite.
- Scores runs with a stability-aware objective instead of raw tokens/sec only.
- Proposes classical candidates and compact QUBO search spaces.
- Submits guarded IBM Quantum / Qiskit Runtime sampling jobs only when explicitly allowed.
- Exposes the same controls as a narrow MCP server for Codex or other clients.

## Current Result Snapshot

Primary hardware:

- 2017 Intel MacBook Air
- Intel Core i5, 2 cores / 4 threads
- 8 GB LPDDR3 RAM
- internal SSD
- macOS 12.7.6

Best strict-quality record observed in this lab:

- `14.03 tok/s`
- model: `byteshape/Qwen3-30B-A3B-Instruct-2507-GGUF`
- quant: `Q3_K_S 2.66bpw`
- context: `16384`
- KV cache: `q6_0/q6_0`
- smart expert reduction: `-ser 3,1`
- cheap SER layers: `24:30`, top-2, threshold `1.0`
- quality gate: Serbia capital, Mars capital, and prime-function prompts

Important caveat: speed-only settings reached higher values but corrupted factual
or code outputs. This repo distinguishes speed records from quality-constrained
records.

See [docs/RESULTS.md](docs/RESULTS.md) and [MOONSHOT_LIST.md](MOONSHOT_LIST.md).

## Quick Start

```bash
git clone YOUR_FORK_OR_REPO_URL qpu-mcp-lab
cd qpu-mcp-lab
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
```

Edit `config.json` so `llama_bin` points at your `llama-cli` binary and
`model_path` points at your local GGUF file. The same settings can be supplied
with environment variables:

```bash
export QPU_MCP_LAB_LLAMA_BIN="$HOME/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli"
export QPU_MCP_LAB_MODEL_PATH="$HOME/qwen-air-tests/models/byteshape-qwen3-30b-a3b-2507/Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf"
```

Validate the local environment:

```bash
.venv/bin/python scripts/validate_environment.py
```

Initialize the experiment DB and inspect credentials:

```bash
.venv/bin/python -m qpu_mcp_lab.cli init-db
.venv/bin/python -m qpu_mcp_lab.cli quantum-credentials
```

Run a single benchmark from JSON:

```bash
.venv/bin/python -m qpu_mcp_lab.cli run --config-json '{
  "label": "public_smoke",
  "prompt_key": "mars_fact_list",
  "ctx_size": 16384,
  "batch_size": 2456,
  "ubatch_size": 144,
  "threads": 4,
  "threads_batch": 4,
  "cache_type_k": "q6_0",
  "cache_type_v": "q6_0",
  "smart_expert_reduction": "3,1",
  "env_ser_cheap_ranges": "24:30",
  "env_ser_cheap_min": 2,
  "env_ser_cheap_thresh": 1.0,
  "n_predict": 128,
  "temp": 0.0,
  "ignore_eos": true
}'
```

## IBM Quantum Credentials

IBM credentials must stay out of source code, config files, logs, and chats. Use
macOS Keychain:

```bash
./scripts/store_ibm_key.sh
```

Or temporary environment variables:

```bash
export IBM_QUANTUM_API_KEY='...'
export IBM_QUANTUM_INSTANCE='...'
```

Real QPU submission is guarded. Commands are dry-run or simulator-first unless
you pass `--allow-real-qpu`.

```bash
.venv/bin/python -m qpu_mcp_lab.cli quantum-backends
.venv/bin/python -m qpu_mcp_lab.cli sweep-qaoa-angles --limit 5
.venv/bin/python -m qpu_mcp_lab.cli submit-micro-frontier --backend ibm_fez --shots 256 --allow-real-qpu
```

## MCP Server

Run the local MCP server over stdio:

```bash
./scripts/run_mcp_server.sh
```

The MCP tools expose narrow operations such as `bench_run_config`,
`optimizer_build_qubo`, `quantum_list_backends`, and
`quantum_submit_micro_frontier_job`. They intentionally do not expose arbitrary
shell access or secrets.

## Reproducibility

Start with [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The short version:

1. Use the same GGUF quant.
2. Build a compatible `llama-cli` with the quantized Flash Attention kernels used
   in this lab.
3. Run the environment validator.
4. Run the strict-quality record lane.
5. Report speed, prompt eval, page faults, context switches, thermal state, and
   exact output quality gates.

## Prior Inspiration

This lab was inspired by:

- Dan Woods / `flash-moe`: SSD-backed expert streaming for very large MoEs.
- Andrej Karpathy / `autoresearch`: automated experiment loops and keep/discard search.
- ByteShape / Potato OS Raspberry Pi demonstrations for Qwen3-30B-A3B.
- IBM Quantum / Qiskit Runtime for guarded hybrid candidate sampling.

See [docs/RESULTS.md](docs/RESULTS.md) for a fuller project narrative.
