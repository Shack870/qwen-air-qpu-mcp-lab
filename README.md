# Qwen Air QPU/MCP Lab

This is a local optimization harness for the 2017 MacBook Air Qwen3 MoE experiment.
It gives an LLM or Codex client narrow, auditable tools instead of arbitrary shell access:

- run validated `llama.cpp` benchmark configs
- log runs into SQLite
- score runs with a stability-aware objective
- propose classical candidates
- build compact QUBO surrogates
- sample QUBOs locally
- connect to IBM Quantum through a guarded adapter
- expose the whole thing as an MCP server

The IBM API key belongs in macOS Keychain, not in source code, config files, logs, or chat.

```bash
cd /Users/jodyshackelford/qwen-air-tests/qpu-mcp-lab
./scripts/store_ibm_key.sh
```

Equivalent manual commands:

```bash
security add-generic-password -a "$USER" -s ibm_quantum_api_key -w 'PASTE_API_KEY_HERE' -U
security add-generic-password -a "$USER" -s ibm_quantum_instance_crn -w 'PASTE_INSTANCE_OR_CRN_HERE' -U
```

Environment variables also work for one terminal session:

```bash
export IBM_QUANTUM_API_KEY='...'
export IBM_QUANTUM_INSTANCE='...'
```

## Local Commands

The venv uses the official MCP Python SDK and IBM's Qiskit Runtime client:

- https://github.com/modelcontextprotocol/python-sdk
- https://quantum.cloud.ibm.com/docs/guides/initialize-account
- https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/qiskit-runtime-service

```bash
cd /Users/jodyshackelford/qwen-air-tests/qpu-mcp-lab
.venv/bin/python -m qpu_mcp_lab.cli init-db
.venv/bin/python -m qpu_mcp_lab.cli quantum-credentials
.venv/bin/python -m qpu_mcp_lab.cli best --limit 5
.venv/bin/python -m qpu_mcp_lab.cli propose --limit 8
.venv/bin/python -m qpu_mcp_lab.cli build-qubo
.venv/bin/python -m qpu_mcp_lab.cli sample-qubo --top-k 8
.venv/bin/python -m qpu_mcp_lab.cli submit-qaoa --backend ibm_brisbane
.venv/bin/python -m qpu_mcp_lab.cli sweep-qaoa-angles --limit 5
.venv/bin/python -m qpu_mcp_lab.cli submit-micro-frontier --backend ibm_fez --shots 256
.venv/bin/python -m qpu_mcp_lab.cli decode-job-candidates JOB_ID --top-k 12
```

Run the MCP server over stdio:

```bash
cd /Users/jodyshackelford/qwen-air-tests/qpu-mcp-lab
./scripts/run_mcp_server.sh
```

## Current Benchmark Defaults

The default config starts from the current best MacBook Air frontier:

- binary: `/Users/jodyshackelford/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli`
- model: `Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf`
- context: `16384`
- batch: `1792`
- ubatch: `96`
- threads: `4`
- KV: `q6_0/q6_0`
- flash attention: on
- smart expert reduction: `3,1`

Real IBM QPU submission is guarded. `submit-qaoa` is a dry run by default. A real
hardware job requires:

```bash
.venv/bin/python -m qpu_mcp_lab.cli submit-qaoa --backend BACKEND_NAME --shots 256 --allow-real-qpu
```

Use small jobs first; Open Plan time is precious.

## QPU Workflow

The QPU path is now a loop, not a one-off call:

1. Fit or hand-build a compact frontier QUBO.
2. Sweep QAOA angles locally with a statevector simulator.
3. Submit only the chosen tiny QUBO to IBM.
4. Persist the IBM job and counts in SQLite.
5. Decode the sampled bitstrings into llama configs.
6. Run the MacBook benchmark on the decoded candidates.
7. Feed the real result back into the next QUBO.
