# Reproducibility Protocol

This document describes how to validate the Qwen Air benchmark claims on another
machine. It is intentionally conservative: record claims should include speed,
quality, and system-state evidence.

## 1. Hardware And OS

Record:

```bash
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
df -h /
diskutil list
```

The reference system was a 2017 Intel MacBook Air with 8 GB RAM, internal SSD,
and macOS 12.7.6.

## 2. Model

Reference model:

- Hugging Face repo: `byteshape/Qwen3-30B-A3B-Instruct-2507-GGUF`
- quant: `Q3_K_S 2.66bpw`
- local filename used here: `Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf`

Do not commit model files to this repository.

## 3. llama.cpp / ik_llama.cpp Build

Reference binary path:

```bash
~/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli
```

The record family used a source build with the quantized Flash Attention kernels
enabled. Record the exact commit:

```bash
git -C ~/src/ik_llama.cpp rev-parse HEAD
~/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli --version
```

If you use a different fork or commit, keep the run, but label it separately.

## 4. Configure The Harness

```bash
cp config.example.json config.json
```

Edit `config.json`:

```json
{
  "llama_bin": "~/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli",
  "model_path": "~/qwen-air-tests/models/byteshape-qwen3-30b-a3b-2507/Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf",
  "llama_repo": "~/src/ik_llama.cpp",
  "safe_memory_gb": 6.5,
  "default_backend": "local-simulator",
  "allow_real_qpu_jobs_by_default": false
}
```

Validate:

```bash
.venv/bin/python scripts/validate_environment.py
```

## 5. Strict-Quality Record Lane

Use the record-family config:

```bash
.venv/bin/python -m qpu_mcp_lab.cli run --config-json '{
  "label": "strict_record_reproduction",
  "prompt": "<|im_start|>user\nContinue this comma-separated list of Mars facts: red planet, thin atmosphere,<|im_end|>\n<|im_start|>assistant\n",
  "ctx_size": 16384,
  "batch_size": 2456,
  "ubatch_size": 144,
  "threads": 4,
  "threads_batch": 4,
  "cache_type_k": "q6_0",
  "cache_type_v": "q6_0",
  "flash_attn": true,
  "smart_expert_reduction": "3,1",
  "env_veclib_threads": 1,
  "env_omp_wait_policy": "ACTIVE",
  "env_omp_dynamic": "FALSE",
  "env_ser_cheap_ranges": "24:30",
  "env_ser_cheap_min": 2,
  "env_ser_cheap_thresh": 1.0,
  "n_predict": 128,
  "temp": 0.0,
  "ignore_eos": true,
  "no_display_prompt": true,
  "timeout_seconds": 420
}'
```

Expected reference result on the original machine:

- strict-quality record: `14.03 tok/s`
- clean-room aggregate lane: `13.91 tok/s`
- prior non-QPU optimized high: `6.49 tok/s`
- starting point: about `0.09 tok/s`

Exact repeats vary with page cache state, context switches, thermals, background
processes, and prompt shape. A failed repeat is useful data if it includes the
resource counters.

## 6. Quality Gate

A speed result is not a quality result unless it passes the strict prompts:

1. `What is the capital of Serbia?`
2. `What is the capital of Mars?`
3. `Write a compact Python function named is_prime that checks whether n is prime.`

Known failure modes:

- `ser 1,*` can be fast but factually corrupt.
- Broad top-2 Routerclamp can become repetitive or break code.
- Long natural-language prompts can be slower and can expose degradation even
  when short benchmark continuations look good.

## 7. Report Format

For outside validation, report:

- hardware and OS
- `llama-cli --version`
- llama.cpp/ik_llama.cpp commit
- model repo, filename, quant, and file size
- full config JSON
- prompt and output
- `gen_tps`, `pp_tps`, and wall time
- max RSS, page faults, swaps, and context switches
- whether quality gates passed

## 8. Quantum Path

The IBM Quantum workflow proposes compact binary candidate choices. The MacBook
benchmark remains the judge.

Recommended validation order:

1. Run local classical candidates.
2. Build a QUBO.
3. Sweep QAOA angles locally.
4. Submit a tiny guarded IBM job only with `--allow-real-qpu`.
5. Decode sampled bitstrings into benchmark configs.
6. Run those configs locally and compare against random/classical baselines.

Never publish IBM API keys, job credentials, or private account identifiers.
