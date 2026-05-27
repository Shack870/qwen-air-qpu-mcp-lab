# Community Validation Guide

This project is strongest when other people try to reproduce, falsify, or beat
the reported results.

## What To Validate

The main public claim is:

> A 2017 Intel MacBook Air with 8GB RAM achieved a quality-gated 14.03
> generation tokens/sec on Qwen3-30B-A3B-Instruct-2507-GGUF at 16,384 context,
> CPU-only, using a quantum-enhanced Codex autoresearch loop to discover the
> operating regime.

The record is not the same as the speed-only edge. The project also observed
16.53 tok/s, but rejected it because output coherence failed.

## Report These Fields

Please open a GitHub issue using the **Validation result** template and include:

- hardware and OS
- model repo, filename, quant, and file size
- exact `llama-cli` or harness command
- `llama-cli --version` and source commit
- context, batch, ubatch, thread settings
- KV cache type
- expert scheduling / SER settings
- prompt and exact output
- generation tok/s and prompt-eval tok/s
- peak RSS, page faults, swaps, and thermal notes when available
- quality status: strict pass, coherent but not strict-gated, speed-only, or failed

## Strict Gate Prompts

Use these short prompts to distinguish speed from usable output:

1. `What is the capital of Serbia? Answer with one short sentence.`
2. `What is the capital of Mars? Answer factually in one short sentence.`
3. `Write a compact Python function named is_prime that checks whether n is prime.`

## Comparison Categories

Useful validation categories:

- same hardware, same quant, same command
- same hardware, different quant
- different low-RAM Intel Mac
- Raspberry Pi 5 8GB or 16GB
- old Intel laptop with Linux
- modern CPU-only baseline
- GPU-offload counterexample

## What Counts As A Win

Any of these helps the project:

- reproducing the strict record within reasonable variance
- showing a better quality-gated setting
- showing a speed-only setting that fails quality
- finding a command/documentation bug
- showing the method does not transfer to a specific machine

Negative results are welcome. The goal is science, not scoreboard theater.

