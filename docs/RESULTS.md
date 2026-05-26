# Results And Project Notes

## Summary

This lab explored whether an 8 GB 2017 Intel MacBook Air could run a modern
30B-total-parameter MoE model at usable speed by changing the inference regime,
not the hardware.

The model is not treated as a dense 30B model to be run normally. The practical
target is a quality-constrained MoE operating point: preserve enough expert-path
behavior for factual and code prompts while reducing avoidable compute and memory
pressure.

## High-Water Marks

| Stage | Result | Notes |
| --- | ---: | --- |
| Out-of-box baseline | ~0.09 tok/s | Initial local Qwen MoE run on the MacBook Air. |
| Manual/classical optimization | 6.49 tok/s | Dan Woods, Raspberry Pi, and autoresearch-inspired search. |
| First QPU-guided jump | 13.12 tok/s | First clear IBM-QPU-labelled candidate burst in the local DB. |
| Clean-room aggregate lane | 13.91 tok/s | Codex closed, scripted scoreboard run. |
| Strict-quality record | 14.03 tok/s | Passed Serbia/Mars/prime quality gates. |
| Speed-only high | 16.53 tok/s | Not a quality record; corrupted factual output. |

## Strict-Quality Record Config

```text
model: byteshape/Qwen3-30B-A3B-Instruct-2507-GGUF
quant: Q3_K_S 2.66bpw
ctx: 16384
batch: 2456
ubatch: 144
threads: 4
threads_batch: 4
KV cache: q6_0/q6_0
flash attention: enabled
SER: 3,1
cheap SER layers: 24:30
cheap SER min experts: 2
cheap SER threshold: 1.0
temperature: 0.0
n_predict: 128
```

## What Worked

- Quantized Flash Attention build with IQK/FA kernels.
- `q6_0/q6_0` KV cache as a speed/quality balance on this machine.
- 16k context with large batch and moderate ubatch.
- Smart Expert Reduction `3,1` as the stable base lane.
- A narrow Routerclamp cheap layer band (`24:30`) rather than global expert cuts.
- QPU/MCP harness as a structured candidate sampler and experiment logger.
- Repeatable benchmark records with resource counters and quality gates.

## What Failed Or Was Deprioritized

- Intel GPU / Metal offload did not produce a useful CPU-only record lane.
- SSD as a raw decode accelerator is the wrong framing; SSD is more plausible as
  a cold-expert warehouse in future model-repacking work.
- Global `ser 1,*` and top-2 variants were fast but frequently corrupted factual
  or code output.
- Static hotset clamping caused slowdown or repetition in early prototypes.
- Speculative decoding with available draft paths accepted 0 useful draft tokens
  in the tested setup.
- Browser UI overhead was not the core bottleneck for the record lane.

## Inspirations And Credits

- Jody Shackelford: experiment owner, 2017 MacBook Air lab operator, project lead.
- Codex/GPT-5: coding agent and experiment collaborator.
- Dan Woods / `flash-moe`: SSD-backed expert streaming and OS page-cache framing
  for very large MoE inference.
- Andrej Karpathy / `autoresearch`: keep/discard agentic experiment loops.
- ByteShape / Potato OS / Raspberry Pi 5 Qwen3 demonstrations: proof that
  low-power hardware could run this model family surprisingly well.
- IBM Quantum / Qiskit Runtime: hybrid QPU candidate-sampling path used by the
  MCP harness.

## Claim Boundaries

The current evidence supports:

- legacy CPU-only hardware can achieve usable MoE inference rates with careful
  quantization, KV choices, batch/ubatch tuning, SER, and process-state control;
- QPU-guided candidate sampling can be integrated into a reproducible local
  optimization loop;
- speed-only results must be separated from quality-constrained results.

It does not yet prove:

- that the QPU is globally superior to all classical optimizers for this search;
- that every natural-language prompt maintains full model quality at the record
  speed;
- that the discovered operating point transfers unchanged to other hardware or
  other Qwen3 MoE quants.

That is why this repo emphasizes reproducibility scripts, quality gates, and
machine-state reporting.
