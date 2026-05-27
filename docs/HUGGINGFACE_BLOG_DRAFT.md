# From 0.09 to 14.03 tok/s: Quantum-Enhanced Autoresearch for Qwen3 30B MoE on a 2017 MacBook Air

*Draft for Hugging Face Blog Articles. Create the final article at
<https://huggingface.co/new-blog> and paste/adapt this Markdown.*

## TL;DR

I used a Codex-driven Karpathy-style autoresearch loop, informed by IBM Quantum
candidate sampling, to push CPU-only Qwen3-30B-A3B-Instruct-2507-GGUF inference
on a 2017 Intel MacBook Air from roughly 0.09 generation tokens/sec to a
quality-gated 14.03 generation tokens/sec at 16,384 context.

The IBM QPU did **not** run the model. It did something narrower and more
interesting for systems research: it sampled compact QUBO candidate spaces inside
the experimental loop, changing which llama.cpp configurations Codex tested next.
The MacBook remained the judge.

Links:

- Collection: <https://huggingface.co/collections/Shack870/qwen-air-qpu-mcp-lab-6a174dd8d752afe40a429846>
- Dashboard Space: <https://huggingface.co/spaces/Shack870/qwen-air-qpu-dashboard>
- Dataset artifacts: <https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab>
- GitHub: <https://github.com/Shack870/qwen-air-qpu-mcp-lab>

## Why This Was A Weird Target

The test machine is a 2017 Intel MacBook Air with 8GB RAM. It predates the
public Transformer paper. It has no useful modern LLM GPU path. The target model
is a 30.5B-total-parameter Mixture-of-Experts model with 3.3B activated
parameters per token.

That made the experiment less about "run a 30B model normally" and more about
discovering a quality-preserving routed-compute regime.

## The Milestone Curve

The project moved in stages:

| Stage | Generation tok/s | Notes |
| --- | ---: | --- |
| Out-of-box | ~0.09 | Proof-of-life baseline |
| Classical frontier | 6.49 | llama.cpp, mmap, KV, Flash Attention, batch/ubatch, thermal/process tuning |
| QPU-informed jump | 13.12 | First IBM Quantum-informed candidate burst |
| Clean-room check | 13.91 | Codex closed, scripted run |
| Strict record | 14.03 | Quality-gated record |
| Speed-only edge | 16.53 | Rejected because output degraded |

## The Loop

The key architecture was one synchronized loop:

```text
Jody sets the goal and constraints
    -> Codex proposes, edits, runs, logs, and interprets experiments
    -> the MacBook runs real llama.cpp inference and judges candidates
    -> the local database scores the run frontier
    -> compact candidate choices are compressed into QUBO form
    -> IBM Quantum samples candidate bitstrings
    -> Codex decodes those bitstrings into concrete llama.cpp configs
    -> the MacBook tests them
    -> the loop repeats
```

## What Worked

The stable record family used:

- CPU-only inference
- mmap enabled
- q6_0/q6_0 KV cache
- quantized Flash Attention
- 16,384 context
- large batch with moderate ubatch
- Smart Expert Reduction `3,1`
- a narrow cheap layer band around `24:30`
- deterministic evaluation prompts

The fastest raw lanes were not accepted as records unless the output stayed
coherent.

## What The QPU Contributed

The QPU contribution was indirect but practical. It did not accelerate matrix
multiplication. It helped reshape the search frontier after manual and classical
heuristics had started circling the same region.

The QPU-sampled candidate neighborhood produced the first large post-classical
jump, from 6.49 to 13.12 generation tok/s. Continued local refinement around that
region produced the strict 14.03 result.

## What To Try

The dashboard Space includes:

- a public leaderboard from 1,522 sanitized logged runs
- a config explorer that estimates speed/quality risk from nearest neighbors
- a QPU candidate demo showing how bitstrings map to candidate choices
- project figures and artifact links

## Caveats

This is not a universal claim that all 8GB laptops can run every 30B MoE well.
The result is prompt-sensitive, hardware-sensitive, and depends on a particular
model quant, build, and operating regime.

The scientific claim is narrower:

> A quantum-enhanced Codex autoresearch loop can discover surprising
> quality-preserving operating regimes for routed MoE inference on hardware that
> initially looks out of bounds.

## Call For Replication

If you try it, please report:

- hardware and OS
- model quant and file size
- exact command/config
- tokens/sec
- prompt and output
- page faults, swaps, and memory state if available
- quality pass/fail

Open validation issues here:

<https://github.com/Shack870/qwen-air-qpu-mcp-lab/issues>

