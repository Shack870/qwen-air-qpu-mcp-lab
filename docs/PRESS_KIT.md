# Press Kit

## One-Sentence Summary

Jody Shackelford used a quantum-enhanced Codex autoresearch loop to improve
CPU-only Qwen3 30B MoE inference on a 2017 Intel MacBook Air from roughly 0.09
to a quality-gated 14.03 generation tokens/sec at 16k context.

## Short Description

The Qwen Air QPU/MCP Lab is an open-source research artifact showing how a
Codex-driven experiment loop, informed by IBM Quantum candidate sampling, found a
high-throughput CPU-only operating regime for Qwen3-30B-A3B-Instruct-2507-GGUF
on an 8GB pre-Transformer-era MacBook Air.

The IBM QPU did not run the language model. It sampled compact QUBO candidate
spaces inside the experimental loop. Codex decoded those samples into concrete
llama.cpp configurations, and the MacBook judged each candidate with real local
inference.

## Key Numbers

- Start: ~0.09 generation tok/s
- Classical frontier: 6.49 generation tok/s
- First IBM Quantum-informed jump: 13.12 generation tok/s
- Clean-room Codex-off run: 13.91 generation tok/s
- Strict quality-gated record: 14.03 generation tok/s
- Rejected speed-only edge: 16.53 generation tok/s
- Hardware: 2017 Intel MacBook Air, 8GB RAM, CPU-only
- Model: Qwen3-30B-A3B-Instruct-2507-GGUF, Q3_K_S 2.66bpw
- Context: 16,384 tokens

## Links

- Collection: <https://huggingface.co/collections/Shack870/qwen-air-qpu-mcp-lab-6a174dd8d752afe40a429846>
- Dashboard: <https://huggingface.co/spaces/Shack870/qwen-air-qpu-dashboard>
- Dataset artifacts: <https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab>
- GitHub: <https://github.com/Shack870/qwen-air-qpu-mcp-lab>
- Paper draft: <https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab/blob/main/paper/quantum_enhanced_legacy_moe_inference.md>

## Accurate Headline Options

- Quantum-informed autoresearch pushes Qwen3 30B MoE to 14 tok/s on a 2017 MacBook Air
- From 0.09 to 14.03 tok/s: Qwen3 MoE on legacy hardware with a Codex + IBM Quantum loop
- A pre-Transformer MacBook Air becomes a hybrid quantum optimization lab for MoE inference

## Important Boundary

Do not say:

> IBM Quantum ran Qwen.

Say:

> IBM Quantum sampled compact candidate spaces that informed a Codex-driven
> experimental loop; local MacBook inference remained the benchmark judge.

## Suggested Outreach Targets

- Hugging Face community
- r/LocalLLaMA
- Hacker News / Show HN
- IBM Quantum and Qiskit community channels
- Hackaday
- ServeTheHome
- The Register
- The New Stack
- VentureBeat AI
- The Decoder
- IEEE Spectrum
- University communications office

