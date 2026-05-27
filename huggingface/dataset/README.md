---
license: mit
language:
- en
tags:
- qwen
- qwen3
- mixture-of-experts
- llama-cpp
- cpu-inference
- quantum-computing
- qiskit
- ibm-quantum
- mcp
- autoresearch
- benchmark
pretty_name: Qwen Air QPU MCP Lab
size_categories:
- 1K<n<10K
---

# Qwen Air QPU/MCP Lab Dataset

This dataset repository mirrors the public artifacts for the Qwen Air QPU/MCP
Lab: a CPU-only Qwen3 MoE inference optimization project on a 2017 Intel
MacBook Air with 8GB RAM.

The headline result is a quality-gated improvement from about `0.09` to `14.03`
generation tokens/sec on `Qwen3-30B-A3B-Instruct-2507-GGUF` at 16,384 context,
using a synchronized Codex-driven, IBM Quantum-informed autoresearch loop.

The QPU did not run Qwen. IBM Quantum sampled compact QUBO candidate spaces
inside the research loop; the MacBook judged each candidate through real local
llama.cpp inference.

## Included Artifacts

- `paper/quantum_enhanced_legacy_moe_inference.md` - paper source
- `paper/quantum_enhanced_legacy_moe_inference.pdf` - generated preprint PDF
- `paper/data/public_runs.csv` - sanitized public run log
- `paper/figures/*.svg` - generated figures
- `docs/REPRODUCIBILITY.md` - validation protocol
- `docs/COMMUNITY_VALIDATION.md` - community benchmark reporting guide
- `docs/HUGGINGFACE_BLOG_DRAFT.md` - public blog draft
- `docs/PRESS_KIT.md` - press and launch kit
- `release/qwen-air-qpu-mcp-lab-v0.1-preprint.zip` - packaged release archive

## Links

- GitHub: https://github.com/Shack870/qwen-air-qpu-mcp-lab
- GitHub preprint release: https://github.com/Shack870/qwen-air-qpu-mcp-lab/releases/tag/v0.1-preprint
- Interactive Space: https://huggingface.co/spaces/Shack870/qwen-air-qpu-dashboard
- Collection: https://huggingface.co/collections/Shack870/qwen-air-qpu-mcp-lab-6a174dd8d752afe40a429846
