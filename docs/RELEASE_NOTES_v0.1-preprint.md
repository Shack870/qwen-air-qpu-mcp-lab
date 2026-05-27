# v0.1-preprint Release Notes

This release packages the first public preprint/artifact snapshot of the Qwen
Air QPU/MCP Lab.

## Included

- Paper draft in Markdown, HTML, and PDF form
- Generated SVG figures
- Sanitized public benchmark run log
- Selected milestone and prompt example CSVs
- Reproducibility protocol
- Community validation guide and GitHub issue templates
- Hugging Face Space source for the interactive leaderboard/config explorer
- MCP-style benchmark and IBM Quantum candidate-sampling harness

## Headline Result

- Baseline: approximately 0.09 generation tok/s
- Classical systems optimization frontier: 6.49 generation tok/s
- First IBM Quantum-informed jump: 13.12 generation tok/s
- Clean-room Codex-off validation: 13.91 generation tok/s
- Strict quality-gated record: 14.03 generation tok/s
- Rejected speed-only edge: 16.53 generation tok/s

## Claim Boundary

IBM Quantum did not run Qwen and did not accelerate inference math. The QPU
sampled compact candidate spaces inside a Codex-driven autoresearch loop. The
MacBook judged every candidate through real CPU-only inference.

## Public Links

- GitHub: <https://github.com/Shack870/qwen-air-qpu-mcp-lab>
- Hugging Face Collection: <https://huggingface.co/collections/Shack870/qwen-air-qpu-mcp-lab-6a174dd8d752afe40a429846>
- Hugging Face Dataset: <https://huggingface.co/datasets/Shack870/qwen-air-qpu-mcp-lab>
- Hugging Face Space: <https://huggingface.co/spaces/Shack870/qwen-air-qpu-dashboard>

