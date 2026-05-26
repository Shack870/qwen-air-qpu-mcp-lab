# Paper Artifacts

Main manuscript:

- [Quantum-Enhanced Hyperparameter Tuning for High-Performance On-Device CPU-Only Inference of Mixture-of-Experts LLMs on Legacy Hardware](quantum_enhanced_legacy_moe_inference.md)

Regenerate charts and CSV snapshots:

```bash
python3 paper/make_figures.py
```

Generated figures:

- [Throughput progression](figures/throughput_progression.svg)
- [Quantum-guided search jump](figures/qpu_jump.svg)
- [Speed/quality boundary](figures/quality_boundary.svg)
- [Prompt example throughput](figures/prompt_examples.svg)

Generated data:

- [Milestones](data/milestones.csv)
- [Selected runs](data/selected_runs.csv)
- [Top source summary](data/source_summary_top30.csv)
