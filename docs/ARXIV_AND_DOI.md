# arXiv / TechRxiv / DOI Instructions

This file tracks the parts that require a human web account step.

## DOI Through Zenodo

1. Log in to Zenodo.
2. Connect GitHub in Zenodo account settings.
3. Enable the repository `Shack870/qwen-air-qpu-mcp-lab`.
4. Create the GitHub release `v0.1-preprint`.
5. Zenodo should archive the release and mint a DOI.
6. Add the DOI badge/link to:
   - `README.md`
   - `paper/quantum_enhanced_legacy_moe_inference.md`
   - Hugging Face dataset card
   - Hugging Face Space

The repository includes `.zenodo.json` metadata for the archive.

## arXiv

Recommended initial categories:

- Primary: `cs.LG` or `cs.PF`
- Cross-list candidates: `cs.DC`, `cs.AI`

Use:

- `paper/quantum_enhanced_legacy_moe_inference.pdf`
- GitHub release link
- Hugging Face Collection link
- Hugging Face Dataset link
- Hugging Face Space link

Suggested abstract opening:

> Using a single quantum-enhanced Karpathy-style autoresearch loop, where
> Codex/GPT-5 drove the experimental cycle and IBM Quantum reshaped the
> candidate frontier inside that loop by sampling compact QUBO search spaces, I
> improved CPU-only inference of Qwen3-30B-A3B-Instruct-2507-GGUF on a 2017
> Intel MacBook Air by 155.9x, from approximately 0.09 to a quality-gated 14.03
> generation tokens/sec at 16,384 context.

## TechRxiv

Use TechRxiv if arXiv endorsement or category fit slows down the preprint.

Upload:

- the PDF
- GitHub release link
- Hugging Face Collection link
- press-kit summary

## Hugging Face Blog

Hugging Face Blog Articles are created through the web editor:

<https://huggingface.co/new-blog>

Paste/adapt:

- `docs/HUGGINGFACE_BLOG_DRAFT.md`

