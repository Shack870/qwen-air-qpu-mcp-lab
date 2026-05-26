# Qwen Air Moonshot List

This file is the lab board for serious, testable ideas. It separates what we trust today from the wild branches worth keeping alive.

## Current Records

- Strict quality baseline: 14.03 tok/s, `ser 3,1`, `q6_0/q6_0`, `ctx 16384`, `b2456`, `ub144`, cheap layers `24:30` at top-2, passed Serbia/Mars/prime.
- Clean-room aggregate lane: 13.91 tok/s, `ser 3,1`, `q6_0/q6_0`, `ctx 16384`, `b2560`, `ub96`, `-np 2 -ns 2 -pps`.
- Speed-only high-water mark: 16.53 tok/s, `ser 1,5`, `q6_0/q6_0`, `ctx 16384`, `b2304`, `ub104`.
- Key warning: direct `ser 1,*` has produced factual-output corruption, so it is not a quality record yet.

## Active Track

- Quality-preserving Routerclamp: find the smallest high-quality active expert path that can masquerade as the full 30B MoE.
- Adaptive SER: use reduced expert modes only when they are safe, with fallback to the proven `ser 3,1` path for factual or high-entropy work.
- Workload gating: identify prompt classes where `ser 1,*` is reliable enough to use as an acceleration lane.
- Source-level SER instrumentation: inspect ik_llama.cpp's Smart Expert Reduction path so we can move from blunt flags to layer/task-aware control.
- QPU-guided search: use IBM Quantum as a candidate sampler for compact, binary experiment decisions after the classical harness scores real runs.

## Latest Routerclamp Findings

- Added opt-in `LLAMA_ROUTER_CENSUS=/path` logging for the SER router path. It records layer, selected expert IDs, selected probabilities, active expert count, and selected probability mass during decode-sized rows.
- First census on the record family (`ser 3,1`, cheap `24:30` top-2) showed the middle band `24:30` has the lowest selected top-3 mass (`~0.12-0.14`) and broad expert spread, while late layers are more concentrated but still not safely top-1.
- Inverted Routerclamp failed: global `ser 1,5` with selected layer rescues reached 15.63 tok/s, but every strict Serbia probe immediately collapsed into junk. The smallest quality-safe subnetwork is not a top-1 expert path with a few rescued bands.
- Hotset clamp was prototyped and backed out. A Mars-only top-48 hotset slowed badly and caused repetition, and the implementation point polluted normal throughput. Router census remains useful; hotset clamping needs a broader corpus plus fallback and a less invasive implementation point.
- App-state gating matters: one-off runs without pausing the Codex GPU helper fell to ~2-3 tok/s; the same short record-shape smoke recovered to 13.30 tok/s with the helper paused. Serious benchmark scripts should keep using the helper pause/resume wrapper.
- Added `LLAMA_SER_FULL_RANGES` to the ik_llama.cpp SER path so experiments can restore full routing in arbitrary layer bands such as `18:24` or `24:30`.
- `ser 2,0.65` plus full bands `18:24`, `24:30`, `30:36`, or `42:48` can pass strict Serbia/Mars/prime gates in the cold scout, proving quality can be rescued by layer localization.
- On the record-shaped lane (`b2560`, `ub96`, `-np 2 -ns 2 -pps`), static band rescue stayed quality-clean but slower than `ser 3,1`.
- Exact/top-2-ish routes (`ser 2,1`, `ser 2,0.85`, `ser 2,0.95`) are not quality-safe: they drift subjects, invent geography, and corrupt executable code even when specific bands are rescued.
- Added `LLAMA_SER_CHEAP_RANGES` plus `LLAMA_SER_CHEAP_MIN/THRESH` to flip the experiment: keep `ser 3,1` globally, then cheapen only selected layer bands.
- Best quality-safe cheap-band result so far: global `ser 3,1` with layers `24:30` forced to top-2 (`LLAMA_SER_CHEAP_RANGES=24:30`, min `2`, thresh `1.0`). It passed strict Serbia/Mars/prime and reached 13.58 tok/s in the ridge sweep.
- Individually safe top-2 bands also include `6:12` and `12:18`, but stacking them with `24:30` broke executable prime code. Safe islands are non-additive.
- Current lesson: fewer experts globally is the wrong lever by itself. The active leap is task-aware or phase-aware Routerclamp: cheapen only safe layers, and possibly only during decode after a full-quality prompt read.

## Serious Moonshots

- Hybrid MoE Split: CPU experts, GPU shared compute.
  - Hypothesis: keep expert-heavy MoE work on CPU/SSD-friendly paths while offloading non-expert/shared dense compute to the Intel GPU or Metal path.
  - Status: parked but real. It is not the next safest step because prior GPU/offload attempts were unreliable, but it belongs in the lab as a possible split-compute breakthrough.
  - Test shape: only revisit after Routerclamp source work, and measure quality plus page faults rather than raw tok/s alone.
- Hot-expert GGUF repack: keep shared weights and routers intact, preserve hot experts at higher quality, and demote or remove cold experts.
- Self-speculative MoE: use a reduced-expert Qwen path as a draft model and verify with the proven fuller path.
  - Status: tested with `llama-speculative` on 2026-05-26.
  - Patch made: `llama-speculative` now honors draft context, draft KV types, and `LLAMA_DRAFT_SER`.
  - Findings: Qwen3-0.6B Q8 draft needed a BOS override but accepted 0/128 drafted tokens. Same-target drafts with `LLAMA_DRAFT_SER=1,5`, `2,0.65`, and even `3,1` also accepted 0 tokens in the current example path.
  - Conclusion: do not spend more benchmark time here until we either fix the speculative example's Qwen alignment or build a purpose-trained/metadata-compatible draft.
- Router census: log selected experts by layer/token, then build hot sets and fallback thresholds from measured behavior instead of guesses.
- SSD cold-expert warehouse: use SSD for cold experts only, with hot expert residency and prefetch informed by router statistics.

## Deprioritized Or Banned For Now

- `--run-time-repack`: measured as a major slowdown on this MacBook.
- Direct factual use of `ser 1,*`: fast but currently corrupts factual answers.
- `ser 2,1`, `ser 4,1`, `ser 5,1`: observed as slow or poisoned on this workload.
- Merge/Hadamard experiment flags such as `-mqkv`, `-muge`, `-khad`, `-vhad`: no useful win in current tests.
