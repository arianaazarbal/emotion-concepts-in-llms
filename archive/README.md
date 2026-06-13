# Archived experiments

These experiments are **not part of the v1 replication** and are not covered by
the top-level install. They are kept here because the work is real and may be
revived, but they need heavier dependencies (and, in some cases, results we did
not cleanly reproduce). They are excluded from the curated pipeline so that a
fresh `uv sync` stays lightweight and works out of the box.

Nothing in the core `src/` or `scripts/` imports from this directory.

## Contents

### `reward_hacking/`
Emotion-steered reward hacking on ImpossibleBench / LiveCodeBench, plus the
`build_distress_eval/` harness that replicates *Gemma Needs Help* (Soligo,
Mikulik & Saunders, 2026). **Status: challenging / not cleanly reproduced** —
finding a setting where the open-weight models reliably hack was hard, and the
distress-eval target paper is on Gemma 3, not the Gemma 2 9B we used (see
`build_distress_eval/NOTES.md`).

Extra dependencies: `inspect-ai`, `bitsandbytes`, and a local vLLM/GPU setup.

### `bloom/`
Behavior-elicitation (Anthropic Bloom eval) under emotion steering:
self-preservation and delusion-sycophancy.

Extra dependencies: `inspect-ai`, local GPU.

## Running archived code
Because these scripts were moved, their module paths are now under `archive/`.
Imports inside them may reference the original `src.*` paths (still valid) but
expect to fix a path or two before anything here runs. Treat this directory as
a reference snapshot, not a maintained pipeline.
