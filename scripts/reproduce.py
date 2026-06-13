"""Reproduce the five core findings of the emotion-concepts replication.

This is the one entry point that runs the whole replication end to end:
build the emotion vectors, then reproduce each finding. Every underlying step
is cached/idempotent, so reruns are cheap and you can re-enter at any finding.

Findings (select with ``--findings "[1,3]"``; default runs all):
  1. Denoising sharpens the emotion probe  (probe validation on stories + Pile,
     LLM-judged token fit, and the logit-lens negative result)
  2. Emotion scales with a swept numeric quantity
  3. Steering changes the substance of advice (not just tone)
  4. User-token vs. assistant-token steering
  5. Local vs. long-term (backstory) context

Prerequisites (vector extraction + denoising) run automatically before the
findings unless you pass ``--skip_build``.

Examples
--------
    # Full replication on the default model:
    python -m scripts.reproduce --model gemma2_9b

    # Quick smoke test (tiny sample sizes), only findings 1 and 3:
    python -m scripts.reproduce --model gemma2_9b --debug --findings "[1,3]"

    # A different verified model (llama_8b -> layer 21, qwen_7b -> layer 18):
    python -m scripts.reproduce --model qwen_7b
"""

import os
import subprocess
import sys

import fire

from src.utils.model_names import resolve_model_name, resolve_openrouter_id
from src.utils.utils import load_json

LAYER_PATH = "data/model_to_primary_layer.json"
JUDGE_MODEL = "openai/gpt-4o-mini"
DEFAULT_STRENGTHS = "[-200,-100,-50,0,50,100,200]"
ALL_FINDINGS = [1, 2, 3, 4, 5]


def _primary_layer(model: str) -> int:
    mapping = load_json(LAYER_PATH)
    layer = mapping.get(model)
    if layer is None:
        known = sorted(k for k in mapping if not k.startswith("_"))
        raise ValueError(
            f"No primary layer for {model!r} in {LAYER_PATH}. Known: {known}. "
            f"Add an entry there or pass --layer."
        )
    return int(layer)


def _run(name: str, *args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print("\n" + "=" * 70)
    print(f"[reproduce] {name}")
    print("  $ " + " ".join(cmd))
    print("=" * 70, flush=True)
    subprocess.run(cmd, check=True)


def reproduce(
    model: str = "gemma2_9b",
    layer: int = None,
    findings: list = None,
    start_at_nth_token: int = 50,
    n: int = 12,
    debug: bool = False,
    skip_build: bool = False,
    judge_model: str = JUDGE_MODEL,
    steer_n: int = 50,
):
    """Run the end-to-end replication for ``model``.

    Args:
        model: Short model name from data/model_names.json (e.g. gemma2_9b,
            llama_8b, qwen_7b).
        layer: Probe/steer layer; defaults to data/model_to_primary_layer.json.
        findings: Subset of [1,2,3,4,5] to run; None runs all.
        start_at_nth_token: Token offset for vector extraction/probing.
        n: Stories per (emotion, topic) during extraction.
        debug: Tiny sample sizes for a fast smoke test.
        skip_build: Skip extraction + denoising (use existing vectors).
        judge_model: OpenRouter slug for the steer_advice/Pile judge.
        steer_n: Completions per (emotion, strength) in the steering findings.
    """
    layer = layer if layer is not None else _primary_layer(model)
    findings = ALL_FINDINGS if findings is None else list(findings)
    sat = start_at_nth_token
    story_model = resolve_model_name(model)
    story_or = resolve_openrouter_id(model)

    if debug:
        n = 2
        steer_n = 4
    n_docs = 20 if debug else 200
    strengths = "[-100,0,100]" if debug else DEFAULT_STRENGTHS

    print(
        f"[reproduce] model={model} layer={layer} start_at={sat} "
        f"findings={findings} debug={debug}"
    )

    if not skip_build:
        extract_args = [
            "src.extracting_emotion_vectors.extract_emotion_vectors",
            "--model", model, "--n", str(n), "--start-at-nth-token", str(sat),
        ]
        if debug:
            extract_args += ["--max_topics", "2"]
        _run("build: extract emotion vectors", *extract_args)

        denoise_args = [
            "scripts.pipeline.denoise_emotion_vectors",
            "--model", model, "--story-model", story_model,
            "--variance-threshold", "0.5",
            "--start-at-nth-token", str(sat), "--target-layer", str(layer),
        ]
        if story_or:
            denoise_args += ["--story-model-openrouter", story_or]
        if debug:
            denoise_args += ["--max-stories", "20"]
        _run("build: denoise emotion vectors", *denoise_args)
        _symlink_denoised(model, sat, layer)

    base = ["--model", model, "--layer", str(layer)]
    sat_us = ["--start-at-nth-token", str(sat)]
    sat_sc = ["--start_at_nth_token", str(sat)]

    if 1 in findings:
        _run(
            "finding 1: probe on own stories (sanity)",
            "scripts.pipeline.analyze_emotion_on_stories", *base, *sat_us,
            "--emotion", "angry", "--version", "",
        )
        _run(
            "finding 1: top-activating tokens on the Pile (denoised)",
            "scripts.pipeline.analyze_emotion_on_pile", *base, *sat_sc,
            "--n_docs", str(n_docs), "--top_k_tokens", "10", "--denoised", "--version", "",
        )
        _run(
            "finding 1: LLM-judge token fit (raw vs denoised)",
            "scripts.pipeline.judge_emotion_on_pile", *base,
        )
        _run(
            "finding 1: logit lens (negative result; denoised)",
            "scripts.pipeline.logit_lens", *base, *sat_sc,
            "--top_k", "5", "--denoised", "--version", "",
        )

    if 2 in findings:
        _run(
            "finding 2: emotion scales with quantity",
            "scripts.experiments.prompt_templates_vary_quantity", *base, *sat_sc,
            "--denoised", "--cosine-sim", "--version", "",
        )

    if 3 in findings or 4 in findings:
        positions = ["all"]
        if 4 in findings:
            positions += ["user_only", "generation_only"]
        for pos in positions:
            _run(
                f"finding 3/4: steer advice (steer_positions={pos})",
                "scripts.experiments.steer_advice", *base, *sat_sc, "--denoised",
                f"--strengths={strengths}", "--N", str(steer_n),
                "--prompt-id", "confront_wait_01_rent",
                "--steer-positions", pos,
                "--judge_model", judge_model, "--max-parallel-completions", "20",
                "--version", "",
            )

    if 5 in findings:
        _run(
            "finding 5: local vs long-term (backstory) context",
            "scripts.experiments.probe_unspoken_emotions", *base, *sat_sc,
            "--denoised", "--version", "",
        )

    print("\n[reproduce] done. See results/ for outputs (and figures/README.md for plot slots).")


def _symlink_denoised(model: str, sat: int, layer: int) -> None:
    """Point `.../denoised` at the layer-specific dir so `--denoised` finds it."""
    vec_dir = f"results/emotion_vectors/{model}/start_at_{sat}"
    target = f"denoised_layer{layer}"
    if os.path.isdir(os.path.join(vec_dir, target)):
        link = os.path.join(vec_dir, "denoised")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
        print(f"[reproduce] symlinked {link} -> {target}")


if __name__ == "__main__":
    fire.Fire(reproduce)
