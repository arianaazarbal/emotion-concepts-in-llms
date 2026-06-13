"""
Validate emotion vectors on the original emotion-labeled training stories.

For a fixed (model, version, emotion), turn the model's stories for that emotion
into a JSONL of {"text": ...} samples (seeded shuffle, capped at max_lines),
then run both top_activating_emotions and top_activating_tokens on it.

This is a same-distribution check: samples come from the same stories the
vectors were derived from. For out-of-distribution validation on arbitrary
text, see ``scripts/analyze_emotion_on_pile.py``.

Version is an optional suffix; with version="v0" the pipeline reads/writes under
``{model}_v0``. An empty version falls back to ``{model}`` for backward compat.

Outputs (with TAG = model if version == "" else f"{model}_{version}"):
- results/emotion_samples/{TAG}/{emotion}.jsonl
- results/top_emotions_on_stories/{TAG}/{emotion}[/start_at_token_{N}][/denoised_layer_{L}]/{metric}/...
- results/top_tokens_on_stories/{TAG}/{emotion}[/start_at_token_{N}][/denoised_layer_{L}]/{metric}/...

``{metric}`` is always ``dot_product`` or ``cosine_sim`` so runs with
different probe metrics never collide on disk.
"""

import os
import random

import fire

from src.analyzing_emotion_activations.top_activating_emotions import (
    main as run_top_activating_emotions,
)
from src.analyzing_emotion_activations.top_activating_tokens import (
    main as run_top_activating_tokens,
)
from src.utils.utils import dump_jsonl, load_json


def build_samples_jsonl(
    model: str, emotion: str, max_lines: int, seed: int, version: str = ""
) -> str:
    """Load stories for (model, version, emotion), shuffle with seed, cap, and dump JSONL.

    Returns the path to the written JSONL.
    """
    model_tag = f"{model}_{version}" if version else model
    stories_path = f"results/stories/{model_tag}/emotion_to_stories.json"
    stories_by_emotion = load_json(stories_path)
    if emotion not in stories_by_emotion:
        raise KeyError(
            f"Emotion '{emotion}' not in {stories_path}. Available: {sorted(stories_by_emotion.keys())[:20]}..."
        )
    stories = list(stories_by_emotion[emotion])
    print(
        f"[samples] loaded {len(stories)} stories for emotion='{emotion}' from {stories_path}"
    )

    rng = random.Random(seed)
    rng.shuffle(stories)
    if max_lines is not None and len(stories) > max_lines:
        stories = stories[:max_lines]
        print(f"[samples] truncated to {max_lines} stories (seed={seed})")

    out_dir = f"results/emotion_samples/{model_tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{emotion}.jsonl"
    dump_jsonl([{"text": s} for s in stories], out_path)
    print(f"[samples] wrote {len(stories)} samples to {out_path}")
    return out_path


def main(
    model: str,
    emotion: str,
    layer: int,
    max_lines: int = 50,
    seed: int = 0,
    top_k_emotions: int = 5,
    top_k_tokens: int = 10,
    context_window: int = 10,
    start_at_nth_token: int = 50,
    cosine_sim: bool = False,
    denoised: bool = False,
    version: str = "v0",
):
    """Run the (sample-prep + top_emotions + top_tokens) pipeline for one (model, version, emotion).

    Args:
        model: Short model name registered in data/model_names.json.
        emotion: Emotion key present in results/stories/{model}_{version}/emotion_to_stories.json.
        layer: Transformer layer whose residual stream is projected onto emotion vectors.
        max_lines: Cap on number of stories used (after seeded shuffle). None = use all.
        seed: RNG seed for the shuffle (governs reproducibility of the sample subset).
        top_k_emotions: top_k for top_activating_emotions.
        top_k_tokens: top_k for top_activating_tokens.
        context_window: context window for top_activating_tokens.
        start_at_nth_token: Which emotion-vector extraction variant to load (default 0 = original layout).
        cosine_sim: Use cosine similarity instead of raw dot product.
        denoised: Use PCA-denoised emotion vectors.
        version: Version suffix; ``""`` falls back to the old ``{model}`` layout.
    """
    print(
        f"[pipeline] analyze_emotion model={model} version={version or '(none)'} emotion={emotion} layer={layer} max_lines={max_lines} seed={seed}"
    )

    samples_jsonl = build_samples_jsonl(
        model, emotion, max_lines=max_lines, seed=seed, version=version
    )

    model_tag = f"{model}_{version}" if version else model
    suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
    if denoised:
        suffix += f"/denoised_layer_{layer}"
    suffix += "/cosine_sim" if cosine_sim else "/dot_product"

    if False:
        top_emotions_out = (
            f"results/top_emotions_on_stories/{model_tag}/{emotion}{suffix}"
        )
        print(f"[pipeline] running top_activating_emotions -> {top_emotions_out}")
        run_top_activating_emotions(
            model=model,
            samples_jsonl=samples_jsonl,
            layer=layer,
            top_k=top_k_emotions,
            output_dir=top_emotions_out,
            start_at_nth_token=start_at_nth_token,
            seed=seed,
            cosine_sim=cosine_sim,
            denoised=denoised,
            version=version,
        )

    top_tokens_out = f"results/top_tokens_on_stories/{model_tag}/{emotion}{suffix}"
    print(f"[pipeline] running top_activating_tokens -> {top_tokens_out}")
    run_top_activating_tokens(
        model=model,
        samples_jsonl=samples_jsonl,
        layer=layer,
        top_k=top_k_tokens,
        context_window=context_window,
        output_dir=top_tokens_out,
        seed=seed,
        start_at_nth_token=start_at_nth_token,
        cosine_sim=cosine_sim,
        denoised=denoised,
        version=version,
    )

    print(
        f"[pipeline] done. samples={samples_jsonl}, top_tokens={top_tokens_out}"
    )


if __name__ == "__main__":
    fire.Fire(main)
