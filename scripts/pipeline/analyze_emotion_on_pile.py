"""
Validate emotion vectors on out-of-distribution web text (The Pile).

Pulls a small chunk from a Pile-style HuggingFace dataset (default:
``monology/pile-uncopyrighted``), caches it locally under ``data/``, and
runs ``top_activating_tokens`` on it so we can inspect which tokens most
strongly fire each emotion vector across arbitrary web text. Writes a
self-contained HTML chip page alongside the usual json/md/png outputs.

Unlike ``analyze_emotion_on_stories.py``, there is no ``--emotion`` filter:
the corpus is emotion-agnostic and the analyzer reports top tokens for every
emotion vector from a single probe pass.

Outputs (TAG = ``{model}_{version}`` or ``{model}`` if version empty):
- results/top_tokens_on_pile/{TAG}[/start_at_token_{N}][/denoised_layer_{L}]/{metric}/
    top_tokens_layer{L}.{json,md,html}
    top_token_plots/{emotion}_layer{L}.png

``{metric}`` is always one of ``dot_product`` or ``cosine_sim`` so that runs
with different probe metrics never collide on disk.
"""

import os

import fire
from datasets import load_dataset

from src.analyzing_emotion_activations.top_activating_tokens import (
    main as run_top_activating_tokens,
)
from src.utils.utils import dump_jsonl


def _pile_cache_path(
    dataset: str, split: str, n_docs: int, seed: int, max_chars: int
) -> str:
    tag = dataset.replace("/", "__")
    return (
        f"data/pile_sample_{tag}_{split}_n{n_docs}_seed{seed}_maxchars{max_chars}.jsonl"
    )


def build_pile_samples_jsonl(
    dataset: str = "monology/pile-uncopyrighted",
    split: str = "train",
    n_docs: int = 200,
    seed: int = 0,
    max_chars: int = 4000,
    text_field: str = "text",
) -> str:
    """Stream the first ``n_docs`` non-empty documents from ``dataset`` and
    write them to a JSONL cache. If the cache file already exists it is reused.

    ``max_chars`` truncates each document at the character level (before
    tokenization) to keep sequence lengths and memory in check. ``seed`` is
    accepted for symmetry with other scripts but streaming-first-N is
    deterministic by default; we shuffle a larger stream buffer with it.
    """
    out_path = _pile_cache_path(dataset, split, n_docs, seed, max_chars)
    if os.path.exists(out_path):
        print(f"[pile] using cached samples {out_path}")
        return out_path

    os.makedirs("data", exist_ok=True)
    buffer_size = max(n_docs * 4, 500)
    print(
        f"[pile] streaming {dataset} split={split}, buffering {buffer_size} docs, "
        f"shuffling with seed={seed}, keeping first {n_docs}"
    )
    ds = load_dataset(dataset, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    samples = []
    for row in ds:
        txt = row.get(text_field)
        if not txt:
            continue
        txt = txt[:max_chars]
        samples.append({"text": txt})
        if len(samples) >= n_docs:
            break

    dump_jsonl(samples, out_path)
    print(f"[pile] wrote {len(samples)} samples to {out_path}")
    return out_path


def main(
    model: str,
    layer: int = None,
    n_docs: int = 200,
    max_chars: int = None,
    seed: int = 0,
    top_k_tokens: int = 10,
    context_window: int = 10,
    start_at_nth_token: int = 50,
    cosine_sim: bool = False,
    denoised: bool = False,
    version: str = "v0",
    dataset: str = "monology/pile-uncopyrighted",
    split: str = "train",
    text_field: str = "text",
    batch_size: int = 8,
    max_length: int = 2048,
    max_samples: int = None,
):
    """Run top_activating_tokens over a cached pile-style web-text sample.

    Args:
        model: Short model name registered in data/model_names.json.
        layer: Transformer layer whose residual stream is projected onto
            emotion vectors. Defaults to ~2/3 through the model.
        n_docs: How many documents to sample from the dataset (cached).
        max_chars: Character-level truncation applied before tokenization.
        seed: RNG seed (shuffles the streamed dataset and seeds torch).
        top_k_tokens: Top tokens to report per emotion.
        context_window: Tokens before/after the hit to show in HTML/plots.
        start_at_nth_token: Which emotion-vector extraction variant to load
            (directory suffix only — no tokens are skipped at probe time).
        cosine_sim: Use cosine similarity instead of raw dot product.
        denoised: Use PCA-denoised emotion vectors.
        version: Version suffix for model-folder lookup.
        dataset: HuggingFace dataset id to stream from.
        split: Dataset split.
        text_field: Which field of each row holds the document text.
        batch_size: Forward-pass batch size.
        max_length: Tokenizer truncation cap (in tokens).
        max_samples: If set, cap analyzer input to this many (debug mode).
    """
    samples_jsonl = build_pile_samples_jsonl(
        dataset=dataset,
        split=split,
        n_docs=n_docs,
        seed=seed,
        max_chars=max_chars,
        text_field=text_field,
    )

    model_tag = f"{model}_{version}" if version else model
    suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
    if denoised:
        suffix += f"/denoised_layer_{layer if layer is not None else 'default'}"
    suffix += "/cosine_sim" if cosine_sim else "/dot_product"
    output_dir = f"results/top_tokens_on_pile/{model_tag}{suffix}"

    print(
        f"[pipeline] analyze_emotion_on_pile model={model} version={version or '(none)'} "
        f"n_docs={n_docs} denoised={denoised} cosine_sim={cosine_sim}"
    )
    print(f"[pipeline] writing outputs to {output_dir}")

    run_top_activating_tokens(
        model=model,
        samples_jsonl=samples_jsonl,
        layer=layer,
        top_k=top_k_tokens,
        context_window=context_window,
        max_samples=max_samples,
        output_dir=output_dir,
        seed=seed,
        start_at_nth_token=start_at_nth_token,
        cosine_sim=cosine_sim,
        denoised=denoised,
        batch_size=batch_size,
        max_length=max_length,
        version=version,
    )

    print(f"[pipeline] done. samples={samples_jsonl}, output={output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
