"""
For a given model and a JSONL of samples, compute the top-activating emotion vectors
at each token position and produce both a JSON dump and a per-sample heatmap plot.

Samples in the JSONL may be either:
  - plain text strings, or objects like {"text": "..."}
  - chat conversations: a list of {"role", "content"} dicts, or an object
    like {"messages": [...]}.
If the sample is in chat form, the model's chat template is applied first.

Emotion vectors are loaded from results/emotion_vectors/{model}/*_vectors.pt
(produced by src/extracting_emotion_vectors/extract_emotion_vectors.py).
"""

import os
import random

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.emotion_probe import (
    default_layer,
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import dump_json, load_jsonl


def is_chat_sample(sample) -> bool:
    """Detect whether a JSONL sample represents a chat conversation."""
    if isinstance(sample, dict) and "messages" in sample:
        return True
    if (
        isinstance(sample, list)
        and len(sample) > 0
        and isinstance(sample[0], dict)
        and "role" in sample[0]
        and "content" in sample[0]
    ):
        return True
    return False


def sample_to_text(sample, tokenizer) -> str:
    """Convert a JSONL sample into a tokenizable string."""
    if is_chat_sample(sample):
        messages = sample["messages"] if isinstance(sample, dict) else sample
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict) and "text" in sample:
        return sample["text"]
    raise ValueError(f"Unrecognized sample format: {sample!r}")


def top_k_per_token(tokens, scores, emotions, top_k):
    """For each token, return the top-k (emotion, score) pairs sorted desc."""
    topk = torch.topk(scores, k=top_k, dim=-1)
    out = []
    for i, tok in enumerate(tokens):
        entries = [
            {"emotion": emotions[idx.item()], "score": float(val)}
            for val, idx in zip(topk.values[i], topk.indices[i])
        ]
        out.append({"token": tok, "top": entries})
    return out


def plot_heatmap(tokens, scores, emotions, selected_indices, out_path, title, cosine_sim=False):
    """Plot a heatmap of selected emotions across the sequence.

    Args:
        tokens: list of token strings.
        scores: tensor of shape [seq_len, num_emotions].
        emotions: full list of emotion names.
        selected_indices: which emotion indices to display as rows.
        out_path: where to save the figure.
        title: plot title.
        cosine_sim: whether scores are cosine similarities (affects colorbar label).
    """
    seq_len, _ = scores.shape
    sub = scores[:, selected_indices].T.numpy()  # [num_selected, seq_len]
    selected_emotions = [emotions[i] for i in selected_indices]

    fig, ax = plt.subplots(
        figsize=(max(8, 0.35 * seq_len), max(4, 0.3 * len(selected_emotions)))
    )
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(sub).max(), vmax=np.abs(sub).max())
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(selected_emotions)))
    ax.set_yticklabels(selected_emotions, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="cosine similarity" if cosine_sim else "dot product")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    model: str,
    samples_jsonl: str,
    layer: int = None,
    top_k: int = 5,
    n_display_emotions: int = 15,
    max_samples: int = None,
    output_dir: str = None,
    start_at_nth_token: int = 0,
    seed: int = 0,
    cosine_sim: bool = False,
    denoised: bool = False,
    batch_size: int = 8,
    max_length: int = 2048,
    version: str = "",
):
    """Compute top-activating emotions per token for each sample in a JSONL.

    Args:
        model: Short model name registered in data/model_names.json.
        samples_jsonl: Path to a JSONL file. Each line is a sample (string, {"text": ...},
            list of {"role","content"}, or {"messages": [...]}).
        layer: Which transformer layer's residual stream to project onto emotion vectors.
            Defaults to ~2/3 through the model.
        top_k: Number of top emotions to keep per token in the JSON output.
        n_display_emotions: Number of randomly-selected emotions to show in heatmaps.
        max_samples: If set, only process the first N samples.
        output_dir: Where to save outputs. Defaults to results/top_emotions/{model}{_version}/.
        start_at_nth_token: Which emotion-vector extraction variant to load (default 0 = original layout).
        seed: RNG seed for reproducibility (governs which emotions are displayed).
        cosine_sim: Use cosine similarity instead of raw dot product.
        denoised: Use PCA-denoised emotion vectors.
        batch_size: Forward-pass batch size.
        max_length: Tokenizer truncation cap (in tokens).
        version: Optional version suffix; when non-empty, resolves model folders
            as ``{model}_{version}`` for both vector lookup and default output_dir.
    """
    model_tag = f"{model}_{version}" if version else model
    if output_dir is None:
        output_dir = f"results/top_emotions/{model_tag}"
    os.makedirs(output_dir, exist_ok=True)

    cache_suffix = "_cosine" if cosine_sim else ""
    cache_path = f"{output_dir}/probe_cache{cache_suffix}.pt"
    if os.path.exists(cache_path):
        print(f"[top_emotions] loading cached probe results from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        per_text_scores = cache["per_text_scores"]
        per_text_tokens = cache["per_text_tokens"]
        emotions_list = cache["emotions"]
        layer = cache["layer"]
        samples = load_jsonl(samples_jsonl)
        if max_samples is not None:
            samples = samples[:max_samples]
    else:
        llm, tokenizer = load_model_and_tokenizer(model)
        if layer is None:
            layer = default_layer(llm)
            print(f"[top_emotions] no layer specified, defaulting to layer {layer} (~2/3 through model)")

        emotions, vectors_by_layer = load_emotion_vectors(
            model,
            start_at_nth_token=start_at_nth_token,
            denoised=denoised,
            version=version,
        )
        if layer not in vectors_by_layer:
            raise ValueError(
                f"Layer {layer} not found in emotion vectors. Available: {sorted(vectors_by_layer.keys())}"
            )

        samples = load_jsonl(samples_jsonl)
        if max_samples is not None:
            samples = samples[:max_samples]

        texts = [sample_to_text(s, tokenizer) for s in samples]

        print(f"[top_emotions] probing {len(texts)} samples at layer {layer} (batch_size={batch_size})")
        probe = emotion_probe(
            texts=texts,
            model=llm,
            tokenizer=tokenizer,
            emotion_vectors_by_layer=vectors_by_layer,
            emotions=emotions,
            layers=[layer],
            aggregation="none",
            batch_size=batch_size,
            max_length=max_length,
            cosine_sim=cosine_sim,
        )
        per_text_scores = probe.scores[layer]
        per_text_tokens = probe.tokens
        emotions_list = probe.emotions

        torch.save(
            {"per_text_scores": per_text_scores, "per_text_tokens": per_text_tokens,
             "emotions": emotions_list, "layer": layer},
            cache_path,
        )
        print(f"[top_emotions] cached probe results to {cache_path}")

    rng = random.Random(seed)
    n_emotions = len(emotions_list)
    display_indices = sorted(rng.sample(range(n_emotions), min(n_display_emotions, n_emotions)))

    results = []
    for i, sample in enumerate(samples):
        scores = per_text_scores[i]
        tokens = per_text_tokens[i]
        bos = 1 if tokens and tokens[0] in ("<bos>", "<s>", "<|begin_of_text|>") else 0
        plot_tokens = tokens[bos:]
        plot_scores = scores[bos:]
        per_token = top_k_per_token(tokens, scores, emotions_list, top_k)
        results.append(
            {
                "sample_idx": i,
                "layer": layer,
                "is_chat": is_chat_sample(sample),
                "tokens": per_token,
            }
        )
        plot_heatmap(
            plot_tokens,
            plot_scores,
            emotions_list,
            display_indices,
            out_path=f"{output_dir}/sample_{i:04d}_layer{layer}.png",
            title=f"{model} | sample {i} | layer {layer} | {len(display_indices)} random emotions",
            cosine_sim=cosine_sim,
        )
        print(f"Processed sample {i + 1}/{len(samples)}")

    dump_json(results, f"{output_dir}/results.json")
    print(f"Saved {len(results)} results to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
