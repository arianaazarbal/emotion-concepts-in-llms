"""Plot average residual-stream activation magnitude across token positions."""

import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.emotion_probe import load_model_and_tokenizer
from src.utils.utils import load_json


@torch.no_grad()
def main(
    model: str = "gemma2_9b",
    emotion: str = "afraid",
    layer: int = 26,
    max_samples: int = 50,
    batch_size: int = 8,
    max_length: int = 2048,
    seed: int = 0,
    output_dir: str = "results/diagnostics",
):
    """Compute and plot mean residual-stream norm per token position.

    Args:
        model: Short model name registered in data/model_names.json.
        emotion: Emotion key to load stories for.
        layer: Transformer layer to inspect.
        max_samples: Number of stories to process.
        batch_size: Forward-pass batch size.
        max_length: Tokenizer truncation cap.
        seed: RNG seed for story shuffle.
        output_dir: Where to save the plot.
    """
    import random

    stories_path = f"results/stories/{model}/emotion_to_stories.json"
    stories = load_json(stories_path)[emotion]
    rng = random.Random(seed)
    rng.shuffle(stories)
    stories = stories[:max_samples]

    llm, tokenizer = load_model_and_tokenizer(model)
    device = next(llm.parameters()).device

    all_norms = []

    n_batches = (len(stories) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = stories[b * batch_size : (b + 1) * batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        attn = enc["attention_mask"]
        outputs = llm(**enc, output_hidden_states=True)
        h = outputs.hidden_states[layer]  # [B, S, H]

        for i in range(h.shape[0]):
            true_len = int(attn[i].sum().item())
            norms = h[i, :true_len].float().norm(dim=-1).cpu().numpy()
            all_norms.append(norms)
        print(f"Batch {b + 1}/{n_batches}")

    max_len = max(len(n) for n in all_norms)
    padded = np.full((len(all_norms), max_len), np.nan)
    for i, n in enumerate(all_norms):
        padded[i, : len(n)] = n
    mean_norms = np.nanmean(padded, axis=0)
    counts = np.sum(~np.isnan(padded), axis=0)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.15 * max_len), 5))
    ax.plot(mean_norms, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean L2 norm")
    ax.set_title(
        f"Residual stream magnitude — {model} layer {layer} | "
        f"{emotion} ({len(all_norms)} samples)"
    )
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="BOS")
    ax.legend()
    fig.tight_layout()
    out_path = f"{output_dir}/residual_norms_{model}_layer{layer}_{emotion}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
