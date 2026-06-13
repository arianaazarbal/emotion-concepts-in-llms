"""Denoise emotion vectors by projecting out dominant directions from neutral stories.

Pipeline:
1. Generate neutral stories (no emotion prompt) via OpenRouter, cache to disk.
2. Run each neutral story through the model and collect per-token residual-stream
   activations at every layer.
3. Run PCA on the neutral activations. Find the top-k components explaining
   a target fraction of total variance (default 50%).
4. For each emotion vector, project out these top-k neutral components,
   producing a "denoised" vector that retains the emotion-specific signal
   while removing general narrative structure.
5. Save denoised vectors alongside the originals.
"""

import os
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from src.extracting_emotion_vectors.extract_emotion_vectors import (
    parse_stories,
)
from src.utils.emotion_probe import load_emotion_vectors, load_model_and_tokenizer
from src.utils.generate import generate_with_fallback
from src.utils.utils import dump_json, load_json, load_yaml


def generate_neutral_stories(
    model_full_name: str,
    n_per_topic: int = 12,
    max_concurrency: int = 200,
    local_only: bool = False,
    openrouter_model: str | None = None,
):
    """Generate neutral (emotion-free) stories across all topics.

    Tries the OpenRouter slug (if provided) first; on probe failure or when
    ``openrouter_model`` is None, routes through local vLLM using
    ``model_full_name`` (the HuggingFace id).
    """
    topics = load_json("data/topics.json")
    prompt_template = load_yaml("data/neutral_story_generation.yaml")["prompt"]
    prompts = [
        prompt_template.format(n_stories=n_per_topic, topic=topic)
        for topic in topics
    ]

    source, responses = generate_with_fallback(
        model=model_full_name,
        prompts=prompts,
        N=1,
        local_only=local_only,
        max_concurrency=max_concurrency,
        openrouter_model=openrouter_model,
    )
    print(f"[neutral] generated via {source}")

    all_stories = []
    for topic, resp_list in zip(topics, responses):
        response = resp_list[0] if resp_list else None
        parsed = parse_stories(response) if response else []
        all_stories.extend(parsed)
    print(f"[neutral] collected {len(all_stories)} neutral stories")
    return all_stories


def _iter_neutral_activation_chunks(
    model, tokenizer, stories, start_at_nth_token, batch_size, max_length=None,
    target_layer=None,
):
    """Yield per-layer token activation chunks for each batch of stories."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    n_batches = (len(stories) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch = stories[b_idx * batch_size : (b_idx + 1) * batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        attn = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        batch_chunks = {}
        for layer_idx, h in enumerate(outputs.hidden_states[1:], start=1):
            if target_layer is not None and layer_idx != target_layer:
                continue
            layer_chunks = []
            for i in range(h.shape[0]):
                true_len = int(attn[i].sum().item())
                if true_len <= start_at_nth_token:
                    continue
                layer_chunks.append(h[i, start_at_nth_token:true_len].float().cpu())
            if layer_chunks:
                batch_chunks[layer_idx] = torch.cat(layer_chunks, dim=0)
        done = min((b_idx + 1) * batch_size, len(stories))
        yield batch_chunks, b_idx + 1, n_batches, done
        del outputs

def _count_neutral_tokens_per_layer(
    model, tokenizer, stories, start_at_nth_token, batch_size, max_length=None,
    target_layer=None,
):
    """Count the neutral tokens that will contribute to each layerwise PCA.

    Uses the tokenizer alone — every layer receives the same set of token
    positions, so one pass without a forward is sufficient.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    total_tokens = 0
    n_batches = (len(stories) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch = stories[b_idx * batch_size : (b_idx + 1) * batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attn = enc["attention_mask"]
        for i in range(attn.shape[0]):
            true_len = int(attn[i].sum().item())
            if true_len > start_at_nth_token:
                total_tokens += true_len - start_at_nth_token
        done = min((b_idx + 1) * batch_size, len(stories))
        print(
            f"  [neutral counts] batch {b_idx + 1}/{n_batches}, "
            f"{done}/{len(stories)} stories"
        )

    if total_tokens == 0:
        return {}
    if target_layer is not None:
        return {target_layer: total_tokens}
    num_hidden_layers = model.config.num_hidden_layers
    return {layer: total_tokens for layer in range(1, num_hidden_layers + 1)}


def _flush_ipca_buffer(
    ipca_by_layer,
    n_components_by_layer,
    buffer_by_layer,
    buffered_tokens_by_layer,
    remaining_tokens_by_layer,
    layer,
    pca_batch_tokens,
    final=False,
):
    """Flush buffered activations into a layer's IncrementalPCA fit."""
    n_components = n_components_by_layer[layer]
    flush_threshold = max(pca_batch_tokens, n_components)
    if buffered_tokens_by_layer[layer] == 0:
        return

    while buffered_tokens_by_layer[layer] >= flush_threshold or (
        final and buffered_tokens_by_layer[layer] >= n_components
    ):
        current = torch.cat(buffer_by_layer[layer], dim=0)
        current_size = int(current.shape[0])
        chunk_size = min(current_size, flush_threshold)
        remainder = remaining_tokens_by_layer[layer] - chunk_size
        if remainder != 0 and remainder < n_components:
            chunk_size = remaining_tokens_by_layer[layer] - n_components
        if chunk_size < n_components:
            buffer_by_layer[layer] = [current]
            buffered_tokens_by_layer[layer] = current_size
            return

        ipca_by_layer[layer].partial_fit(current[:chunk_size].numpy())
        remaining_tokens_by_layer[layer] -= chunk_size

        leftover = current[chunk_size:]
        if leftover.numel() == 0:
            buffer_by_layer[layer] = []
            buffered_tokens_by_layer[layer] = 0
            return
        buffer_by_layer[layer] = [leftover]
        buffered_tokens_by_layer[layer] = int(leftover.shape[0])

    if final and buffered_tokens_by_layer[layer] > 0:
        current = torch.cat(buffer_by_layer[layer], dim=0)
        ipca_by_layer[layer].partial_fit(current.numpy())
        remaining_tokens_by_layer[layer] -= int(current.shape[0])
        buffer_by_layer[layer] = []
        buffered_tokens_by_layer[layer] = 0


def compute_pca_components_incremental(
    model,
    tokenizer,
    stories,
    start_at_nth_token,
    batch_size,
    variance_threshold=0.5,
    max_length=None,
    max_pca_components=512,
    pca_batch_tokens=4096,
    target_layer=None,
):
    """Fit layerwise IncrementalPCA without storing all neutral activations."""
    if target_layer is not None:
        num_hidden_layers = model.config.num_hidden_layers
        if target_layer < 1 or target_layer > num_hidden_layers:
            raise ValueError(
                f"target_layer must be in [1, {num_hidden_layers}], got {target_layer}"
            )
    token_counts_by_layer = _count_neutral_tokens_per_layer(
        model,
        tokenizer,
        stories,
        start_at_nth_token,
        batch_size,
        max_length=max_length,
        target_layer=target_layer,
    )
    if not token_counts_by_layer:
        raise ValueError(
            "No neutral activations were collected. Lower start_at_nth_token or add longer stories."
        )

    hidden_size = model.config.hidden_size
    ipca_by_layer = {}
    n_components_by_layer = {}
    for layer, n_tokens in sorted(token_counts_by_layer.items()):
        n_components = min(n_tokens, hidden_size, max_pca_components)
        ipca_by_layer[layer] = IncrementalPCA(n_components=n_components)
        n_components_by_layer[layer] = n_components
        print(
            f"  [IncrementalPCA] layer {layer}: fitting up to {n_components} "
            f"components from {n_tokens} tokens"
        )

    buffer_by_layer = defaultdict(list)
    buffered_tokens_by_layer = defaultdict(int)
    remaining_tokens_by_layer = dict(token_counts_by_layer)
    n_batches = (len(stories) + batch_size - 1) // batch_size

    for batch_chunks, batch_idx, _, done in _iter_neutral_activation_chunks(
        model,
        tokenizer,
        stories,
        start_at_nth_token,
        batch_size,
        max_length=max_length,
        target_layer=target_layer,
    ):
        for layer, chunk in batch_chunks.items():
            buffer_by_layer[layer].append(chunk)
            buffered_tokens_by_layer[layer] += int(chunk.shape[0])
            _flush_ipca_buffer(
                ipca_by_layer,
                n_components_by_layer,
                buffer_by_layer,
                buffered_tokens_by_layer,
                remaining_tokens_by_layer,
                layer,
                pca_batch_tokens,
            )
        print(
            f"  [IncrementalPCA] batch {batch_idx}/{n_batches}, "
            f"{done}/{len(stories)} stories"
        )

    for layer in sorted(ipca_by_layer):
        _flush_ipca_buffer(
            ipca_by_layer,
            n_components_by_layer,
            buffer_by_layer,
            buffered_tokens_by_layer,
            remaining_tokens_by_layer,
            layer,
            pca_batch_tokens,
            final=True,
        )
        if remaining_tokens_by_layer[layer] != 0:
            raise RuntimeError(
                f"Layer {layer} still has {remaining_tokens_by_layer[layer]} tokens after IncrementalPCA fit."
            )

    components_by_layer = {}
    pca_diagnostics = {}
    for layer, ipca in sorted(ipca_by_layer.items()):
        n_tokens = token_counts_by_layer[layer]
        max_components = n_components_by_layer[layer]
        cumvar = ipca.explained_variance_ratio_.cumsum()
        k = int((cumvar < variance_threshold).sum()) + 1
        k = min(k, max_components)

        print(
            f"  [IncrementalPCA] layer {layer}: {k} components explain "
            f"{cumvar[k-1]*100:.1f}% of variance (target {variance_threshold*100:.0f}%)"
        )
        components_by_layer[layer] = torch.tensor(
            ipca.components_[:k], dtype=torch.float32
        )
        pca_diagnostics[layer] = {
            "k": k,
            "n_tokens": n_tokens,
            "hidden_size": hidden_size,
            "max_components_fit": max_components,
            "variance_explained_by_k": float(cumvar[k - 1]),
            "per_component_variance": ipca.explained_variance_ratio_[:max_components].tolist(),
            "cumulative_variance": cumvar[:max_components].tolist(),
        }
    return components_by_layer, pca_diagnostics


def project_out(vectors_by_layer, components_by_layer):
    """Project neutral PCA components out of emotion vectors.

    For each layer, removes the projection onto each of the top-k neutral
    components: v_denoised = v - sum_i (v . c_i) * c_i
    """
    denoised = {}
    for layer, vecs in vectors_by_layer.items():
        if layer not in components_by_layer:
            denoised[layer] = vecs
            continue
        C = components_by_layer[layer]  # [k, hidden]
        projections = vecs @ C.T  # [num_emotions, k]
        denoised[layer] = vecs - projections @ C  # [num_emotions, hidden]
    return denoised


def _save_diagnostics(
    report_dir, emotions, original_by_layer, denoised_by_layer,
    pca_diagnostics, variance_threshold, model, start_at_nth_token,
    target_layer=None,
):
    """Generate and save denoising diagnostics: JSON report + plots."""
    if target_layer is not None:
        layers = [target_layer]
    else:
        layers = sorted(original_by_layer.keys())

    per_layer_report = {}
    for layer in layers:
        orig = original_by_layer[layer]  # [num_emotions, hidden]
        deno = denoised_by_layer[layer]
        orig_norms = orig.norm(dim=-1)
        deno_norms = deno.norm(dim=-1)
        norm_ratio = deno_norms / orig_norms.clamp(min=1e-9)

        pca_info = pca_diagnostics.get(layer, {})
        per_emotion = []
        for e_idx, emotion in enumerate(emotions):
            per_emotion.append({
                "emotion": emotion,
                "original_norm": float(orig_norms[e_idx]),
                "denoised_norm": float(deno_norms[e_idx]),
                "norm_ratio": float(norm_ratio[e_idx]),
            })

        per_layer_report[layer] = {
            "pca_k": pca_info.get("k"),
            "variance_explained": pca_info.get("variance_explained_by_k"),
            "n_tokens_used": pca_info.get("n_tokens"),
            "mean_original_norm": float(orig_norms.mean()),
            "mean_denoised_norm": float(deno_norms.mean()),
            "mean_norm_ratio": float(norm_ratio.mean()),
            "median_norm_ratio": float(norm_ratio.median()),
            "per_emotion": per_emotion,
        }

    summary = {
        "model": model,
        "start_at_nth_token": start_at_nth_token,
        "variance_threshold": variance_threshold,
        "num_emotions": len(emotions),
        "layers": {
            str(layer): {key: value for key, value in info.items() if key != "per_emotion"}
            for layer, info in per_layer_report.items()
        },
    }
    dump_json(summary, f"{report_dir}/summary.json")
    dump_json(per_layer_report, f"{report_dir}/full_report.json")

    for layer in layers:
        info = per_layer_report[layer]
        pca_info = pca_diagnostics.get(layer, {})

        cumvar = pca_info.get("cumulative_variance", [])
        per_comp = pca_info.get("per_component_variance", [])
        k = pca_info.get("k", 0)
        if cumvar:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            n_show = min(len(cumvar), max(k * 3, 50))
            ax1.plot(range(1, n_show + 1), cumvar[:n_show], "b-", linewidth=1.5)
            ax1.axhline(variance_threshold, color="red", linestyle="--", alpha=0.7,
                        label=f"threshold={variance_threshold}")
            ax1.axvline(k, color="green", linestyle="--", alpha=0.7, label=f"k={k}")
            ax1.set_xlabel("Number of components")
            ax1.set_ylabel("Cumulative variance explained")
            ax1.set_title(f"Layer {layer}: PCA cumulative variance")
            ax1.legend()

            ax2.bar(range(1, min(k + 1, len(per_comp) + 1)),
                    per_comp[:k], color="steelblue")
            ax2.set_xlabel("Component index")
            ax2.set_ylabel("Variance explained")
            ax2.set_title(f"Layer {layer}: variance per component (top {k})")
            fig.tight_layout()
            fig.savefig(f"{report_dir}/pca_variance_layer{layer}.png", dpi=150)
            plt.close(fig)

        orig_norms = [e["original_norm"] for e in info["per_emotion"]]
        deno_norms = [e["denoised_norm"] for e in info["per_emotion"]]
        ratios = [e["norm_ratio"] for e in info["per_emotion"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(emotions))
        ax1.bar(x - 0.2, orig_norms, 0.4, label="original", color="steelblue", alpha=0.8)
        ax1.bar(x + 0.2, deno_norms, 0.4, label="denoised", color="coral", alpha=0.8)
        ax1.set_xticks(x[::max(1, len(x) // 20)])
        ax1.set_xticklabels([emotions[i] for i in x[::max(1, len(x) // 20)]],
                            rotation=90, fontsize=6)
        ax1.set_ylabel("L2 norm")
        ax1.set_title(f"Layer {layer}: emotion vector norms")
        ax1.legend()

        ax2.hist(ratios, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax2.axvline(np.median(ratios), color="red", linestyle="--",
                    label=f"median={np.median(ratios):.3f}")
        ax2.set_xlabel("Denoised / Original norm ratio")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Layer {layer}: norm retention distribution")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f"{report_dir}/norms_layer{layer}.png", dpi=150)
        plt.close(fig)


def main(
    model: str,
    start_at_nth_token: int = 50,
    variance_threshold: float = 0.5,
    n_per_topic: int = 12,
    story_model: str = "google/gemma-2-9b-it",
    story_model_openrouter: str | None = None,
    batch_size: int = 8,
    max_concurrency: int = 200,
    seed: int = 0,
    max_stories: int = None,
    max_length: int = None,
    pca_batch_tokens: int = 4096,
    target_layer: int = None,
    local_only: bool = False,
):
    """Run the full denoising pipeline.

    Args:
        model: Short model name registered in data/model_names.json (the model whose vectors we denoise).
        start_at_nth_token: Token offset for activation extraction (match the original vectors).
        variance_threshold: Fraction of neutral variance to project out (default 0.5).
        n_per_topic: Number of neutral stories to generate per topic.
        story_model: HuggingFace model ID used when neutral stories fall back to local vLLM.
        story_model_openrouter: Optional OpenRouter slug for the story model. If set, we
            probe OpenRouter first; otherwise we go straight to local vLLM.
        batch_size: Forward-pass batch size for activation extraction.
        max_concurrency: Max concurrent OpenRouter requests.
        seed: RNG seed.
        max_stories: Cap on neutral stories to use (for quick testing).
        max_length: Optional tokenizer truncation cap for neutral stories.
        pca_batch_tokens: Approximate token chunk size per IncrementalPCA partial fit.
        target_layer: Optional 1-indexed residual-stream layer to denoise. If provided,
            only that layer's neutral activations/PCA are computed and only that layer
            is changed in the saved vectors.
    """
    torch.manual_seed(seed)

    neutral_dir = f"results/neutral_stories/{model}"
    os.makedirs(neutral_dir, exist_ok=True)
    stories_path = f"{neutral_dir}/neutral_stories.json"

    if os.path.exists(stories_path):
        print(f"[denoise] loading cached neutral stories from {stories_path}")
        neutral_stories = load_json(stories_path)
    else:
        neutral_stories = generate_neutral_stories(
            story_model,
            n_per_topic=n_per_topic,
            max_concurrency=max_concurrency,
            local_only=local_only,
            openrouter_model=story_model_openrouter,
        )
        dump_json(neutral_stories, stories_path)
        print(f"[denoise] cached {len(neutral_stories)} neutral stories to {stories_path}")

    if max_stories is not None:
        neutral_stories = neutral_stories[:max_stories]

    layer_suffix = f"_layer{target_layer}" if target_layer is not None else ""
    cache_suffix = f"_max{len(neutral_stories)}" if max_stories is not None else ""
    variance_tag = str(variance_threshold).replace(".", "p")
    pca_stem = (
        f"{neutral_dir}/neutral_pca_start{start_at_nth_token}_var{variance_tag}"
        f"_bt{pca_batch_tokens}{layer_suffix}{cache_suffix}"
    )
    pca_path = f"{pca_stem}.pt"
    pca_diag_path = f"{pca_stem}.json"

    if os.path.exists(pca_path) and os.path.exists(pca_diag_path):
        print(f"[denoise] loading cached neutral PCA from {pca_path}")
        components_by_layer = torch.load(pca_path, map_location="cpu", weights_only=True)
        pca_diagnostics = load_json(pca_diag_path)
        pca_diagnostics = {int(layer): info for layer, info in pca_diagnostics.items()}
    else:
        llm, tokenizer = load_model_and_tokenizer(model)
        print(
            f"[denoise] fitting IncrementalPCA on {len(neutral_stories)} "
            f"neutral stories"
        )
        components_by_layer, pca_diagnostics = compute_pca_components_incremental(
            llm,
            tokenizer,
            neutral_stories,
            start_at_nth_token,
            batch_size,
            variance_threshold=variance_threshold,
            max_length=max_length,
            pca_batch_tokens=pca_batch_tokens,
            target_layer=target_layer,
        )
        torch.save(components_by_layer, pca_path)
        dump_json(pca_diagnostics, pca_diag_path)
        print(f"[denoise] cached neutral PCA to {pca_path}")
        del llm
        torch.cuda.empty_cache()

    print("[denoise] loading original emotion vectors")
    emotions, vectors_by_layer = load_emotion_vectors(model, start_at_nth_token=start_at_nth_token)

    print("[denoise] projecting out neutral components")
    denoised_by_layer = project_out(vectors_by_layer, components_by_layer)

    out_dir = (
        f"results/emotion_vectors/{model}/start_at_{start_at_nth_token}/denoised{layer_suffix}"
        if start_at_nth_token != 0
        else f"results/emotion_vectors/{model}/denoised{layer_suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)
    for e_idx, emotion in enumerate(emotions):
        emotion_vecs = {
            layer: denoised_by_layer[layer][e_idx] for layer in denoised_by_layer
        }
        torch.save(emotion_vecs, f"{out_dir}/{emotion}_vectors.pt")
    print(f"[denoise] saved {len(emotions)} denoised emotion vectors to {out_dir}/")

    report_dir = f"{out_dir}/diagnostics"
    os.makedirs(report_dir, exist_ok=True)
    _save_diagnostics(
        report_dir, emotions, vectors_by_layer, denoised_by_layer,
        pca_diagnostics, variance_threshold, model, start_at_nth_token,
        target_layer=target_layer,
    )
    print(f"[denoise] saved diagnostics to {report_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
