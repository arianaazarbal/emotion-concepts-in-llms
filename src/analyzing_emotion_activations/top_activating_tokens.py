"""
For a given model and a JSONL of samples, compute the top-activating *tokens*
for each emotion vector across the entire corpus, and display the surrounding
context around each top token.

This is the dual of top_activating_emotions.py: instead of asking
"which emotions activate most strongly at this token?", it asks
"which token positions in this corpus activate this emotion most strongly?"
and shows a windowed context around the activating token.

Emotion vectors are loaded from results/emotion_vectors/{model}/*_vectors.pt
(produced by src/extracting_emotion_vectors/extract_emotion_vectors.py).
"""

import bisect
import html
import os

import fire
import matplotlib.pyplot as plt
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

from src.analyzing_emotion_activations.top_activating_emotions import (
    is_chat_sample,
    sample_to_text,
)
from src.utils.emotion_probe import (
    default_layer,
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import dump_json, load_jsonl


def format_context(tokenizer, tokens, hit_idx: int, window: int) -> str:
    """Render a windowed context around tokens[hit_idx], highlighting the hit.

    Uses convert_tokens_to_string so the output looks natural across tokenizer
    families (BPE, sentencepiece, etc.).
    """
    start = max(0, hit_idx - window)
    end = min(len(tokens), hit_idx + window + 1)
    prefix = tokenizer.convert_tokens_to_string(tokens[start:hit_idx])
    target = tokenizer.convert_tokens_to_string([tokens[hit_idx]])
    suffix = tokenizer.convert_tokens_to_string(tokens[hit_idx + 1 : end])
    return f"{prefix}**[{target}]**{suffix}"


def global_to_local(sample_offsets, global_idx: int):
    """Map a flat token index back to (sample_idx, local_token_idx)."""
    s_idx = bisect.bisect_right(sample_offsets, global_idx) - 1
    return s_idx, global_idx - sample_offsets[s_idx]


def plot_top_token_activation(
    tokens, scores, emotion, emotion_idx, hit_token_idx, out_path, model, layer, tokenizer,
    context_window=10,
    cosine_sim=False,
):
    """Render a paragraph of context tokens, shaded by emotion activation.

    Each token in a ±context_window window around the top-activating token is
    drawn as colored text; color encodes the per-token probe score. The hit
    token itself gets a bold black outline.
    """
    start = max(0, hit_token_idx - context_window)
    end = min(len(tokens), hit_token_idx + context_window + 1)
    window_tokens = tokens[start:end]
    window_vals = scores[start:end, emotion_idx].numpy()
    hit_in_window = hit_token_idx - start

    display_strs = []
    for tok in window_tokens:
        s = tokenizer.convert_tokens_to_string([tok])
        display_strs.append(s if s else tok)

    vmax = float(max(abs(window_vals.min()), abs(window_vals.max()), 1e-9))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    metric_label = "cosine similarity" if cosine_sim else "dot product"

    fig = plt.figure(figsize=(12, 2.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[3, 1],
        left=0.02, right=0.98, top=0.82, bottom=0.15, hspace=0.5,
    )
    ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    top_y = 0.85
    line_y = top_y
    line_height = 0.45
    x = 0.02
    max_x = 0.98
    gap = 0.006
    fontsize = 12

    for i, (token_str, val) in enumerate(zip(display_strs, window_vals)):
        display = token_str.strip() or " "
        is_hit = i == hit_in_window
        facecolor = cmap(norm(val))
        r, g, b = facecolor[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if luminance > 0.55 else "white"
        bbox = dict(
            boxstyle="round,pad=0.25",
            facecolor=facecolor,
            edgecolor="black" if is_hit else "none",
            linewidth=2.0 if is_hit else 0,
        )

        def _place(x_, y_):
            return ax.text(
                x_, y_, display, ha="left", va="center",
                transform=ax.transAxes, bbox=bbox,
                fontsize=fontsize,
                fontweight="bold" if is_hit else "normal",
                color=text_color,
            )

        t = _place(x, line_y)
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        width = bb.width

        if x + width > max_x and x > 0.02:
            t.remove()
            line_y -= line_height
            x = 0.02
            t = _place(x, line_y)
            bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            width = bb.width

        x += width + gap

    lines_used = max(1, 1 + int(round((top_y - line_y) / line_height)))
    if lines_used > 1:
        fig.set_size_inches(12, 2.0 + 0.45 * (lines_used - 1), forward=True)

    ax.set_title(
        f'{model} | layer {layer} | "{emotion}" (±{context_window} tokens)',
        fontsize=12, pad=10,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(metric_label, fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_HTML_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 1.5em auto; padding: 0 1em; color: #222; }
h1 { margin-bottom: 0.2em; }
h2 { margin-top: 1.6em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 0.8em; }
.ex { margin: 0.6em 0 1.2em; }
.ex-meta { color: #555; font-size: 0.85em; margin-bottom: 0.25em; font-family: ui-monospace, Menlo, monospace; }
.chips { line-height: 2.2; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 14px; }
.chip { display: inline-block; padding: 0.05em 0.3em; margin: 0 1px; border-radius: 3px; white-space: pre; }
.chip.hit { outline: 2px solid #000; font-weight: 700; }
.legend { display: inline-flex; align-items: center; gap: 0.4em; margin-top: 0.5em; font-size: 0.85em; color: #666; }
.legend-bar { width: 160px; height: 12px; border: 1px solid #ccc; }
"""


def _gradient_css():
    cmap = plt.get_cmap("RdBu_r")
    stops = [f"{to_hex(cmap(v / 10))} {v * 10}%" for v in range(11)]
    return f"background: linear-gradient(to right, {', '.join(stops)});"


def write_html_chip_view(
    results,
    per_text_tokens,
    per_text_scores,
    emotions,
    tokenizer,
    out_path,
    model,
    layer,
    context_window,
    cosine_sim,
    num_samples,
    total_tokens,
    title_suffix="",
):
    """Write a self-contained HTML page showing, for each emotion, each top-K
    hit as a row of colored chips (one per token in a ±context_window window).
    Color encodes the per-token probe score; each row is normalized by the max
    absolute score within that window. The hit token is outlined and bold.
    """
    cmap = plt.get_cmap("RdBu_r")
    metric_label = "cosine similarity" if cosine_sim else "dot product"
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Top activating tokens — {html.escape(model)} layer {layer}</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        f"<h1>Top activating tokens — {html.escape(model)} (layer {layer}){html.escape(title_suffix)}</h1>",
        f"<div class='meta'>Corpus: {num_samples} samples, {total_tokens} tokens. "
        f"Metric: {metric_label}. Context window: ±{context_window}.</div>",
        f"<div class='legend'>− <span class='legend-bar' style='{_gradient_css()}'></span> +</div>",
    ]

    for emotion, examples in results.items():
        parts.append(f"<h2>{html.escape(emotion)}</h2>")
        e_idx = emotion_to_idx[emotion]
        for ex in examples:
            s_idx = ex["sample_idx"]
            hit_idx = ex["token_idx"]
            tokens = per_text_tokens[s_idx]
            scores = per_text_scores[s_idx][:, e_idx]
            start = max(0, hit_idx - context_window)
            end = min(len(tokens), hit_idx + context_window + 1)
            window_vals = scores[start:end].tolist()
            window_tokens = tokens[start:end]
            hit_in_window = hit_idx - start
            vmax = max(abs(min(window_vals)), abs(max(window_vals)), 1e-9)
            norm = Normalize(vmin=-vmax, vmax=vmax)

            chip_html = []
            for i, (tok, val) in enumerate(zip(window_tokens, window_vals)):
                s = tokenizer.convert_tokens_to_string([tok]) or tok
                display = s.replace("\n", "⏎")
                rgb = cmap(norm(val))
                r, g, b = rgb[:3]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                fg = "#000" if luminance > 0.55 else "#fff"
                bg = to_hex(rgb)
                cls = "chip hit" if i == hit_in_window else "chip"
                chip_html.append(
                    f"<span class='{cls}' style='background:{bg};color:{fg};' "
                    f"title='score={val:.3f}'>{html.escape(display)}</span>"
                )
            parts.append(
                f"<div class='ex'>"
                f"<div class='ex-meta'>#{ex['rank']}  score={ex['score']:.3f}  "
                f"sample={s_idx}  pos={hit_idx}</div>"
                f"<div class='chips'>{''.join(chip_html)}</div>"
                f"</div>"
            )

    parts.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def main(
    model: str,
    samples_jsonl: str,
    layer: int = None,
    top_k: int = 10,
    context_window: int = 10,
    max_samples: int = None,
    output_dir: str = None,
    seed: int = 0,
    start_at_nth_token: int = 0,
    cosine_sim: bool = False,
    denoised: bool = False,
    batch_size: int = 8,
    max_length: int = 2048,
    version: str = "",
):
    """Find top-activating tokens per emotion vector across a corpus.

    Args:
        model: Short model name registered in data/model_names.json.
        samples_jsonl: Path to a JSONL file of samples (string, {"text": ...},
            list of {"role","content"}, or {"messages": [...]}).
        layer: Transformer layer whose residual stream is projected onto
            emotion vectors. Defaults to ~2/3 through the model.
        top_k: Number of top-activating tokens to keep per emotion.
        context_window: Number of tokens before/after the hit to display.
        max_samples: If set, only process the first N samples (for quick testing).
        output_dir: Where to save outputs. Defaults to results/top_tokens/{model}{_version}/.
        seed: RNG seed for reproducibility.
        start_at_nth_token: Which emotion-vector extraction variant to load (default 0 = original layout).
        cosine_sim: Use cosine similarity instead of raw dot product.
        denoised: Use PCA-denoised emotion vectors.
        batch_size: Forward-pass batch size.
        max_length: Tokenizer truncation cap (in tokens).
        version: Optional version suffix; when non-empty, resolves model folders
            as ``{model}_{version}`` for both vector lookup and default output_dir.
    """
    torch.manual_seed(seed)

    model_tag = f"{model}_{version}" if version else model
    if output_dir is None:
        output_dir = f"results/top_tokens/{model_tag}"
    os.makedirs(output_dir, exist_ok=True)

    llm, tokenizer = load_model_and_tokenizer(model)
    if layer is None:
        layer = default_layer(llm)
        print(f"[top_tokens] no layer specified, defaulting to layer {layer} (~2/3 through model)")

    emotions, vectors_by_layer = load_emotion_vectors(
        model,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        version=version,
    )
    if layer not in vectors_by_layer:
        raise ValueError(
            f"Layer {layer} not found in emotion vectors. "
            f"Available: {sorted(vectors_by_layer.keys())}"
        )

    samples = load_jsonl(samples_jsonl)
    if max_samples is not None:
        samples = samples[:max_samples]

    texts = [sample_to_text(s, tokenizer) for s in samples]

    print(f"[top_tokens] probing {len(texts)} samples at layer {layer} (batch_size={batch_size})")
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
    per_text_scores = probe.scores[layer]  # list of [seq_len_i, num_emotions]
    per_text_tokens = probe.tokens

    sample_offsets = []
    offset = 0
    for scores in per_text_scores:
        sample_offsets.append(offset)
        offset += scores.shape[0]

    all_scores = torch.cat(per_text_scores, dim=0)  # [total_tokens, num_emotions]
    effective_k = min(top_k, all_scores.shape[0])
    topk = torch.topk(all_scores, k=effective_k, dim=0)

    results = {}
    for e_idx, emotion in enumerate(probe.emotions):
        examples = []
        for rank in range(effective_k):
            global_idx = int(topk.indices[rank, e_idx])
            score = float(topk.values[rank, e_idx])
            s_idx, t_idx = global_to_local(sample_offsets, global_idx)
            tokens = per_text_tokens[s_idx]
            examples.append(
                {
                    "rank": rank + 1,
                    "score": score,
                    "sample_idx": s_idx,
                    "token_idx": t_idx,
                    "token": tokens[t_idx],
                    "context": format_context(tokenizer, tokens, t_idx, context_window),
                }
            )
        results[emotion] = examples

    plots_dir = f"{output_dir}/top_token_plots"
    os.makedirs(plots_dir, exist_ok=True)
    for e_idx, emotion in enumerate(probe.emotions):
        top_hit = results[emotion][0]
        s_idx = top_hit["sample_idx"]
        t_idx = top_hit["token_idx"]
        plot_top_token_activation(
            tokens=per_text_tokens[s_idx],
            scores=per_text_scores[s_idx],
            emotion=emotion,
            emotion_idx=e_idx,
            hit_token_idx=t_idx,
            out_path=f"{plots_dir}/{emotion}_layer{layer}.png",
            model=model,
            layer=layer,
            tokenizer=tokenizer,
            context_window=context_window,
            cosine_sim=cosine_sim,
        )
    print(f"[top_tokens] saved {len(probe.emotions)} top-token activation plots to {plots_dir}")

    payload = {
        "model": model,
        "layer": layer,
        "top_k": effective_k,
        "context_window": context_window,
        "num_samples": len(samples),
        "total_tokens": all_scores.shape[0],
        "per_emotion": results,
    }
    json_path = f"{output_dir}/top_tokens_layer{layer}.json"
    dump_json(payload, json_path)

    md_path = f"{output_dir}/top_tokens_layer{layer}.md"
    with open(md_path, "w") as f:
        f.write(f"# Top activating tokens — {model} (layer {layer})\n\n")
        f.write(
            f"Corpus: {len(samples)} samples, {all_scores.shape[0]} tokens. "
            f"Top {effective_k} per emotion, context window ±{context_window}.\n\n"
        )
        for emotion, examples in results.items():
            f.write(f"## {emotion}\n\n")
            for ex in examples:
                f.write(
                    f"{ex['rank']}. (score={ex['score']:.3f}, "
                    f"sample={ex['sample_idx']}, pos={ex['token_idx']}) "
                    f"{ex['context']}\n\n"
                )

    html_path = f"{output_dir}/top_tokens_layer{layer}.html"
    write_html_chip_view(
        results=results,
        per_text_tokens=per_text_tokens,
        per_text_scores=per_text_scores,
        emotions=probe.emotions,
        tokenizer=tokenizer,
        out_path=html_path,
        model=model,
        layer=layer,
        context_window=context_window,
        cosine_sim=cosine_sim,
        num_samples=len(samples),
        total_tokens=all_scores.shape[0],
    )

    print(f"Saved top-token results to {json_path}, {md_path}, {html_path}")


if __name__ == "__main__":
    fire.Fire(main)
