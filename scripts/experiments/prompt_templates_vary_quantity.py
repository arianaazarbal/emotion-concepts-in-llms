"""
Scale emotional intensity by varying a numerical quantity in a prompt template.

For each template in ``data/prompt_templates_vary_quantity.json``, format the
prompt with each value of X, render it as a user message via the model's chat
template (with ``add_generation_prompt=True``), and measure emotion-probe
activation at three assistant-header positions:

    (a) before_role — the start-of-turn marker preceding the role name
    (b) role        — the role-name token itself (e.g. "model" for Gemma,
                       "assistant" for Llama/Qwen)
    (c) after_role  — the token immediately after the role name

Per-model role-token metadata lives in ``data/chat_template_probe_tokens.json``.

Outputs (TAG = ``{model}_{version}`` or ``{model}`` if version empty):
- results/prompt_templates_vary_quantity/{TAG}[/start_at_token_{N}][/denoised_layer_{L}]/{metric}/
    {template_name}.png           (per-template: 3 subplots, one per position)
    combined_before_role.png      (one subplot per template, before_role only)
    combined_role.png             (one subplot per template, role only)
    combined_after_role.png       (one subplot per template, after_role only)
    results.json

Re-run with ``--replot`` to regenerate only the three combined plots from the
cached ``results.json`` (tokenizer-only; no model load).
"""

import os

import fire
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.model_names import resolve_model_name
from src.utils.utils import dump_json, load_json


def find_role_token_idx(input_ids: torch.Tensor, tokenizer, role_substring: str) -> int:
    """Find the last token in ``input_ids`` whose stripped decoding equals
    ``role_substring`` (case-insensitive). Used to locate "assistant" / "model"
    in a chat-template-rendered prompt so we can probe the assistant header.
    """
    target = role_substring.strip().lower()
    for i in range(len(input_ids) - 1, -1, -1):
        tok = tokenizer.decode([int(input_ids[i])], skip_special_tokens=False).strip().lower()
        if tok == target:
            return i
    raise ValueError(
        f"Could not find role token '{role_substring}' in chat-template-rendered prompt. "
        f"Last 10 tokens: "
        + ", ".join(
            repr(tokenizer.decode([int(t)], skip_special_tokens=False))
            for t in input_ids[-10:]
        )
    )


def probe_at_positions(
    prompts,
    model,
    tokenizer,
    vectors_layer: torch.Tensor,
    layer: int,
    role_substring: str,
    cosine_sim: bool = False,
):
    """Run forward passes over ``prompts`` (each a user string), locate the
    role-name token in each, and compute emotion-probe scores at positions
    [role-1, role, role+1] for every emotion.

    Returns:
        scores: list of [3, num_emotions] tensors, one per prompt.
        position_tokens: dict of {position_label: [decoded_str, ...]} with one
            list per position (before_role / role / after_role). Tokens are
            normally identical across prompts for a given chat template.
    """
    device = next(model.parameters()).device
    vectors = vectors_layer.to(device=device, dtype=torch.float32)
    if cosine_sim:
        vectors = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-9)

    out = []
    position_tokens = {"before_role": [], "role": [], "after_role": []}
    for prompt in prompts:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        role_idx = find_role_token_idx(input_ids[0], tokenizer, role_substring)
        positions = [role_idx - 1, role_idx, role_idx + 1]
        if positions[0] < 0 or positions[-1] >= input_ids.shape[1]:
            raise ValueError(
                f"Probe positions {positions} out of bounds for seq len {input_ids.shape[1]}"
            )

        for label, pos in zip(("before_role", "role", "after_role"), positions):
            tok = tokenizer.decode([int(input_ids[0, pos])], skip_special_tokens=False)
            position_tokens[label].append(tok)

        with torch.no_grad():
            out_obj = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = out_obj.hidden_states[layer][0]  # [seq_len, hidden_size]
        picked = hidden[positions].float()  # [3, hidden_size]
        if cosine_sim:
            picked = picked / (picked.norm(dim=-1, keepdim=True) + 1e-9)
        scores = picked @ vectors.T  # [3, num_emotions]
        out.append(scores.cpu())
    return out, position_tokens


def discover_header_tokens(tokenizer, role_substring: str) -> dict:
    """Render a dummy chat prompt and return the three header tokens
    (before_role / role / after_role) as decoded strings. Used by --replot
    to label combined plots when the cached results.json predates per-position
    token storage.
    """
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"][0]
    role_idx = find_role_token_idx(input_ids, tokenizer, role_substring)
    return {
        "before_role": tokenizer.decode([int(input_ids[role_idx - 1])], skip_special_tokens=False),
        "role": tokenizer.decode([int(input_ids[role_idx])], skip_special_tokens=False),
        "after_role": tokenizer.decode([int(input_ids[role_idx + 1])], skip_special_tokens=False),
    }


def _plot_template(template_name, template_spec, records, emotions, position_labels, out_path, metric_label):
    """Plot one figure with ``len(position_labels)`` subplots for one template.

    X axis: the numeric values swept. One curve per emotion in ``emotions``.
    """
    fig, axes = plt.subplots(
        1, len(position_labels),
        figsize=(5.2 * len(position_labels), 4.2),
        sharey=True,
    )
    if len(position_labels) == 1:
        axes = [axes]

    xs = [r["x"] for r in records]
    cmap = plt.get_cmap("tab10")
    emotion_colors = {e: cmap(i % 10) for i, e in enumerate(emotions)}

    for pos_idx, pos_label in enumerate(position_labels):
        ax = axes[pos_idx]
        for e_idx, emotion in enumerate(emotions):
            ys = [r["scores"][pos_idx][e_idx] for r in records]
            ax.plot(xs, ys, marker="o", label=emotion, color=emotion_colors[emotion])
        ax.set_title(f"probe @ {pos_label}")
        ax.set_xlabel(template_spec.get("x_label", "X"))
        ax.axhline(0.0, color="#bbb", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(metric_label)
    axes[-1].legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(
        f"{template_name} — prompt: “{template_spec['prompt']}”",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_POSITION_BLURBS = {
    "before_role": 'token immediately before role name "{role}"',
    "role": 'role name token "{role}"',
    "after_role": 'token immediately after role name "{role}"',
}


def _plot_combined_at_position(
    per_template,
    emotions,
    position_labels,
    target_position: str,
    role_substring: str,
    out_path,
    metric_label,
    fallback_token: str = None,
):
    """Combined figure: one subplot per template showing only the probe
    activation at ``target_position`` (one of before_role / role / after_role).

    Token strings for each position are pulled from each template entry's
    ``{position}_tokens`` list when present; otherwise ``fallback_token``
    (discovered from the tokenizer) is used for the title.
    """
    pos_idx = position_labels.index(target_position)
    names = list(per_template.keys())
    n = len(names)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols, 3.6 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    cmap = plt.get_cmap("tab10")
    emotion_colors = {e: cmap(i % 10) for i, e in enumerate(emotions)}

    tokens_key = f"{target_position}_tokens"
    all_tokens = []
    for ax, t_name in zip(axes_flat, names):
        entry = per_template[t_name]
        records = entry["records"]
        xs = [r["x"] for r in records]
        for e_idx, emotion in enumerate(emotions):
            ys = [r["scores"][pos_idx][e_idx] for r in records]
            ax.plot(xs, ys, marker="o", label=emotion, color=emotion_colors[emotion])
        ax.set_title(f"“{entry['prompt_template']}”", fontsize=9)
        ax.set_xlabel(entry.get("x_label", "X"), fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.axhline(0.0, color="#bbb", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        all_tokens.extend(entry.get(tokens_key, []))

    for ax in axes_flat[n:]:
        ax.axis("off")

    axes_flat[0].legend(loc="best", frameon=True, fontsize=8)

    unique_tokens = list(dict.fromkeys(all_tokens))
    if not unique_tokens and fallback_token is not None:
        unique_tokens = [fallback_token]
    if len(unique_tokens) == 1:
        tok_display = repr(unique_tokens[0])
    elif unique_tokens:
        tok_display = " / ".join(repr(t) for t in unique_tokens)
    else:
        tok_display = "?"
    blurb = _POSITION_BLURBS[target_position].format(role=role_substring)
    fig.suptitle(
        f"Probe @ activation on token {tok_display} ({blurb})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def emit_combined_plots(
    per_template,
    emotions,
    position_labels,
    role_substring,
    output_dir,
    metric_label,
    fallback_tokens: dict = None,
):
    """Write the three ``combined_{position}.png`` figures into ``output_dir``."""
    fallback_tokens = fallback_tokens or {}
    paths = {}
    for pos in ("before_role", "role", "after_role"):
        out_path = f"{output_dir}/combined_{pos}.png"
        _plot_combined_at_position(
            per_template=per_template,
            emotions=emotions,
            position_labels=position_labels,
            target_position=pos,
            role_substring=role_substring,
            out_path=out_path,
            metric_label=metric_label,
            fallback_token=fallback_tokens.get(pos),
        )
        paths[pos] = out_path
    return paths


def main(
    model: str,
    layer: int = None,
    emotions: list = None,
    templates: list = None,
    start_at_nth_token: int = 50,
    denoised: bool = False,
    version: str = "v0",
    cosine_sim: bool = False,
    templates_path: str = "data/prompt_templates_vary_quantity.json",
    chat_tokens_path: str = "data/chat_template_probe_tokens.json",
    output_dir: str = None,
    replot: bool = False,
):
    """Probe emotion vectors across quantity-varying prompt templates.

    Args:
        model: Short model name registered in data/model_names.json.
        layer: Transformer layer to probe. Defaults to ~2/3 depth.
        emotions: Which emotions to include. Defaults to the config's
            ``default_emotions`` (happy, afraid, sad, calm).
        templates: Which templates to run (names). Defaults to all in the JSON.
        start_at_nth_token: Emotion-vector extraction variant to load
            (directory suffix only — never skips tokens at probe time).
        denoised: If True, load PCA-denoised vectors.
        version: Version suffix for model-folder lookup.
        cosine_sim: Use cosine similarity instead of raw dot product.
        templates_path: Path to the templates JSON.
        chat_tokens_path: Path to the per-model chat-template role-token JSON.
        output_dir: Override output dir.
        replot: If True, skip inference and regenerate the three combined plots
            from the cached ``results.json`` in ``output_dir``. Loads only the
            tokenizer (not the model) — no GPU needed.
    """
    templates_cfg = load_json(templates_path)
    chat_cfg = load_json(chat_tokens_path)
    all_templates = templates_cfg["templates"]
    default_emotions = templates_cfg.get("default_emotions", ["happy", "afraid", "sad", "calm"])
    position_labels = chat_cfg.get("position_labels", ["before_role", "role", "after_role"])

    if model not in chat_cfg["models"]:
        raise KeyError(
            f"Model '{model}' not found in {chat_tokens_path}. "
            f"Add an entry with 'role_token_substring'."
        )
    role_substring = chat_cfg["models"][model]["role_token_substring"]

    if emotions is None:
        emotions = default_emotions
    if isinstance(emotions, str):
        emotions = [emotions]

    template_names = list(templates) if templates else list(all_templates.keys())
    missing = [t for t in template_names if t not in all_templates]
    if missing:
        raise KeyError(f"Templates not in {templates_path}: {missing}")

    print(f"[prompt_templates] model={model} version={version or '(none)'} "
          f"denoised={denoised} cosine_sim={cosine_sim} emotions={emotions}")

    if replot:
        model_tag = f"{model}_{version}" if version else model
        if output_dir is None:
            suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
            if denoised and layer is not None:
                suffix += f"/denoised_layer_{layer}"
            suffix += "/cosine_sim" if cosine_sim else "/dot_product"
            output_dir = f"results/prompt_templates_vary_quantity/{model_tag}{suffix}"
        results_path = f"{output_dir}/results.json"
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"--replot: no cached {results_path}. Run without --replot first."
            )
        cached = load_json(results_path)
        metric_label = "cosine similarity" if cached.get("cosine_sim") else "dot product"
        tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(model))
        fallback_tokens = discover_header_tokens(tokenizer, role_substring)
        print(f"[prompt_templates] replot: regenerating combined plots in {output_dir}")
        paths = emit_combined_plots(
            per_template=cached["per_template"],
            emotions=cached["emotions"],
            position_labels=cached.get("position_labels", position_labels),
            role_substring=role_substring,
            output_dir=output_dir,
            metric_label=metric_label,
            fallback_tokens=fallback_tokens,
        )
        for pos, p in paths.items():
            print(f"[prompt_templates]   → saved {p}")
        return

    print(f"[prompt_templates] running {len(template_names)} templates: {template_names}")

    llm, tokenizer = load_model_and_tokenizer(model)
    if layer is None:
        layer = default_layer(llm)
        print(f"[prompt_templates] no layer specified, defaulting to layer {layer}")

    all_emotions_list, vectors_by_layer = load_emotion_vectors(
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
    unknown = [e for e in emotions if e not in all_emotions_list]
    if unknown:
        raise KeyError(
            f"Requested emotions not in vector set: {unknown}. "
            f"Available: {all_emotions_list[:15]}..."
        )
    e_indices = [all_emotions_list.index(e) for e in emotions]
    vectors_layer = vectors_by_layer[layer][e_indices]  # [len(emotions), hidden]

    model_tag = f"{model}_{version}" if version else model
    if output_dir is None:
        suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
        if denoised:
            suffix += f"/denoised_layer_{layer}"
        suffix += "/cosine_sim" if cosine_sim else "/dot_product"
        output_dir = f"results/prompt_templates_vary_quantity/{model_tag}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    metric_label = "cosine similarity" if cosine_sim else "dot product"
    results = {
        "model": model,
        "version": version,
        "layer": layer,
        "denoised": denoised,
        "cosine_sim": cosine_sim,
        "start_at_nth_token": start_at_nth_token,
        "emotions": emotions,
        "position_labels": position_labels,
        "per_template": {},
    }

    for t_name in template_names:
        spec = all_templates[t_name]
        values = spec["values"]
        prompts = [spec["prompt"].format(X=v) for v in values]
        print(f"[prompt_templates] {t_name}: {len(values)} prompts")

        scores, position_tokens = probe_at_positions(
            prompts=prompts,
            model=llm,
            tokenizer=tokenizer,
            vectors_layer=vectors_layer,
            layer=layer,
            role_substring=role_substring,
            cosine_sim=cosine_sim,
        )

        records = [
            {"x": v, "prompt": p, "scores": s.tolist()}
            for v, p, s in zip(values, prompts, scores)
        ]
        results["per_template"][t_name] = {
            "prompt_template": spec["prompt"],
            "x_label": spec.get("x_label", "X"),
            "unit": spec.get("unit", ""),
            "expected": spec.get("expected", ""),
            "before_role_tokens": position_tokens["before_role"],
            "role_tokens": position_tokens["role"],
            "after_role_tokens": position_tokens["after_role"],
            "records": records,
        }

        _plot_template(
            template_name=t_name,
            template_spec=spec,
            records=records,
            emotions=emotions,
            position_labels=position_labels,
            out_path=f"{output_dir}/{t_name}.png",
            metric_label=metric_label,
        )
        print(f"[prompt_templates]   → saved {output_dir}/{t_name}.png")

    paths = emit_combined_plots(
        per_template=results["per_template"],
        emotions=emotions,
        position_labels=position_labels,
        role_substring=role_substring,
        output_dir=output_dir,
        metric_label=metric_label,
    )
    for pos, p in paths.items():
        print(f"[prompt_templates]   → saved {p}")

    dump_json(results, f"{output_dir}/results.json")
    print(f"[prompt_templates] done. results.json and {len(template_names) + 3} plots in {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
