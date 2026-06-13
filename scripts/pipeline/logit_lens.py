"""
Logit lens over emotion vectors.

For each emotion vector at a given layer, apply the model's final residual
norm and project through the unembedding (``lm_head``). Report the top-K
tokens pushed up and down by that vector. Reproduces the "top upweighted /
downweighted tokens" table from the paper (Table 1).

Outputs (TAG = ``{model}_{version}`` or ``{model}`` if version empty):
- results/logit_lens/{TAG}[/start_at_token_{N}][/denoised]/layer_{L}.json
- results/logit_lens/{TAG}[/start_at_token_{N}][/denoised]/layer_{L}.md

Denoised and raw variants live in separate directories, so you can diff.
"""

import os

import fire
import torch

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import dump_json


def _get_final_norm(model):
    """Return the final residual-stream norm module (RMSNorm / LayerNorm).

    Most HuggingFace causal LMs (Llama, Gemma, Qwen, Mistral) expose this as
    ``model.model.norm``. Falls back to a noop identity if absent.
    """
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    if norm is None:
        print("[logit_lens] WARN: model.model.norm not found; using identity")
        return torch.nn.Identity()
    return norm


def compute_logit_lens(
    vectors: torch.Tensor,
    model,
    top_k: int = 5,
    normalize: bool = True,
):
    """Project a stack of emotion vectors through (final norm → lm_head).

    Args:
        vectors: [num_emotions, hidden_size] tensor on any device/dtype.
        model: HuggingFace causal LM with ``model.model.norm`` and ``lm_head``.
        top_k: Number of top-up and top-down tokens per vector.
        normalize: If True (default), apply the model's final residual-stream
            norm before the unembed — this matches what actually happens in a
            forward pass and is the standard logit-lens recipe. If False, skip
            the norm and return the raw ``vector @ lm_head.T`` projection.

    Returns:
        up_values, up_indices, down_values, down_indices: each
        [num_emotions, top_k] CPU tensors. Indices are into the tokenizer vocab.
    """
    lm_head = model.get_output_embeddings()
    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    with torch.no_grad():
        v = vectors.to(device=device, dtype=dtype)
        if normalize:
            v = _get_final_norm(model)(v)
        logits = lm_head(v)  # [num_emotions, vocab_size]

    logits = logits.float().cpu()
    up = torch.topk(logits, k=top_k, dim=-1)
    down = torch.topk(-logits, k=top_k, dim=-1)
    return up.values, up.indices, -down.values, down.indices


def _decode(tokenizer, token_id: int) -> str:
    """Decode a single token id to a human-readable string (no special tokens)."""
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def main(
    model: str,
    layer: int = None,
    top_k: int = 5,
    start_at_nth_token: int = 50,
    denoised: bool = False,
    version: str = "v0",
    normalize: bool = True,
    output_dir: str = None,
):
    """Run logit lens on all emotion vectors for one (model, version) at one layer.

    Args:
        model: Short model name registered in data/model_names.json.
        layer: Transformer layer whose vectors to read. Defaults to ~2/3 depth.
        top_k: How many top-up / top-down tokens per emotion.
        start_at_nth_token: Emotion-vector extraction variant to load.
        denoised: If True, load PCA-denoised vectors.
        version: Version suffix for model-folder lookup.
        normalize: If True (default), apply the model's final norm before the
            unembed. Pass ``--normalize False`` to get the raw ``vec @ W_U^T``
            projection; results land in a ``/no_norm/`` subdir so you can
            diff against the normalized version.
        output_dir: Override output dir. Defaults to the conventional
            results/logit_lens/{model_tag}[/start_at_token_{N}][/denoised][/no_norm]/.
    """
    print(f"[logit_lens] model={model} version={version or '(none)'} "
          f"denoised={denoised} start_at_nth_token={start_at_nth_token} normalize={normalize}")

    llm, tokenizer = load_model_and_tokenizer(model)
    if layer is None:
        layer = default_layer(llm)
        print(f"[logit_lens] no layer specified, defaulting to layer {layer} (~2/3 through model)")

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

    vectors = vectors_by_layer[layer]  # [num_emotions, hidden_size]
    print(f"[logit_lens] projecting {vectors.shape[0]} emotion vectors at layer {layer} "
          f"({'with' if normalize else 'without'} final norm)")
    up_vals, up_idx, down_vals, down_idx = compute_logit_lens(
        vectors, llm, top_k=top_k, normalize=normalize,
    )

    per_emotion = {}
    for i, emotion in enumerate(emotions):
        up = [
            {"token": _decode(tokenizer, up_idx[i, k]), "token_id": int(up_idx[i, k]),
             "logit": float(up_vals[i, k])}
            for k in range(top_k)
        ]
        down = [
            {"token": _decode(tokenizer, down_idx[i, k]), "token_id": int(down_idx[i, k]),
             "logit": float(down_vals[i, k])}
            for k in range(top_k)
        ]
        per_emotion[emotion] = {"up": up, "down": down}

    model_tag = f"{model}_{version}" if version else model
    if output_dir is None:
        suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
        if denoised:
            suffix += "/denoised"
        if not normalize:
            suffix += "/no_norm"
        output_dir = f"results/logit_lens/{model_tag}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model": model,
        "version": version,
        "layer": layer,
        "top_k": top_k,
        "denoised": denoised,
        "normalize": normalize,
        "start_at_nth_token": start_at_nth_token,
        "per_emotion": per_emotion,
    }
    json_path = f"{output_dir}/layer_{layer}.json"
    dump_json(payload, json_path)

    md_path = f"{output_dir}/layer_{layer}.md"
    tags = []
    if denoised:
        tags.append("denoised")
    tags.append("with final norm" if normalize else "raw, no final norm")
    tag_suffix = f" ({', '.join(tags)})"
    projection_desc = (
        "(final norm → lm_head)" if normalize else "(lm_head only, no final norm)"
    )
    with open(md_path, "w") as f:
        f.write(f"# Logit lens — {model} layer {layer}{tag_suffix}\n\n")
        f.write(f"Projecting emotion vectors through {projection_desc}. "
                f"Top {top_k} up / down tokens per emotion.\n\n")
        f.write("| emotion | ↑ up | ↓ down |\n")
        f.write("|---|---|---|\n")
        for emotion, rec in per_emotion.items():
            up_str = ", ".join(
                f"`{e['token']}` ({e['logit']:+.2f})" for e in rec["up"]
            )
            down_str = ", ".join(
                f"`{e['token']}` ({e['logit']:+.2f})" for e in rec["down"]
            )
            f.write(f"| {emotion} | {up_str} | {down_str} |\n")

    print(f"[logit_lens] saved {json_path} and {md_path}")


if __name__ == "__main__":
    fire.Fire(main)
