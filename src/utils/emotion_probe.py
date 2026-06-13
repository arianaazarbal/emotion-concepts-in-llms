"""
General-purpose utility for computing "emotion probe values": the dot product
between a model's residual-stream activations and a set of emotion vectors.

The probe value at (text, token, emotion, layer) is

    dot(hidden_state[layer, token], emotion_vector[emotion, layer])

By default we return per-token scores (one value per token, per emotion, per
layer) so callers can do their own aggregation. Mean / max / first / last
reductions are also available via the `aggregation` arg.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.model_names import resolve_model_name
from src.utils.vector_cache import resolve_vector_subdir


def load_emotion_vectors(
    model_short_name: str,
    start_at_nth_token: int = 0,
    denoised: bool = False,
    version: str = "",
):
    """Load all per-emotion vectors for a model.

    Args:
        model_short_name: Short model name registered in data/model_names.json.
        start_at_nth_token: Which extraction variant to load. 0 (default) loads
            from the original layout (results/emotion_vectors/{model}/), other
            values load from results/emotion_vectors/{model}/start_at_{n}/.
        denoised: If True, load from the denoised/ subdirectory.
        version: Optional version suffix; when non-empty, resolves to
            ``results/emotion_vectors/{model}_{version}/...``.

    Returns:
        emotions: list of emotion names (sorted, fixed order).
        vectors_by_layer: dict[int, Tensor] of shape [num_emotions, hidden_size].
    """
    model_tag = f"{model_short_name}_{version}" if version else model_short_name
    rel = (
        f"{model_tag}/start_at_{start_at_nth_token}"
        if start_at_nth_token != 0
        else f"{model_tag}"
    )
    if denoised:
        rel = f"{rel}/denoised"
    vec_dir = resolve_vector_subdir(rel)
    if not os.path.isdir(vec_dir):
        raise FileNotFoundError(
            f"No emotion vectors found at {vec_dir}. Run extract_emotion_vectors first."
        )
    files = sorted(f for f in os.listdir(vec_dir) if f.endswith("_vectors.pt"))
    if not files:
        raise FileNotFoundError(f"No *_vectors.pt files in {vec_dir}.")

    emotions = []
    per_emotion = {}
    for f in files:
        emotion = f[: -len("_vectors.pt")]
        emotions.append(emotion)
        per_emotion[emotion] = torch.load(f"{vec_dir}/{f}", map_location="cpu")

    layers = sorted(next(iter(per_emotion.values())).keys())
    vectors_by_layer = {}
    for layer in layers:
        stacked = torch.stack([per_emotion[e][layer] for e in emotions], dim=0)
        vectors_by_layer[layer] = stacked  # [num_emotions, hidden_size]
    return emotions, vectors_by_layer


@dataclass
class EmotionProbeResult:
    """Result of an emotion-probe computation.

    Fields:
        emotions: ordered emotion names (length E).
        layers: ordered layer indices that were probed.
        aggregation: per-token reduction used ("none" | "mean" | "max" | "last" | "first").
        scores: dict mapping layer -> result tensor.
            - When aggregation == "none": list of tensors of shape [valid_seq_len_i, E],
              one per input text (variable length).
            - Otherwise: tensor of shape [num_texts, E].
        tokens: per-text list of token strings (one inner list per input text).
            Only populated when aggregation == "none".
    """

    emotions: List[str]
    layers: List[int]
    aggregation: str
    scores: dict
    tokens: Optional[List[List[str]]] = None


def default_layer(model: AutoModelForCausalLM) -> int:
    """Return a sensible default probe/steer layer for the given model.

    Generic default is ~2/3 through the transformer stack. gemma2-9b is
    pinned to 26 specifically (calibrated for this project's emotion
    vectors — see memory).
    """
    n = model.config.num_hidden_layers
    if getattr(model.config, "model_type", None) == "gemma2" and n == 42:
        return 26
    return max(1, (2 * n) // 3)


def load_model_and_tokenizer(
    model_short_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Convenience loader: resolves the short name, picks GPU/bf16 when available, sets pad token."""
    full_name = resolve_model_name(model_short_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        full_name, torch_dtype=dtype, device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def _reduce(valid: torch.Tensor, how: str, num_emotions: int) -> torch.Tensor:
    """Reduce a [valid_len, E] tensor along the token dimension."""
    if valid.shape[0] == 0:
        return torch.zeros(num_emotions)
    if how == "mean":
        return valid.mean(dim=0)
    if how == "max":
        return valid.max(dim=0).values
    if how == "first":
        return valid[0]
    if how == "last":
        return valid[-1]
    raise ValueError(f"Unknown aggregation: {how}")


@torch.no_grad()
def emotion_probe(
    texts: Union[str, List[str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    emotion_vectors_by_layer: dict,
    emotions: List[str],
    layers: Optional[Union[int, List[int]]] = None,
    selected_emotions: Optional[List[str]] = None,
    aggregation: str = "none",
    batch_size: int = 8,
    max_length: int = 2048,
    start_at_nth_token: int = 0,
    cosine_sim: bool = False,
) -> EmotionProbeResult:
    """Compute emotion probe values for a batch of texts.

    Args:
        texts: A single string or a list of strings.
        model: HF causal LM (must support output_hidden_states=True).
        tokenizer: matched tokenizer; pad token + right-padding will be enforced.
        emotion_vectors_by_layer: dict {layer_idx: tensor [num_emotions, hidden]}.
            Use load_emotion_vectors() from top_activating_emotions to construct.
        emotions: ordered list of emotion names matching the rows of each tensor in
            emotion_vectors_by_layer.
        layers: layer index, list of indices, or None. None defaults to a single
            layer ~2/3 through the model.
        selected_emotions: subset of emotion names to score. None = all.
        aggregation: how to reduce per-token scores per text. Default "none"
            returns per-token tensors. Other options: "mean", "max", "first", "last".
        batch_size: forward-pass batch size. Right-padded.
        max_length: tokenizer truncation cap (in tokens). Guards against OOM
            when callers pass long inputs to a model with a huge max_position_embeddings.
        start_at_nth_token: drop the first N tokens of every text from the
            output (useful for skipping BOS-like positions).
        cosine_sim: Use cosine similarity instead of raw dot product.

    Returns:
        EmotionProbeResult. See its docstring for the shape of `scores`.
    """
    if isinstance(texts, str):
        texts = [texts]

    if layers is None:
        layers = [default_layer(model)]
    elif isinstance(layers, int):
        layers = [layers]
    layers = list(layers)

    available = set(emotion_vectors_by_layer.keys())
    missing = [l for l in layers if l not in available]
    if missing:
        raise ValueError(
            f"Requested layers {missing} not present in emotion_vectors_by_layer "
            f"(available: {sorted(available)})"
        )

    if selected_emotions is not None:
        try:
            idx = [emotions.index(e) for e in selected_emotions]
        except ValueError as e:
            raise ValueError(f"Unknown emotion in selected_emotions: {e}") from None
        emotion_idx = torch.tensor(idx, dtype=torch.long)
        emotions_out = list(selected_emotions)
    else:
        emotion_idx = None
        emotions_out = list(emotions)
    num_emotions = len(emotions_out)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    vec_by_layer = {}
    for l in layers:
        v = emotion_vectors_by_layer[l].to(device=device, dtype=model_dtype)
        if emotion_idx is not None:
            v = v.index_select(0, emotion_idx.to(device))
        if cosine_sim:
            v = torch.nn.functional.normalize(v, dim=-1)
        vec_by_layer[l] = v  # [E, H]

    if aggregation == "none":
        per_text_scores = {l: [] for l in layers}
        per_text_tokens: List[List[str]] = []
    else:
        agg_rows = {l: [] for l in layers}

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch = texts[b_idx * batch_size : (b_idx + 1) * batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        attn = enc["attention_mask"]
        outputs = model(**enc, output_hidden_states=True)

        for l in layers:
            h = outputs.hidden_states[l]  # [B, S, H]
            if cosine_sim:
                h = torch.nn.functional.normalize(h, dim=-1)
            v = vec_by_layer[l]  # [E, H]
            scores = torch.matmul(h, v.transpose(0, 1))  # [B, S, E]

            for i in range(h.shape[0]):
                true_len = int(attn[i].sum().item())
                start = min(start_at_nth_token, true_len)
                valid = scores[i, start:true_len]  # [valid_len, E]
                if aggregation == "none":
                    per_text_scores[l].append(valid.float().cpu())
                else:
                    agg_rows[l].append(
                        _reduce(valid.float().cpu(), aggregation, num_emotions)
                    )

        if aggregation == "none":
            for i in range(len(batch)):
                true_len = int(attn[i].sum().item())
                start = min(start_at_nth_token, true_len)
                token_ids = enc["input_ids"][i, start:true_len].tolist()
                per_text_tokens.append(tokenizer.convert_ids_to_tokens(token_ids))

    if aggregation == "none":
        return EmotionProbeResult(
            emotions=emotions_out,
            layers=layers,
            aggregation=aggregation,
            scores=per_text_scores,
            tokens=per_text_tokens,
        )

    return EmotionProbeResult(
        emotions=emotions_out,
        layers=layers,
        aggregation=aggregation,
        scores={l: torch.stack(agg_rows[l], dim=0) for l in layers},
    )


def quick_probe(
    text: str,
    model_short_name: str,
    emotion: Optional[str] = None,
    layer: Optional[int] = None,
    aggregation: str = "mean",
    start_at_nth_token: int = 0,
    version: str = "",
):
    """One-call interactive helper: load model + emotion vectors and probe a single text.

    Loads the model and the saved emotion vectors for `model_short_name`, then
    computes the probe value at a single layer (default ~2/3 through the model)
    with mean aggregation. Returns:

    - a float, if `emotion` is given (probe value for that emotion)
    - a dict {emotion: float}, otherwise (probe values for all emotions)

    For repeated calls, prefer constructing the model + vectors once and
    calling `emotion_probe` directly to avoid reloading.
    """
    model, tokenizer = load_model_and_tokenizer(model_short_name)
    emotions, vec_by_layer = load_emotion_vectors(
        model_short_name,
        start_at_nth_token=start_at_nth_token,
        version=version,
    )
    if layer is None:
        layer = default_layer(model)
    result = emotion_probe(
        texts=text,
        model=model,
        tokenizer=tokenizer,
        emotion_vectors_by_layer=vec_by_layer,
        emotions=emotions,
        layers=[layer],
        selected_emotions=[emotion] if emotion is not None else None,
        aggregation=aggregation,
        start_at_nth_token=start_at_nth_token,
    )
    row = result.scores[layer][0]  # [E]
    if emotion is not None:
        return float(row.item())
    return dict(zip(result.emotions, row.tolist()))
