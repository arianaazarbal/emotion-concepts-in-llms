"""Local batched generation with emotion-vector activation steering.

Hooks the output of one or more transformer blocks and adds a scaled emotion
vector to the residual stream on every forward pass under `model.generate()`.

Parameter semantics (layers, emotions, selected_emotion,
emotion_vectors_by_layer, batch_size, max_length) mirror those in
`src.utils.emotion_probe` so callers can reuse `load_emotion_vectors` and
`load_model_and_tokenizer` from there. The exception is
``normalize_steering_vector``: at probe time ``cosine_sim`` switches the
similarity metric, but at steering time the equivalent knob just divides out
the stored vector's norm so that ``coeff`` is a direct residual-stream
magnitude — hence the different name.

Layer convention follows emotion_probe / extract_emotion_vectors: layer index
`l` refers to `outputs.hidden_states[l]`, i.e. the output of transformer
block `l-1`. Steering at `l=0` (raw embeddings) is not supported.
"""

import contextlib
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_block(model: AutoModelForCausalLM, layer_idx: int):
    """Return the transformer block whose output corresponds to hidden_states[layer_idx]."""
    if layer_idx < 1:
        raise ValueError(
            f"Cannot steer at layer {layer_idx}; minimum is 1 (output of first block). "
            "Layer 0 corresponds to raw embeddings and is not supported."
        )
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "transformer.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            if layer_idx - 1 >= len(obj):
                raise ValueError(
                    f"Layer {layer_idx} out of range; model has {len(obj)} blocks."
                )
            return obj[layer_idx - 1]
    raise ValueError(
        f"Could not locate transformer blocks on {type(model).__name__}; "
        "tried model.layers, transformer.h, gpt_neox.layers, transformer.layers."
    )


def _build_steering_by_layer(
    emotion_vectors_by_layer: Dict[int, torch.Tensor],
    emotions: List[str],
    selected_emotion: str,
    layers: List[int],
    coeff: float,
    normalize_steering_vector: bool,
) -> Dict[int, torch.Tensor]:
    """Slice the requested emotion row per layer and return (coeff * [unit?] vector)."""
    if selected_emotion not in emotions:
        raise ValueError(f"Unknown emotion {selected_emotion!r}; known: {emotions}")
    e_idx = emotions.index(selected_emotion)
    out = {}
    for l in layers:
        if l not in emotion_vectors_by_layer:
            raise ValueError(
                f"Layer {l} not in emotion_vectors_by_layer "
                f"(available: {sorted(emotion_vectors_by_layer)})"
            )
        v = emotion_vectors_by_layer[l][e_idx]
        if normalize_steering_vector:
            v = torch.nn.functional.normalize(v, dim=-1)
        out[l] = coeff * v
    return out


@contextlib.contextmanager
def _steering_hooks(
    model: AutoModelForCausalLM,
    steering_by_layer: Dict[int, torch.Tensor],
    steer_positions: str,
    state: dict,
    debug: bool = False,
):
    """Register forward hooks on the selected blocks that add `vec` to their output.

    ``steer_positions`` selects which tokens receive the steering vector:

      * ``"all"``             – every token (prompt + generation).
      * ``"generation_only"`` – only newly generated tokens; the prompt-encoding
                                forward pass is skipped.
      * ``"user_only"``       – only tokens covered by ``state["user_mask"]``
                                (a ``[B, T]`` bool tensor marking the user's
                                content span inside the chat-templated prompt).
                                Generation steps are skipped entirely.

    When ``debug`` is True, each invocation prints a one-line summary of what
    got steered (phase, shape, #positions touched).

    Registered with `prepend=True` so the steering fires before any
    output-capturing hooks transformers installs for `output_hidden_states=True`.
    """
    handles = []

    def make_hook(vec: torch.Tensor, layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            B, T = h.shape[0], h.shape[1]
            is_gen = state["is_generation"]
            vec_t = vec.to(device=h.device, dtype=h.dtype)

            if is_gen:
                if steer_positions == "user_only":
                    if debug:
                        print(f"[steer-debug] layer={layer_idx} phase=gen shape=({B},{T}) steered=0 (user_only skips gen)")
                    return output
                h = h + vec_t
                if debug:
                    print(f"[steer-debug] layer={layer_idx} phase=gen shape=({B},{T}) steered={B*T}")
            else:
                if steer_positions == "generation_only":
                    if debug:
                        print(f"[steer-debug] layer={layer_idx} phase=prompt shape=({B},{T}) steered=0 (generation_only skips prompt)")
                    return output
                if steer_positions == "user_only":
                    mask = state.get("user_mask")
                    if mask is None:
                        raise RuntimeError("user_only requested but state['user_mask'] is unset")
                    if mask.shape[0] != B:
                        nrs = B // mask.shape[0]
                        mask = mask.repeat_interleave(nrs, dim=0)
                    mask = mask.to(device=h.device)
                    h = h + mask[..., None].to(h.dtype) * vec_t
                    if debug:
                        n_steered = int(mask.sum().item())
                        print(f"[steer-debug] layer={layer_idx} phase=prompt shape=({B},{T}) steered={n_steered} (user_only mask)")
                else:
                    h = h + vec_t
                    if debug:
                        print(f"[steer-debug] layer={layer_idx} phase=prompt shape=({B},{T}) steered={B*T} (all)")

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    try:
        for l, vec in steering_by_layer.items():
            handles.append(
                _get_block(model, l).register_forward_hook(make_hook(vec, l), prepend=True)
            )
        yield
    finally:
        for h in handles:
            h.remove()


def _build_user_mask(
    tokenizer: AutoTokenizer,
    raw_prompts: List[str],
    formatted_prompts: List[str],
    input_ids: torch.Tensor,
    max_length: int,
    debug: bool = False,
) -> torch.Tensor:
    """Return a ``[B, T]`` bool mask marking tokens of the user's content span.

    Uses the fast tokenizer's ``return_offsets_mapping`` to locate which tokens
    cover the character span ``formatted.find(raw) : find(raw) + len(raw)``.
    Any token whose char-span falls strictly inside the user content span is
    marked True; chat-template scaffolding, special tokens (which get
    ``(0, 0)`` offsets) and left-padding tokens are all False.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "steer_positions='user_only' requires a fast tokenizer "
            "(for return_offsets_mapping). Got a slow tokenizer."
        )

    enc = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    B, T = input_ids.shape
    if tuple(offsets.shape[:2]) != (B, T):
        raise RuntimeError(
            f"offset_mapping shape {tuple(offsets.shape)} incompatible with "
            f"input_ids shape {(B, T)}; tokenizer re-encoding diverged."
        )

    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, (formatted, raw) in enumerate(zip(formatted_prompts, raw_prompts)):
        start_char = formatted.find(raw)
        if start_char == -1:
            raise ValueError(
                f"Could not locate raw prompt {raw!r} inside formatted prompt "
                f"{formatted!r}; chat template may have mutated the content."
            )
        end_char = start_char + len(raw)
        row_offsets = offsets[i].tolist()
        for j, (s, e) in enumerate(row_offsets):
            if s == 0 and e == 0:
                continue
            if s >= start_char and e <= end_char:
                mask[i, j] = True
        if debug:
            row_ids = input_ids[i].tolist()
            steered_ids = [row_ids[j] for j in range(T) if mask[i, j]]
            decoded = tokenizer.decode(steered_ids, skip_special_tokens=False)
            n = int(mask[i].sum().item())
            print(
                f"[steer-debug] user_mask row={i} span=chars[{start_char}:{end_char}] "
                f"n_steered_tokens={n} decoded_steered={decoded!r}"
            )
    return mask


@torch.no_grad()
def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    emotion_vectors_by_layer: Dict[int, torch.Tensor],
    emotions: List[str],
    selected_emotion: str,
    coeff: float = 1.0,
    layers: Optional[Union[int, List[int]]] = None,
    normalize_steering_vector: bool = True,
    N: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    batch_size: int = 8,
    max_length: int = 2048,
    seed: int = 0,
    steer_positions: str = "all",
    apply_chat_template: bool = True,
    max_samples: Optional[int] = None,
    max_parallel_completions: Optional[int] = None,
    steer_debug: bool = False,
    **_ignored,
) -> List[List[str]]:
    """Generate completions with (coeff * emotion_vector) added at chosen layers.

    Args:
        model: HF causal LM already loaded (see `emotion_probe.load_model_and_tokenizer`).
        tokenizer: matched tokenizer; pad token + left-padding are enforced for generate().
        prompts: User-turn prompts.
        emotion_vectors_by_layer: dict {layer_idx: tensor [num_emotions, hidden]}.
            Build via `emotion_probe.load_emotion_vectors` (which handles the
            `start_at_nth_token` / `denoised` / `version` extraction variants).
        emotions: ordered emotion names matching the rows of each tensor.
        selected_emotion: single emotion name whose vector is used for steering.
        coeff: multiplier applied to the (optionally unit-normalized) vector;
            negative values steer away from the emotion.
        layers: layer index, list of indices, or None (default: a single layer
            ~2/3 through the model, matching emotion_probe's default).
        normalize_steering_vector: if True (default), unit-normalize each
            layer's stored vector before scaling by ``coeff`` — makes ``coeff``
            a direct residual-stream magnitude and interpretable across
            emotions whose raw vectors have different norms. Set False to
            scale the raw vector (rarely what you want).
        N: completions per prompt (HF `num_return_sequences`).
        max_new_tokens: completion length cap.
        temperature, top_p, do_sample: standard HF sampling knobs.
        batch_size: forward-pass batch size. Left-padded for generation.
        max_length: tokenizer truncation cap for input prompts (in tokens).
        seed: torch manual seed for reproducibility.
        steer_positions: which token positions receive the steering vector.
            ``"all"`` steers every forward pass (prompt + generation);
            ``"generation_only"`` skips the prompt-encoding pass so only newly
            generated tokens are steered; ``"user_only"`` steers only the
            tokens covering the user's content span inside the chat-templated
            prompt (chat-template scaffolding and generation tokens are left
            untouched). ``"user_only"`` requires a fast tokenizer and
            ``apply_chat_template=True`` to have a meaningful span to target.
        apply_chat_template: wrap each prompt as a user-turn chat message.
        max_samples: optional truncation of `prompts` for testing.
        max_parallel_completions: if set, cap HF `num_return_sequences` per
            `model.generate()` call at this value and loop to accumulate the
            full ``N`` completions. Use this when long prompts × large N blow
            up the KV cache (e.g. the sue-prompt at N=50 on a 24 GB GPU).
            When None, a single call produces all N completions (previous
            behaviour, maximal throughput).
        steer_debug: if True, print per-hook diagnostics (phase, batch/seq
            shape, number of positions actually receiving the steering vector,
            and — for ``user_only`` — the decoded mask contents).
        **_ignored: swallow kwargs the other generate_* backends accept so
            callers can pass the same config dict interchangeably.

    Returns:
        List (per prompt) of lists (length N) of decoded completion strings,
        with the prompt tokens stripped.
    """
    from src.utils.emotion_probe import default_layer

    if max_samples is not None:
        prompts = prompts[:max_samples]
    if not prompts:
        return []

    if steer_positions not in {"all", "generation_only", "user_only"}:
        raise ValueError(
            f"steer_positions must be 'all', 'generation_only', or 'user_only', "
            f"got {steer_positions!r}"
        )
    if steer_positions == "user_only" and not apply_chat_template:
        raise ValueError(
            "steer_positions='user_only' requires apply_chat_template=True so "
            "that there is a distinguishable user-content span to target."
        )

    if layers is None:
        layers = [default_layer(model)]
    elif isinstance(layers, int):
        layers = [layers]
    layers = list(layers)

    steering_by_layer = _build_steering_by_layer(
        emotion_vectors_by_layer, emotions, selected_emotion, layers, coeff,
        normalize_steering_vector,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if apply_chat_template:
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
    else:
        formatted = list(prompts)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device

    state = {"is_generation": False, "user_mask": None}

    def _toggle_pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if isinstance(ids, torch.Tensor) and ids.dim() == 2:
            state["is_generation"] = ids.shape[1] == 1
        return None

    toggle_handle = model.register_forward_pre_hook(_toggle_pre_hook, with_kwargs=True)

    print(
        f"[steer] model={type(model).__name__} emotion={selected_emotion} "
        f"coeff={coeff} layers={layers} "
        f"normalize_steering_vector={normalize_steering_vector} "
        f"steer_positions={steer_positions} prompts={len(prompts)} N={N}"
    )

    per_call_N = N if max_parallel_completions is None else max(1, min(N, int(max_parallel_completions)))
    n_chunks = (N + per_call_N - 1) // per_call_N

    results: List[List[str]] = []
    try:
        with _steering_hooks(
            model,
            steering_by_layer,
            steer_positions=steer_positions,
            state=state,
            debug=steer_debug,
        ):
            n_batches = (len(formatted) + batch_size - 1) // batch_size
            bos_str = tokenizer.bos_token if isinstance(tokenizer.bos_token, str) else None
            for b_idx in range(n_batches):
                batch = formatted[b_idx * batch_size : (b_idx + 1) * batch_size]
                raw_batch = prompts[b_idx * batch_size : (b_idx + 1) * batch_size]
                # Avoid a spurious double-BOS when the chat template already
                # emits a literal <bos> at the start of every formatted prompt.
                add_special = not (
                    apply_chat_template
                    and bos_str
                    and all(s.startswith(bos_str) for s in batch)
                )
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special,
                ).to(device)

                if steer_positions == "user_only":
                    state["user_mask"] = _build_user_mask(
                        tokenizer,
                        raw_prompts=raw_batch,
                        formatted_prompts=batch,
                        input_ids=enc["input_ids"].cpu(),
                        max_length=max_length,
                        debug=steer_debug,
                    )
                else:
                    state["user_mask"] = None

                per_prompt: List[List[str]] = [[] for _ in batch]
                produced = 0
                for chunk_idx in range(n_chunks):
                    chunk_N = min(per_call_N, N - produced)
                    if chunk_N <= 0:
                        break
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        num_return_sequences=chunk_N,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    prompt_len = enc["input_ids"].shape[1]
                    new_tokens = out[:, prompt_len:]
                    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    for i in range(len(batch)):
                        per_prompt[i].extend(decoded[i * chunk_N : (i + 1) * chunk_N])
                    produced += chunk_N
                    del out, new_tokens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                for comps in per_prompt:
                    results.append(comps)
    finally:
        toggle_handle.remove()
        tokenizer.padding_side = prev_padding_side

    print(f"[steer] finished {len(prompts)} prompts x {N} completions")
    return results
