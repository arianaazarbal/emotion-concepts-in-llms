"""Steering-validation experiment for the calm-injection follow-up.

Question: when the model is generated WITH the calm vector added to its
residual stream at layer L (generate_with_steering), does the resulting
text — probed cleanly without steering — show higher calm-probe values
than an unsteered continuation?

If yes: probes are responsive to "actually calm-flavored" generated text,
and the v3 null we saw means inline `<system instruction>` snippets
don't push the model's generation toward calm semantically.

If no: even direct activation steering doesn't shift the probe — that
would be a much bigger concern about the probes/vectors.

We compare on the *frustrated prefill* setup (same as exp2): plain user
prompt + frustrated assistant prefill, then continue under steering.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

import fire
import numpy as np
import torch

from scripts.extras.calm_injection_experiment import (
    FRUSTRATED_REMINDER,
    RELEVANT_EMOTIONS,
    _build_prompt_ids,
    _generate,
    _probe_per_token,
    _safe_chat_tokenize,
    load_prompts,
)
from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)


@torch.no_grad()
def _generate_with_steering_at(
    model, tokenizer, prefix_ids: List[int],
    vec: torch.Tensor, layer: int, coeff: float,
    max_new_tokens: int, temperature: float, top_p: float, seed: int,
) -> List[int]:
    """Generate continuation while adding (coeff * vec) to layer L output, gen-only.

    Lightweight inline version of generate_with_steering, so we control
    exactly which positions get steered.
    """
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids], device=device)
    attn = torch.ones_like(inp)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    block = None
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "transformer.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            block = obj[layer - 1]
            break
    if block is None:
        raise RuntimeError("could not find transformer block list on model")

    state = {"is_generation": False}

    def pre_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if isinstance(ids, torch.Tensor) and ids.dim() == 2:
            state["is_generation"] = ids.shape[1] == 1
        return None

    vec_t = vec.to(device=device, dtype=next(model.parameters()).dtype)

    def block_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["is_generation"]:
            h = h + coeff * vec_t
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    h1 = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = block.register_forward_hook(block_hook, prepend=True)
    try:
        out = model.generate(
            input_ids=inp, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        h1.remove(); h2.remove()
    new_ids = out[0, len(prefix_ids):].tolist()
    stops = set()
    for t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(t, int):
            stops.add(t)
    trimmed = []
    for t in new_ids:
        if t in stops:
            break
        trimmed.append(int(t))
    return trimmed


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 12,
    n_completions: int = 3,
    coeffs: tuple = (-2.0, 0.0, 2.0, 4.0, 6.0),
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    prompt_source: str = "alpaca",
    out_dir: str = "results/calm_steering_validation",
    debug: bool = False,
):
    """Generate with calm-vector steering at given layer, probe cleanly.

    For each (prompt, completion_idx), pick a random cutoff inside the
    frustrated prefill (same setup as exp2), then generate continuations
    at multiple steering coefficients. Probe each continuation cleanly
    (no steering in the probe forward pass) and report mean per-emotion
    probe values on the continuation tokens.

    `coeffs` is a tuple of unit-vector multipliers; 0.0 is the no-steer
    control.
    """
    if debug:
        n_prompts = 3
        n_completions = 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[steer-val] loading {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[steer-val] loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)
    print(f"[steer-val] layer={layer}")

    emotions_all, vec_by_layer = load_emotion_vectors(
        model, start_at_nth_token=start_at_nth_token, version=version,
    )
    if "calm" not in emotions_all:
        raise RuntimeError("'calm' not in emotion_vectors")
    calm_idx = emotions_all.index("calm")
    sel_idx = [emotions_all.index(e) for e in RELEVANT_EMOTIONS]
    device = next(mdl.parameters()).device
    dtype = next(mdl.parameters()).dtype
    vec_dot = vec_by_layer[layer][sel_idx].to(device=device, dtype=dtype)
    vec_unit = torch.nn.functional.normalize(vec_dot, dim=-1)
    calm_vec_unit = torch.nn.functional.normalize(
        vec_by_layer[layer][calm_idx].to(device=device, dtype=dtype), dim=-1,
    )

    prompts = load_prompts(n_prompts, seed=seed, source=prompt_source)
    rng = random.Random(seed + 11)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": model, "n_prompts": n_prompts,
            "n_completions": n_completions, "coeffs": list(coeffs),
            "max_new_tokens": max_new_tokens, "seed": seed, "layer": int(layer),
            "start_at_nth_token": start_at_nth_token, "version": version,
            "prompt_source": prompt_source,
            "relevant_emotions": RELEVANT_EMOTIONS,
            "prompts": prompts,
        }, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.jsonl")
    runs_f = open(runs_path, "w")

    try:
        for k, rec in enumerate(prompts):
            prompt = rec["prompt"]
            ids_plain = _build_prompt_ids(tok, prompt)
            ids_frust = _build_prompt_ids(tok, prompt + FRUSTRATED_REMINDER)
            for c_idx in range(n_completions):
                seed_p = seed + k * 1000 + c_idx
                gen_frust = _generate(
                    mdl, tok, ids_frust, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                if len(gen_frust) < 10:
                    continue
                n = len(gen_frust)
                cutoff = rng.randrange(max(1, int(0.3 * n)), max(2, int(0.7 * n)))
                kept = gen_frust[:cutoff + 1]
                for coeff in coeffs:
                    cont_ids = _generate_with_steering_at(
                        mdl, tok, ids_plain + kept,
                        vec=calm_vec_unit, layer=layer, coeff=float(coeff),
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, seed=seed + k * 10000 + c_idx * 100 + int((coeff + 10) * 7),
                    )
                    if not cont_ids:
                        continue
                    full = ids_plain + kept + cont_ids
                    P = len(ids_plain)
                    sc = _probe_per_token(mdl, tok, full, layer, vec_dot, vec_unit)
                    base_start = P + cutoff + 1
                    base_end = P + len(kept)
                    cont_start = P + len(kept)
                    cont_end = cont_start + len(cont_ids)

                    def _means(arr, s, e):
                        if e <= s:
                            return {e_: float("nan") for e_ in RELEVANT_EMOTIONS}
                        sub = arr[s:e]
                        return {e_: float(sub[:, i].mean()) for i, e_ in enumerate(RELEVANT_EMOTIONS)}

                    row = {
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "prompt": prompt,
                        "coeff": float(coeff),
                        "cutoff": int(cutoff),
                        "n_kept": int(len(kept)),
                        "n_cont": int(len(cont_ids)),
                        "kept_text": tok.decode(kept, skip_special_tokens=False),
                        "continued_text": tok.decode(cont_ids, skip_special_tokens=False),
                        "metrics": {},
                    }
                    for metric_name, arr in [("dot", sc["dot"]), ("cos", sc["cos"])]:
                        row["metrics"][metric_name] = {
                            "cont_mean": _means(arr, cont_start, cont_end),
                            "cont_next1": _means(arr, cont_start, cont_start + 1),
                            "cont_next5": _means(arr, cont_start, min(cont_start + 5, cont_end)),
                            "cont_next10": _means(arr, cont_start, min(cont_start + 10, cont_end)),
                        }
                    runs_f.write(json.dumps(row) + "\n")
                    runs_f.flush()
                print(f"  [steer-val] prompt {k+1}/{len(prompts)} c_idx {c_idx}/{n_completions} done")
    finally:
        runs_f.close()
    print(f"[steer-val] wrote {runs_path}")


if __name__ == "__main__":
    fire.Fire(main)
