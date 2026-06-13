"""8-emotion pre-prompt ablation on neutral alpaca completions.

Tests the pre-prompt mechanism with NO prefill contradiction:

For each (prompt_idx, completion_idx):
  1. Generate plain alpaca continuation (no emotion priming) and pick a cutoff.
  2. For each condition c in {control, 8 emotion prompts}: build
       user_text = (prefix or none) + prompt
     and continue from kept neutral prefill at the same cutoff. Probe cleanly.

8 prompted emotions:
  Positive: happy, calm, excited, grateful
  Negative: frustrated, sad, angry, anxious

Prefix template: "Respond as though you are {emotion}."
Neutral control: "Please respond thoroughly and clearly."

Expected: prompted-emotion probe rises in its own condition vs the matched
control row (paired Δ). Since the prefill is neutral, there is no
contradiction → expected direction.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

import fire
import torch

from scripts.extras.calm_injection_experiment import (
    _build_prompt_ids,
    _generate,
    _pick_random_cutoff,
    _probe_per_token,
    load_prompts,
)
from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)

EMOTIONS_8 = [
    ("happy", "positive"),
    ("calm", "positive"),
    ("excited", "positive"),
    ("grateful", "positive"),
    ("frustrated", "negative"),
    ("sad", "negative"),
    ("angry", "negative"),
    ("anxious", "negative"),
]

CONTROL_PREFIX = "Please respond thoroughly and clearly."

PRE_TEMPLATE = "Respond as though you are {emo}."


def _means(arr, s, e, emotions):
    if e <= s:
        return {k: float("nan") for k in emotions}
    sub = arr[s:e]
    return {k: float(sub[:, i].mean()) for i, k in enumerate(emotions)}


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 20,
    n_completions: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    prompt_source: str = "alpaca",
    out_dir: str = "results/alpaca_emotion_preprompt",
    debug: bool = False,
):
    if debug:
        n_prompts = 3
        n_completions = 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[a8] loading {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[a8] loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)
    print(f"[a8] layer={layer}")

    probe_emos = [e for e, _ in EMOTIONS_8]
    emotions_all, vec_by_layer = load_emotion_vectors(
        model, start_at_nth_token=start_at_nth_token, version=version,
    )
    missing = [e for e in probe_emos if e not in emotions_all]
    if missing:
        raise RuntimeError(f"missing emotion vectors: {missing}")
    sel_idx = [emotions_all.index(e) for e in probe_emos]
    device = next(mdl.parameters()).device
    dtype = next(mdl.parameters()).dtype
    vec_dot = vec_by_layer[layer][sel_idx].to(device=device, dtype=dtype)
    vec_unit = torch.nn.functional.normalize(vec_dot, dim=-1)

    prompts = load_prompts(n_prompts, seed=seed, source=prompt_source)
    rng = random.Random(seed + 41)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": model, "n_prompts": n_prompts,
            "n_completions": n_completions, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "layer": int(layer), "start_at_nth_token": start_at_nth_token,
            "version": version, "prompt_source": prompt_source,
            "probe_emotions": probe_emos,
            "emotions_8": [{"emotion": e, "valence": v} for e, v in EMOTIONS_8],
            "control_prefix": CONTROL_PREFIX,
            "pre_template": PRE_TEMPLATE,
            "prompts": prompts,
        }, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.jsonl")
    runs_f = open(runs_path, "w")

    try:
        for k, rec in enumerate(prompts):
            prompt = rec["prompt"]
            ids_user_plain = _build_prompt_ids(tok, prompt)
            for c_idx in range(n_completions):
                seed_p = seed + k * 1000 + c_idx
                gen_plain = _generate(
                    mdl, tok, ids_user_plain, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                if len(gen_plain) < 10:
                    print(f"  [skip] prompt {k} c_idx {c_idx}: neutral too short")
                    continue
                cutoff = _pick_random_cutoff(rng, len(gen_plain))
                kept = gen_plain[:cutoff + 1]

                conditions: List[tuple] = [("control", "neutral", CONTROL_PREFIX)]
                for emo, val in EMOTIONS_8:
                    conditions.append((emo, val, PRE_TEMPLATE.format(emo=emo)))

                for label, valence, prefix in conditions:
                    user_text = prefix + "\n\n" + prompt if prefix else prompt
                    ids_user = _build_prompt_ids(tok, user_text)
                    cell_seed = seed + k * 10000 + c_idx * 200 + hash(label) % 1000
                    full_prefix = ids_user + kept
                    cont_ids = _generate(
                        mdl, tok, full_prefix, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p, do_sample=True, seed=cell_seed,
                    )
                    if not cont_ids:
                        continue
                    full = full_prefix + cont_ids
                    P = len(ids_user)
                    cont_start = P + len(kept)
                    cont_end = cont_start + len(cont_ids)
                    sc = _probe_per_token(mdl, tok, full, layer, vec_dot, vec_unit)
                    row = {
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "prompt": prompt,
                        "label": label,
                        "valence": valence,
                        "prefix": prefix,
                        "cutoff": int(cutoff),
                        "n_kept": int(len(kept)),
                        "n_cont": int(len(cont_ids)),
                        "kept_text": tok.decode(kept, skip_special_tokens=False),
                        "continued_text": tok.decode(cont_ids, skip_special_tokens=False),
                        "metrics": {},
                    }
                    for metric_name, arr in [("dot", sc["dot"]), ("cos", sc["cos"])]:
                        row["metrics"][metric_name] = {
                            "cont_mean":  _means(arr, cont_start, cont_end, probe_emos),
                            "cont_next1": _means(arr, cont_start, cont_start + 1, probe_emos),
                            "cont_next5": _means(arr, cont_start, min(cont_start + 5, cont_end), probe_emos),
                            "cont_next10": _means(arr, cont_start, min(cont_start + 10, cont_end), probe_emos),
                        }
                    runs_f.write(json.dumps(row) + "\n")
                    runs_f.flush()
                print(f"  [a8] prompt {k+1}/{len(prompts)} c_idx {c_idx+1}/{n_completions} done")
    finally:
        runs_f.close()
    print(f"[a8] wrote {runs_path}")


if __name__ == "__main__":
    fire.Fire(main)
