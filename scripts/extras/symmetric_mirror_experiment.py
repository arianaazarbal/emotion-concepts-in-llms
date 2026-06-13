"""Symmetric mirror of Exp R: CALM prefill + frust vs calm vs ctrl pre-prompts.

Closes the backfire-symmetry hypothesis from the three earlier experiments:

  prefill direction | instruction direction | prediction
  ------------------+-----------------------+-----------
  frustrated        | frust (aligned)       | large amplification (Exp R: +0.045 cos)
  frustrated        | calm (contradicts)    | small backfire (v3-preprompt: +0.006 wrong-sign)
  CALM (this exp)   | calm (aligned)        | large amplification (predicted)
  CALM (this exp)   | frust (contradicts)   | small backfire (predicted: weak/null/wrong-sign on `frustrated`)

For each (prompt × c_idx):
  1. Generate calm prefill via CALM_REMINDER.
  2. For each j ∈ {0,1,2}: pick cutoff[j], then run 3 conditions at that cutoff:
     - calm_pre[j]:  "Respond as though you are {calm,peaceful,serene}"  → ALIGNED
     - frust_pre[j]: "Respond as though you are {frustrated,irritated,angry}" → CONTRADICTS
     - ctrl_pre[j]:  neutral prefix → control
  3. Plus one baseline cell per (prompt, c_idx) at cutoff[0].

Total cells: 12 × 3 × (1 + 3 × 3) = 360.
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
    RELEVANT_EMOTIONS,
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

CALM_REMINDER = (
    "\n\n(Respond in a calm, peaceful tone — express tranquility, patience, "
    "and serenity throughout your reply.)"
)

CALM_EMOS = ["calm", "peaceful", "serene"]
FRUST_EMOS = ["frustrated", "irritated", "angry"]

CTRL_PREFIXES = [
    "Please respond thoroughly and clearly.",
    "Be detailed and precise in your reply.",
    "Approach this with structure, specificity, and care.",
]


def _means(arr, s, e):
    if e <= s:
        return {k: float("nan") for k in RELEVANT_EMOTIONS}
    sub = arr[s:e]
    return {k: float(sub[:, i].mean()) for i, k in enumerate(RELEVANT_EMOTIONS)}


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 12,
    n_completions: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    prompt_source: str = "alpaca",
    out_dir: str = "results/symmetric_mirror",
    debug: bool = False,
):
    if debug:
        n_prompts = 3
        n_completions = 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[sym] loading {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[sym] loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)
    print(f"[sym] layer={layer}")

    emotions_all, vec_by_layer = load_emotion_vectors(
        model, start_at_nth_token=start_at_nth_token, version=version,
    )
    missing = [e for e in RELEVANT_EMOTIONS if e not in emotions_all]
    if missing:
        raise RuntimeError(f"missing emotion vectors: {missing}")
    sel_idx = [emotions_all.index(e) for e in RELEVANT_EMOTIONS]
    device = next(mdl.parameters()).device
    dtype = next(mdl.parameters()).dtype
    vec_dot = vec_by_layer[layer][sel_idx].to(device=device, dtype=dtype)
    vec_unit = torch.nn.functional.normalize(vec_dot, dim=-1)

    prompts = load_prompts(n_prompts, seed=seed, source=prompt_source)
    rng = random.Random(seed + 71)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": model, "n_prompts": n_prompts,
            "n_completions": n_completions, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "layer": int(layer), "start_at_nth_token": start_at_nth_token,
            "version": version, "prompt_source": prompt_source,
            "relevant_emotions": RELEVANT_EMOTIONS,
            "calm_reminder": CALM_REMINDER,
            "calm_pre_emos": CALM_EMOS,
            "frust_pre_emos": FRUST_EMOS,
            "ctrl_prefixes": CTRL_PREFIXES,
            "prompts": prompts,
        }, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.jsonl")
    runs_f = open(runs_path, "w")

    try:
        for k, rec in enumerate(prompts):
            prompt = rec["prompt"]
            ids_user_plain = _build_prompt_ids(tok, prompt)
            ids_user_calm = _build_prompt_ids(tok, prompt + CALM_REMINDER)
            for c_idx in range(n_completions):
                seed_p = seed + k * 1000 + c_idx
                gen_calm = _generate(
                    mdl, tok, ids_user_calm, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                if len(gen_calm) < 10:
                    print(f"  [skip] prompt {k} c_idx {c_idx}: calm prefill too short")
                    continue
                cutoffs = [_pick_random_cutoff(rng, len(gen_calm)) for _ in range(3)]

                conditions: List[tuple] = [(-1, "baseline", "none", "", cutoffs[0])]
                for j in range(3):
                    conditions.append((j, "calm_pre",  "calm",  f"Respond as though you are {CALM_EMOS[j]}.",  cutoffs[j]))
                    conditions.append((j, "frust_pre", "frust", f"Respond as though you are {FRUST_EMOS[j]}.", cutoffs[j]))
                    conditions.append((j, "ctrl_pre",  "ctrl",  CTRL_PREFIXES[j],                              cutoffs[j]))

                for j, label, direction, prefix, cutoff in conditions:
                    if prefix:
                        user_text = prefix + "\n\n" + prompt
                    else:
                        user_text = prompt
                    ids_user = _build_prompt_ids(tok, user_text)
                    kept = gen_calm[:cutoff + 1]
                    full_prefix = ids_user + kept
                    cell_seed = seed + k * 10000 + c_idx * 200 + (j + 1) * 17 + hash(label) % 11
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
                        "pair_id": int(j),
                        "label": label,
                        "direction": direction,
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
                            "cont_mean":  _means(arr, cont_start, cont_end),
                            "cont_next1": _means(arr, cont_start, cont_start + 1),
                            "cont_next5": _means(arr, cont_start, min(cont_start + 5, cont_end)),
                            "cont_next10": _means(arr, cont_start, min(cont_start + 10, cont_end)),
                        }
                    runs_f.write(json.dumps(row) + "\n")
                    runs_f.flush()
                print(f"  [sym] prompt {k+1}/{len(prompts)} c_idx {c_idx+1}/{n_completions} done")
    finally:
        runs_f.close()
    print(f"[sym] wrote {runs_path}")


if __name__ == "__main__":
    fire.Fire(main)
