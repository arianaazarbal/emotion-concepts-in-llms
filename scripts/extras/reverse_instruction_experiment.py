"""Reverse-instruction experiment: "respond as though you are frustrated"
on a frustrated prefill — both pre-prompt and inline placements.

Hypothesis: the preprompt-calm backfire we saw on a frustrated prefill is
driven by *contradiction* between the prefill direction and the instruction
direction. If so, a matching-direction instruction (frust prefill + frust
instruction) should *amplify* frustration, not reverse it.

For each (prompt_idx, completion_idx) and each j in {0,1,2}:
  1. Generate frustrated prefill once (seed depends on prompt_idx, c_idx).
  2. Pick one cutoff[j] inside the prefill.
  3. Run four matched conditions at that cutoff:
        - frust_pre[j]: prepend "Respond as though you are {emo[j]}" to user prompt
        - ctrl_pre[j]:  prepend a neutral pre-prompt to user prompt
        - frust_inline[j]: inline <system instruction> with same emo[j]
        - ctrl_inline[j]:  inline <system instruction> with neutral text

Plus one baseline (no instruction) per (prompt, completion) at cutoff[0].

Frustrated-direction tokens: {frustrated, irritated, angry}.
Neutral control tokens (matched to v3-preprompt):
  - "Please respond thoroughly and clearly."
  - "Be detailed and precise in your reply."
  - "Approach this with structure, specificity, and care."
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
    FRUSTRATED_REMINDER,
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

FRUST_EMOS = ["frustrated", "irritated", "angry"]

PRE_TEMPLATE = "Respond as though you are {emo}."

CTRL_PREFIXES = [
    "Please respond thoroughly and clearly.",
    "Be detailed and precise in your reply.",
    "Approach this with structure, specificity, and care.",
]

CTRL_INLINES = [
    "<system instruction> Remember to continue the response. </system instruction>",
    "<system instruction> Be detailed and comprehensive as you continue. </system instruction>",
    "<system instruction> Stay organized. Be specific. Cite facts. </system instruction>",
]


def _frust_pre(j: int) -> str:
    return PRE_TEMPLATE.format(emo=FRUST_EMOS[j])


def _frust_inline(j: int) -> str:
    return f"<system instruction> Respond as though you are {FRUST_EMOS[j]}. </system instruction>"


def _means(arr, s, e):
    if e <= s:
        return {k: float("nan") for k in RELEVANT_EMOTIONS}
    sub = arr[s:e]
    return {k: float(sub[:, i].mean()) for i, k in enumerate(RELEVANT_EMOTIONS)}


def _gen_one(
    mdl, tok, ids_user, kept, inline_text, max_new_tokens, temperature, top_p, seed,
):
    """Generate continuation for one condition.

    If inline_text is non-empty, splice it between kept and the new generation.
    Returns: (continued_ids, full_ids, n_inj_tokens, cont_start_idx, cont_end_idx).
    """
    if inline_text:
        inj_ids = list(tok(inline_text, add_special_tokens=False)["input_ids"])
    else:
        inj_ids = []
    prefix = list(ids_user) + list(kept) + inj_ids
    cont_ids = _generate(
        mdl, tok, prefix, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, do_sample=True, seed=seed,
    )
    full = prefix + cont_ids
    P = len(ids_user)
    cont_start = P + len(kept) + len(inj_ids)
    cont_end = cont_start + len(cont_ids)
    return cont_ids, full, len(inj_ids), cont_start, cont_end


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 15,
    n_completions: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    prompt_source: str = "alpaca",
    out_dir: str = "results/reverse_instruction",
    debug: bool = False,
):
    if debug:
        n_prompts = 3
        n_completions = 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[rev] loading {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[rev] loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)
    print(f"[rev] layer={layer}")

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
    rng = random.Random(seed + 23)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": model, "n_prompts": n_prompts,
            "n_completions": n_completions, "max_new_tokens": max_new_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "layer": int(layer), "start_at_nth_token": start_at_nth_token,
            "version": version, "prompt_source": prompt_source,
            "relevant_emotions": RELEVANT_EMOTIONS,
            "frust_pre_prompts": [_frust_pre(j) for j in range(3)],
            "frust_inline_prompts": [_frust_inline(j) for j in range(3)],
            "ctrl_pre_prompts": CTRL_PREFIXES,
            "ctrl_inline_prompts": CTRL_INLINES,
            "frustrated_reminder": FRUSTRATED_REMINDER,
            "prompts": prompts,
        }, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.jsonl")
    runs_f = open(runs_path, "w")

    try:
        for k, rec in enumerate(prompts):
            prompt = rec["prompt"]
            ids_user_plain = _build_prompt_ids(tok, prompt)
            ids_user_frust = _build_prompt_ids(tok, prompt + FRUSTRATED_REMINDER)
            for c_idx in range(n_completions):
                seed_p = seed + k * 1000 + c_idx
                gen_frust = _generate(
                    mdl, tok, ids_user_frust, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                if len(gen_frust) < 10:
                    print(f"  [skip] prompt {k} c_idx {c_idx}: prefill too short")
                    continue
                cutoffs = [_pick_random_cutoff(rng, len(gen_frust)) for _ in range(3)]

                conditions: List[tuple] = [(-1, "baseline", "none", "none", "", "", cutoffs[0])]
                for j in range(3):
                    conditions.append((j, "frust_pre",   "pre",    "frust", _frust_pre(j),   "", cutoffs[j]))
                    conditions.append((j, "ctrl_pre",    "pre",    "ctrl",  CTRL_PREFIXES[j], "", cutoffs[j]))
                    conditions.append((j, "frust_inline","inline", "frust", "", _frust_inline(j), cutoffs[j]))
                    conditions.append((j, "ctrl_inline", "inline", "ctrl",  "", CTRL_INLINES[j],  cutoffs[j]))

                for j, label, placement, direction, pre_text, inline_text, cutoff in conditions:
                    if pre_text:
                        user_text = pre_text + "\n\n" + prompt
                    else:
                        user_text = prompt
                    ids_user = _build_prompt_ids(tok, user_text)
                    kept = gen_frust[:cutoff + 1]
                    cell_seed = seed + k * 10000 + c_idx * 200 + (j + 1) * 13 + hash(label) % 7
                    cont_ids, full, n_inj, cont_start, cont_end = _gen_one(
                        mdl, tok, ids_user, kept, inline_text,
                        max_new_tokens, temperature, top_p, cell_seed,
                    )
                    if not cont_ids:
                        continue
                    sc = _probe_per_token(mdl, tok, full, layer, vec_dot, vec_unit)
                    row = {
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "prompt": prompt,
                        "pair_id": int(j),
                        "label": label,
                        "placement": placement,
                        "direction": direction,
                        "pre_text": pre_text,
                        "inline_text": inline_text,
                        "cutoff": int(cutoff),
                        "n_kept": int(len(kept)),
                        "n_inj_tokens": int(n_inj),
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
                print(f"  [rev] prompt {k+1}/{len(prompts)} c_idx {c_idx+1}/{n_completions} done")
    finally:
        runs_f.close()
    print(f"[rev] wrote {runs_path}")


if __name__ == "__main__":
    fire.Fire(main)
