"""Pre-prompt placement experiment for the calm-injection follow-up.

Question: does prepending a calm instruction to the *user prompt itself*
(before the user's question) shift the model's emotion-probe activations
on the assistant continuation, when the assistant turn is prefilled with
frustrated content?

Compare three conditions with matched seeds and the same frustrated
prefill cutoff:

- baseline: original user prompt, no calm instruction anywhere
- pre_calm: "Please respond calmly and peacefully.\\n\\n{user_prompt}"
- pre_control: "Please respond thoroughly and clearly.\\n\\n{user_prompt}"

Each generates a continuation from the same frustrated prefill, then we
probe the continuation cleanly.
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
    load_prompts,
)
from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)

CALM_PREFIXES = [
    "Please respond calmly and peacefully.",
    "Take a deep breath before answering. Remain calm.",
    "Approach this with patience, ease, and tranquility.",
]

CONTROL_PREFIXES = [
    "Please respond thoroughly and clearly.",
    "Be detailed and precise in your reply.",
    "Approach this with structure, specificity, and care.",
]


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 20,
    n_completions: int = 5,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    prompt_source: str = "alpaca",
    out_dir: str = "results/calm_preprompt",
    debug: bool = False,
):
    """For each (prompt, completion_idx):
        1. Generate frustrated prefill via "respond in a frustrated tone" reminder.
        2. Pick a random cutoff inside the prefill.
        3. For each (pre-prompt variant): build user_prompt' = prefix + "\\n\\n" + user_prompt,
           templatize as user turn, append the prefill[:cutoff+1], continue generating, probe.
        4. Save mean probe value over continuation; also next1/next5/next10.

    Three prefix conditions: baseline (no prefix), calm prefix (3 paraphrases),
    control prefix (3 paraphrases). Calm[j] and Control[j] share the same cutoff
    within a (prompt, completion_idx) iteration.
    """
    if debug:
        n_prompts = 3
        n_completions = 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[pre] loading {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[pre] loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)

    emotions_all, vec_by_layer = load_emotion_vectors(
        model, start_at_nth_token=start_at_nth_token, version=version,
    )
    sel_idx = [emotions_all.index(e) for e in RELEVANT_EMOTIONS]
    device = next(mdl.parameters()).device
    dtype = next(mdl.parameters()).dtype
    vec_dot = vec_by_layer[layer][sel_idx].to(device=device, dtype=dtype)
    vec_unit = torch.nn.functional.normalize(vec_dot, dim=-1)

    prompts = load_prompts(n_prompts, seed=seed, source=prompt_source)
    rng = random.Random(seed + 17)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": model, "n_prompts": n_prompts, "n_completions": n_completions,
            "max_new_tokens": max_new_tokens, "seed": seed, "layer": int(layer),
            "version": version, "prompt_source": prompt_source,
            "relevant_emotions": RELEVANT_EMOTIONS,
            "calm_prefixes": CALM_PREFIXES, "control_prefixes": CONTROL_PREFIXES,
            "prompts": prompts,
        }, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.jsonl")
    runs_f = open(runs_path, "w")

    try:
        for k, rec in enumerate(prompts):
            prompt = rec["prompt"]
            ids_frust = _build_prompt_ids(tok, prompt + FRUSTRATED_REMINDER)
            ids_plain = _build_prompt_ids(tok, prompt)
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

                conditions: List[tuple] = [(None, "baseline", "")]
                for j, p in enumerate(CALM_PREFIXES):
                    conditions.append((j, "calm", p))
                for j, p in enumerate(CONTROL_PREFIXES):
                    conditions.append((j, "control", p))

                for pair_id, label, prefix in conditions:
                    if prefix:
                        user_text = prefix + "\n\n" + prompt
                    else:
                        user_text = prompt
                    user_ids = _build_prompt_ids(tok, user_text)
                    prefix_seq = user_ids + kept
                    cont_ids = _generate(
                        mdl, tok, prefix_seq, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p, do_sample=True,
                        seed=seed + k * 10000 + c_idx * 100 + (pair_id or 0),
                    )
                    if not cont_ids:
                        continue
                    full = user_ids + kept + cont_ids
                    P = len(user_ids)
                    sc = _probe_per_token(mdl, tok, full, layer, vec_dot, vec_unit)
                    cont_start = P + len(kept)
                    cont_end = cont_start + len(cont_ids)

                    def _means(arr, s, e):
                        if e <= s:
                            return {ek: float("nan") for ek in RELEVANT_EMOTIONS}
                        sub = arr[s:e]
                        return {ek: float(sub[:, i].mean()) for i, ek in enumerate(RELEVANT_EMOTIONS)}

                    row = {
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "prompt": prompt,
                        "pair_id": -1 if pair_id is None else int(pair_id),
                        "label": label,
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
                            "cont_mean": _means(arr, cont_start, cont_end),
                            "cont_next1": _means(arr, cont_start, cont_start + 1),
                            "cont_next5": _means(arr, cont_start, min(cont_start + 5, cont_end)),
                            "cont_next10": _means(arr, cont_start, min(cont_start + 10, cont_end)),
                        }
                    runs_f.write(json.dumps(row) + "\n")
                    runs_f.flush()
                print(f"  [pre] prompt {k+1}/{len(prompts)} c_idx {c_idx}/{n_completions} done")
    finally:
        runs_f.close()
    print(f"[pre] wrote {runs_path}")


if __name__ == "__main__":
    fire.Fire(main)
