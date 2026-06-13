"""Calm-instruction injection experiments.

Two experiments measuring how an inlined "remain calm / take a deep breath"
instruction shifts a model's per-token emotion-probe activations on the
remainder of an assistant turn:

1. **Baseline**: pick random single-turn chat prompts, generate a response,
   splice one of three calm-instruction snippets after a random intermediate
   token, continue generating, and compare emotion probes before/after.

2. **Prefilled-frustrated**: generate frustrated completions (with an
   "act frustrated" reminder appended), validate avg frustration is higher,
   then use those completions as a prefilled assistant turn for the
   no-reminder version of the prompt and run the same injection procedure.

Probes are context-aware (one forward pass over the full chat-formatted
sequence). Both metrics (dot product, cosine similarity) are computed.
Inline injection: the snippet is added as raw text to the assistant turn,
no chat-template scaffolding around it.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import fire
import numpy as np
import torch

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)

RELEVANT_EMOTIONS = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]

INJECTION_TEXTS = [
    "<system instruction> Remember to take a deep breath! </system instruction>",
    "<system instruction> Remember to remain calm. Everything is ok! </system instruction>",
    "<system instruction> Take a deep breath and smile :) </system instruction>",
    "<system instruction> Stay peaceful and serene as you continue. </system instruction>",
    "<system instruction> Approach this gently, with patience and ease. </system instruction>",
    "<system instruction> Relax. Slow down. Find your calm center. </system instruction>",
    "<system instruction> Be tranquil and unhurried in your reply. </system instruction>",
    "<system instruction> Let go of any tension. Respond from a place of quiet. </system instruction>",
]

NEUTRAL_CONTROL_TEXTS = [
    "<system instruction> Remember to continue the response. </system instruction>",
    "<system instruction> Remember to answer thoroughly. </system instruction>",
    "<system instruction> Continue the response and finish your answer. </system instruction>",
    "<system instruction> Be detailed and comprehensive as you continue. </system instruction>",
    "<system instruction> Use clear structure and concrete examples. </system instruction>",
    "<system instruction> Stay organized. Be specific. Cite facts. </system instruction>",
    "<system instruction> Be precise and unambiguous in your reply. </system instruction>",
    "<system instruction> Don't skip steps. Make every word count. </system instruction>",
]

FRUSTRATED_REMINDER = (
    "\n\n(Respond in a frustrated tone — express irritation, impatience, "
    "and exasperation throughout your reply.)"
)


def _safe_chat_tokenize(tokenizer, formatted: str) -> List[int]:
    """Tokenize a chat-formatted string, avoiding double-BOS."""
    bos = tokenizer.bos_token
    add_special = not (isinstance(bos, str) and bos and formatted.startswith(bos))
    ids = tokenizer(formatted, add_special_tokens=add_special)["input_ids"]
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError(f"Expected list[int] from tokenizer; got {type(ids).__name__}")
    return list(ids)


def _build_prompt_ids(tokenizer, prompt: str) -> List[int]:
    """Wrap as user-turn and tokenize."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _safe_chat_tokenize(tokenizer, formatted)


def _trim_eos(ids: List[int], tokenizer) -> List[int]:
    stop = set()
    for t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(t, int):
            stop.add(t)
    out = []
    for t in ids:
        if t in stop:
            break
        out.append(int(t))
    return out


@torch.no_grad()
def _generate(
    model, tokenizer, prefix_ids: List[int], max_new_tokens: int,
    temperature: float, top_p: float, do_sample: bool, seed: int,
) -> List[int]:
    """Continue generation from a flat list of token ids; return new ids (eos-trimmed)."""
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids], device=device)
    attn = torch.ones_like(inp)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    out = model.generate(
        input_ids=inp,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else 1.0,
        top_p=float(top_p) if do_sample else 1.0,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = out[0, len(prefix_ids):].tolist()
    return _trim_eos(new_ids, tokenizer)


@torch.no_grad()
def _probe_per_token(
    model, tokenizer, full_ids: List[int], layer: int,
    vec_dot: torch.Tensor, vec_unit: torch.Tensor,
) -> dict:
    """Single forward pass; return per-token dot and cosine for the *full sequence*.

    `vec_dot` has shape [E, H] (raw emotion vectors, used for dot product).
    `vec_unit` has shape [E, H] (unit-normalized, used for cosine).
    """
    device = next(model.parameters()).device
    inp = torch.tensor([full_ids], device=device)
    out = model(input_ids=inp, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # [S, H]
    dot_scores = (h @ vec_dot.T).float().cpu()  # [S, E]
    h_unit = torch.nn.functional.normalize(h, dim=-1)
    cos_scores = (h_unit @ vec_unit.T).float().cpu()  # [S, E]
    return {"dot": dot_scores.numpy(), "cos": cos_scores.numpy()}


def _pick_random_cutoff(rng: random.Random, n: int, lo: float = 0.3, hi: float = 0.7) -> int:
    """Random token index in [lo*n, hi*n)."""
    if n < 4:
        return max(0, n // 2)
    a, b = max(1, int(lo * n)), max(2, int(hi * n))
    return rng.randrange(a, b)


def _pick_period_cutoff(
    tokenizer, gen_ids: List[int], rng: random.Random, lo: float = 0.3, hi: float = 0.7,
) -> Optional[int]:
    """Pick a cutoff right after a period token (`.` or `!` / `?`) in [lo*n, hi*n)."""
    n = len(gen_ids)
    if n < 4:
        return None
    a, b = max(1, int(lo * n)), max(2, int(hi * n))
    candidates = []
    for i in range(a, b):
        s = tokenizer.decode([gen_ids[i]], skip_special_tokens=False)
        if any(p in s for p in [".", "!", "?"]):
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)


def load_prompts(n_prompts: int, seed: int, source: str = "alpaca") -> List[dict]:
    """Random single-turn chat prompts.

    Sources:
        alpaca: tatsu-lab/alpaca (filter to single-turn = empty `input` field).
        category_v2: mostly math/coding (low emotional content).
        advice: 9 emotional advice queries.
    """
    if source == "alpaca":
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        pool = [x["instruction"] for x in ds if not x["input"].strip()]
    elif source == "category_v2":
        with open("data/category_prompts_v2.json") as f:
            data = json.load(f)
        pool = [p["prompt"] for p in data["prompts"]]
    elif source == "advice":
        with open("data/advice_prompts.json") as f:
            data = json.load(f)
        pool = [p["query"] for p in data]
    else:
        raise ValueError(f"unknown prompt source: {source}")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(pool)), min(n_prompts, len(pool)))
    return [{"idx": i, "prompt": pool[i]} for i in idxs]


@dataclass
class ProbeRegion:
    """Mean per-emotion probe values over a [start, end) token span (assistant tokens only)."""
    label: str
    start: int  # full-sequence index
    end: int

    def slice_means(self, scores: np.ndarray, emotions: List[str]) -> dict:
        if self.end <= self.start:
            return {e: float("nan") for e in emotions}
        sub = scores[self.start:self.end]  # [span, E]
        return {e: float(sub[:, i].mean()) for i, e in enumerate(emotions)}

    def slice_values(self, scores: np.ndarray, emotions: List[str]) -> dict:
        """Per-token values (list of length end-start) per emotion."""
        if self.end <= self.start:
            return {e: [] for e in emotions}
        sub = scores[self.start:self.end]
        return {e: sub[:, i].tolist() for i, e in enumerate(emotions)}


def _gen_and_probe_one(
    model, tokenizer, prompt_ids: List[int], gen_ids: List[int],
    cutoff: int, injection_text: str, max_new_tokens: int,
    temperature: float, top_p: float, do_sample: bool, seed: int,
    layer: int, vec_dot: torch.Tensor, vec_unit: torch.Tensor,
) -> dict:
    """Run one injection: splice + continue + probe full sequence (with + without injection)."""
    P = len(prompt_ids)
    kept = gen_ids[: cutoff + 1]

    inj_ids = list(tokenizer(injection_text, add_special_tokens=False)["input_ids"])
    prefix_inj = prompt_ids + kept + inj_ids
    cont_ids = _generate(
        model, tokenizer, prefix_inj, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, do_sample=do_sample, seed=seed,
    )
    full_inj = prefix_inj + cont_ids

    full_base = prompt_ids + gen_ids

    scores_base = _probe_per_token(
        model, tokenizer, full_base, layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
    )
    scores_inj = _probe_per_token(
        model, tokenizer, full_inj, layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
    )

    n_kept, n_inj, n_cont = len(kept), len(inj_ids), len(cont_ids)
    base_start = P + cutoff + 1
    inj_start = P + n_kept + n_inj
    base_end = P + len(gen_ids)
    inj_end = P + n_kept + n_inj + n_cont

    def _windows(start: int, end: int, prefix: str) -> List[tuple]:
        return [
            (f"{prefix}_next1", ProbeRegion(f"{prefix}_next1", start, min(start + 1, end))),
            (f"{prefix}_next5", ProbeRegion(f"{prefix}_next5", start, min(start + 5, end))),
            (f"{prefix}_next10", ProbeRegion(f"{prefix}_next10", start, min(start + 10, end))),
            (f"{prefix}_rest", ProbeRegion(f"{prefix}_rest", start, end)),
        ]

    base_regions = _windows(base_start, base_end, "base")
    inj_regions = _windows(inj_start, inj_end, "inj")

    result = {
        "injection_text": injection_text,
        "cutoff": int(cutoff),
        "n_kept": int(n_kept),
        "n_inj_tokens": int(n_inj),
        "n_cont_tokens": int(n_cont),
        "n_baseline_gen_tokens": int(len(gen_ids)),
        "continued_text": tokenizer.decode(cont_ids, skip_special_tokens=False),
        "kept_text": tokenizer.decode(kept, skip_special_tokens=False),
        "metrics": {},
        # back-compat keys (also redundantly emitted under metrics):
    }
    for metric, scores_base_m, scores_inj_m in [
        ("dot", scores_base["dot"], scores_inj["dot"]),
        ("cos", scores_base["cos"], scores_inj["cos"]),
    ]:
        d = {}
        for name, rgn in base_regions:
            d[f"{name}_mean"] = rgn.slice_means(scores_base_m, RELEVANT_EMOTIONS)
        for name, rgn in inj_regions:
            d[f"{name}_mean"] = rgn.slice_means(scores_inj_m, RELEVANT_EMOTIONS)
        # back-compat aliases used by earlier runs / plot script
        d["base_rest_mean"] = d["base_rest_mean"]
        d["inj_rest_mean"] = d["inj_rest_mean"]
        d["base_next_token"] = d["base_next1_mean"]
        d["inj_next_token"] = d["inj_next1_mean"]
        result["metrics"][metric] = d
    return result


def run_exp1(
    model, tokenizer, prompt_records: List[dict],
    layer: int, vec_dot: torch.Tensor, vec_unit: torch.Tensor,
    max_new_tokens: int, temperature: float, top_p: float, seed: int,
    cutoff_strategy: str, out_dir: str, include_control: bool = False,
) -> dict:
    """Experiment 1: baseline generation + calm injection at random intermediate token.

    Saves per-example results to out_dir/exp1_runs.jsonl.
    """
    n_pair = len(INJECTION_TEXTS)
    inj_pairs: List[tuple] = []
    for j in range(n_pair):
        inj_pairs.append((j, INJECTION_TEXTS[j], "calm"))
        if include_control:
            inj_pairs.append((j, NEUTRAL_CONTROL_TEXTS[j], "control"))

    print(f"[exp1] running on {len(prompt_records)} prompts, cutoff={cutoff_strategy}, "
          f"injections={len(inj_pairs)} (control={include_control})")
    runs = []
    rng = random.Random(seed)
    runs_path = os.path.join(out_dir, "exp1_runs.jsonl")
    with open(runs_path, "w") as f:
        for k, rec in enumerate(prompt_records):
            prompt = rec["prompt"]
            prompt_ids = _build_prompt_ids(tokenizer, prompt)
            gen_ids = _generate(
                model, tokenizer, prompt_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, do_sample=True, seed=seed + k,
            )
            if len(gen_ids) < 6:
                print(f"  [skip] prompt {k}: baseline too short ({len(gen_ids)} tokens)")
                continue
            baseline_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            cutoffs_for_pairs: List[int] = []
            for _ in range(n_pair):
                if cutoff_strategy == "period":
                    c = _pick_period_cutoff(tokenizer, gen_ids, rng)
                    if c is None:
                        c = _pick_random_cutoff(rng, len(gen_ids))
                else:
                    c = _pick_random_cutoff(rng, len(gen_ids))
                cutoffs_for_pairs.append(c)
            for j_pair, inj_text, inj_label in inj_pairs:
                cutoff = cutoffs_for_pairs[j_pair]
                one = _gen_and_probe_one(
                    model, tokenizer, prompt_ids=prompt_ids, gen_ids=gen_ids,
                    cutoff=cutoff, injection_text=inj_text,
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    do_sample=True, seed=seed + k * 100 + j_pair,
                    layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
                )
                row = {
                    "exp": "exp1",
                    "prompt_idx": int(rec["idx"]),
                    "prompt": prompt,
                    "baseline_text": baseline_text,
                    "injection_id": j_pair,
                    "injection_label": inj_label,
                    "cutoff_strategy": cutoff_strategy,
                    **one,
                }
                runs.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"  [exp1] prompt {k+1}/{len(prompt_records)} done")
    print(f"[exp1] wrote {len(runs)} rows -> {runs_path}")
    return {"runs": runs, "path": runs_path}


def run_exp2(
    model, tokenizer, prompt_records: List[dict],
    layer: int, vec_dot: torch.Tensor, vec_unit: torch.Tensor,
    max_new_tokens: int, temperature: float, top_p: float, seed: int,
    cutoff_strategy: str, out_dir: str, include_control: bool = False,
    n_completions: int = 1,
) -> dict:
    """Experiment 2: prefill-frustrated + calm injection.

    For each prompt:
        (a) generate `n_completions` distinct frustrated prefills (different seeds);
        (b) for each prefill, validate that frustrated probe is higher than
            a plain (no-reminder) baseline generated with the same seed;
        (c) for each prefill, splice the calm-vs-control injections at matched
            cutoffs and probe.

    Saves all rows incrementally to avoid losing partial progress on long runs.
    """
    print(f"[exp2] running on {len(prompt_records)} prompts, n_completions={n_completions}")
    rng = random.Random(seed + 7)
    runs = []
    validation = []

    runs_path = os.path.join(out_dir, "exp2_runs.jsonl")
    val_path = os.path.join(out_dir, "exp2_validation.jsonl")
    runs_f = open(runs_path, "w")
    val_f = open(val_path, "w")

    try:
        for k, rec in enumerate(prompt_records):
            prompt = rec["prompt"]
            prompt_plain = prompt
            prompt_frust = prompt + FRUSTRATED_REMINDER

            ids_plain = _build_prompt_ids(tokenizer, prompt_plain)
            ids_frust = _build_prompt_ids(tokenizer, prompt_frust)

            for c_idx in range(n_completions):
                seed_p = seed + k * 1000 + c_idx
                gen_plain = _generate(
                    model, tokenizer, ids_plain, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                gen_frust = _generate(
                    model, tokenizer, ids_frust, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True, seed=seed_p,
                )
                if len(gen_plain) < 6 or len(gen_frust) < 6:
                    print(f"  [skip] prompt {k} c_idx {c_idx}: too short "
                          f"(plain={len(gen_plain)}, frust={len(gen_frust)})")
                    continue

                full_plain = ids_plain + gen_plain
                full_frust = ids_frust + gen_frust
                Pp, Pf = len(ids_plain), len(ids_frust)

                sc_plain = _probe_per_token(model, tokenizer, full_plain, layer, vec_dot, vec_unit)
                sc_frust = _probe_per_token(model, tokenizer, full_frust, layer, vec_dot, vec_unit)

                rgn_plain = ProbeRegion("plain", Pp, Pp + len(gen_plain))
                rgn_frust = ProbeRegion("frust", Pf, Pf + len(gen_frust))
                for metric, base_arr, frust_arr in [
                    ("dot", sc_plain["dot"], sc_frust["dot"]),
                    ("cos", sc_plain["cos"], sc_frust["cos"]),
                ]:
                    vrow = {
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "metric": metric,
                        "plain_mean": rgn_plain.slice_means(base_arr, RELEVANT_EMOTIONS),
                        "frust_mean": rgn_frust.slice_means(frust_arr, RELEVANT_EMOTIONS),
                    }
                    validation.append(vrow)
                    val_f.write(json.dumps(vrow) + "\n")
                val_f.flush()

                prefill_ids_only = gen_frust
                if len(prefill_ids_only) < 6:
                    continue

                n_pair = len(INJECTION_TEXTS)
                cutoffs_for_pairs: List[int] = []
                for _ in range(n_pair):
                    if cutoff_strategy == "period":
                        c = _pick_period_cutoff(tokenizer, prefill_ids_only, rng)
                        if c is None:
                            c = _pick_random_cutoff(rng, len(prefill_ids_only))
                    else:
                        c = _pick_random_cutoff(rng, len(prefill_ids_only))
                    cutoffs_for_pairs.append(c)

                inj_pairs: List[tuple] = []
                for j in range(n_pair):
                    inj_pairs.append((j, INJECTION_TEXTS[j], "calm"))
                    if include_control:
                        inj_pairs.append((j, NEUTRAL_CONTROL_TEXTS[j], "control"))
                for j_pair, inj_text, inj_label in inj_pairs:
                    cutoff = cutoffs_for_pairs[j_pair]
                    one = _gen_and_probe_one(
                        model, tokenizer, prompt_ids=ids_plain, gen_ids=prefill_ids_only,
                        cutoff=cutoff, injection_text=inj_text,
                        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                        do_sample=True,
                        seed=seed + k * 10000 + c_idx * 100 + j_pair + 1,
                        layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
                    )
                    row = {
                        "exp": "exp2",
                        "prompt_idx": int(rec["idx"]),
                        "completion_idx": int(c_idx),
                        "prompt": prompt_plain,
                        "frustrated_prefill_text": tokenizer.decode(gen_frust, skip_special_tokens=False),
                        "injection_id": j_pair,
                        "injection_label": inj_label,
                        "cutoff_strategy": cutoff_strategy,
                        **one,
                    }
                    runs.append(row)
                    runs_f.write(json.dumps(row) + "\n")
                runs_f.flush()
            print(f"  [exp2] prompt {k+1}/{len(prompt_records)} done "
                  f"(n_runs_so_far={len(runs)})")
    finally:
        runs_f.close()
        val_f.close()

    print(f"[exp2] wrote {len(runs)} injection rows -> {runs_path}")
    print(f"[exp2] wrote {len(validation)} validation rows -> {val_path}")
    return {"runs": runs, "validation": validation, "runs_path": runs_path, "val_path": val_path}


def main(
    model: str = "gemma2_9b",
    n_prompts: int = 20,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    layer: Optional[int] = None,
    start_at_nth_token: int = 50,
    version: str = "v0",
    cutoff_strategy: str = "random",
    exp: str = "both",
    prompt_source: str = "alpaca",
    out_dir: str = "results/calm_injection",
    debug: bool = False,
    include_control: bool = False,
    n_completions: int = 1,
):
    """Run calm-injection experiments.

    Args:
        model: short model name (gemma2_9b).
        n_prompts: how many random prompts to sample.
        max_samples: truncate prompts (for debug runs).
        max_new_tokens: cap on baseline and continuation lengths.
        seed: master seed for prompt sampling, cutoff choice, generation.
        layer: probe/decoder layer. None = default (26 for gemma2_9b).
        start_at_nth_token: emotion-vector extraction variant (50 = v0 default).
        version: emotion-vector version suffix (v0).
        cutoff_strategy: "random" or "period".
        exp: "exp1", "exp2", or "both".
        prompt_source: "category_v2" or "advice".
        out_dir: results directory.
        debug: shorthand for max_samples=3.
    """
    if debug:
        max_samples = 3
    if max_samples is not None:
        n_prompts = min(n_prompts, max_samples)

    os.makedirs(out_dir, exist_ok=True)

    print(f"[main] loading model {model} ...")
    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model)
    print(f"[main] model loaded in {time.time() - t0:.1f}s")
    if layer is None:
        layer = default_layer(mdl)
    print(f"[main] layer={layer}")

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
    print(f"[main] sampled {len(prompts)} prompts (source={prompt_source})")

    metadata = {
        "model": model,
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "layer": int(layer),
        "start_at_nth_token": start_at_nth_token,
        "version": version,
        "cutoff_strategy": cutoff_strategy,
        "prompt_source": prompt_source,
        "relevant_emotions": RELEVANT_EMOTIONS,
        "injection_texts": INJECTION_TEXTS,
        "frustrated_reminder": FRUSTRATED_REMINDER,
        "prompts": prompts,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if exp in ("exp1", "both"):
        run_exp1(
            mdl, tok, prompts, layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            seed=seed, cutoff_strategy=cutoff_strategy, out_dir=out_dir,
            include_control=include_control,
        )
    if exp in ("exp2", "both"):
        run_exp2(
            mdl, tok, prompts, layer=layer, vec_dot=vec_dot, vec_unit=vec_unit,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            seed=seed, cutoff_strategy=cutoff_strategy, out_dir=out_dir,
            include_control=include_control, n_completions=n_completions,
        )
    print(f"[main] done. outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
