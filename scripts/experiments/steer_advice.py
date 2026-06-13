"""
Emotion-vector steering on advice prompts; LLM-judge which option each
steered generation recommends.

For each advice prompt in ``data/advice_prompts.json`` the script:
  1. Runs a no-steering baseline generation (N completions).
  2. For each emotion listed under ``option_1_emotions`` / ``option_2_emotions``
     (or an explicit ``--emotions`` override), runs steered generations at
     every non-zero strength in ``--strengths``. Steering is applied at *all*
     token positions (prompt + generation), at a single residual-stream layer.
  3. Classifies every completion with an LLM judge (default
     ``openai/gpt-4o-mini``) into ``option_1`` / ``option_2`` / ``unclear``.
  4. Plots the rate of recommending option_1 vs. steering strength, one curve
     per emotion, with a secondary panel showing the unclear-rate so mis-judged
     samples are visible at a glance.

Calibration tips:
  * ``--prompt_id`` restricts to a single prompt so you can iterate quickly.
  * Generations are persisted to ``generations.json`` and reused on re-run
    (unless ``--force_rerun_generation``). Judge responses are cached to a
    JSONL alongside them. Either can be skipped via ``--skip_generation`` or
    ``--skip_judge`` for partial reruns.

Outputs (TAG = ``{model}_{version}`` or ``{model}`` if version empty):
    results/steer_advice/{TAG}[/start_at_token_{N}][/denoised_layer_{L}]/{metric}/
      layer_{L}/
        {prompt_id}/
          generations.json
          judge_responses.jsonl
          results.json
          plot.png
        all_prompts_summary.json
"""

import asyncio
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.generate_openrouter import generate as openrouter_generate
from src.utils.generate_with_steering import generate_with_steering
from src.utils.utils import dump_json, load_json

load_dotenv()


def resolve_prompt_dir(
    model: str,
    version: str,
    start_at_nth_token: int,
    layer: int,
    denoised: bool,
    normalize_steering_vector: bool,
    prompt_id: str,
    steer_positions: str = "all",
    output_dir_override: Optional[str] = None,
) -> str:
    """Compute the per-prompt output directory following project conventions.

    When ``steer_positions != "all"`` a ``/steer_{positions}`` segment is
    inserted before ``layer_{L}`` so that ``all`` / ``generation_only`` /
    ``user_only`` runs never overwrite one another. The default ``"all"`` path
    is left unchanged for backwards compatibility with already-cached results.
    """
    model_tag = f"{model}_{version}" if version else model
    if output_dir_override:
        return f"{output_dir_override}/{prompt_id}"
    suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
    if denoised:
        suffix += f"/denoised_layer_{layer}"
    suffix += "/unit_normalized" if normalize_steering_vector else "/raw_magnitude"
    if steer_positions != "all":
        suffix += f"/steer_{steer_positions}"
    return f"results/steer_advice/{model_tag}{suffix}/layer_{layer}/{prompt_id}"


def collect_emotions_for_prompt(spec: dict) -> List[str]:
    """Return de-duplicated union of option_1 and option_2 emotions in order."""
    seen = set()
    out = []
    for e in list(spec.get("option_1_emotions", [])) + list(spec.get("option_2_emotions", [])):
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _strength_key(s: float) -> str:
    return f"{float(s):g}"


def run_generations(
    spec: dict,
    strengths: List[float],
    emotions: List[str],
    model,
    tokenizer,
    vectors_by_layer,
    all_emotions: List[str],
    layer: int,
    N: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
    steer_positions: str,
    normalize_steering_vector: bool,
    batch_size: int,
    max_parallel_completions: Optional[int],
    output_path: str,
    force_rerun: bool,
    force_rerun_baseline: bool = False,
    steer_debug: bool = False,
) -> dict:
    """Run steered + baseline generations for one prompt; cache to ``output_path``.

    Each (emotion, strength) call is cached by its key in ``steered`` (and the
    baseline in ``baseline``); existing keys are skipped unless ``force_rerun``.
    """
    if os.path.exists(output_path) and not force_rerun:
        cached = load_json(output_path)
    else:
        cached = None

    fresh = {
        "prompt_id": spec["id"],
        "axis": spec.get("axis"),
        "query": spec["query"],
        "option_1": spec["option_1"],
        "option_2": spec["option_2"],
        "option_1_emotions": list(spec.get("option_1_emotions", [])),
        "option_2_emotions": list(spec.get("option_2_emotions", [])),
        "layer": int(layer),
        "N": int(N),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "do_sample": bool(do_sample),
        "seed": int(seed),
        "steer_positions": steer_positions,
        "normalize_steering_vector": bool(normalize_steering_vector),
        "strengths": sorted({float(s) for s in strengths}),
        "emotions": list(dict.fromkeys(emotions)),
        "baseline": None,
        "steered": {e: {} for e in emotions},
    }
    if cached is None:
        cached = fresh
    else:
        cached["strengths"] = sorted({float(s) for s in cached.get("strengths", []) + strengths})
        cached["emotions"] = list(dict.fromkeys(list(cached.get("emotions", [])) + emotions))
        cached.setdefault("steered", {})
        for e in cached["emotions"]:
            cached["steered"].setdefault(e, {})

    common = dict(
        model=model,
        tokenizer=tokenizer,
        prompts=[spec["query"]],
        emotion_vectors_by_layer={layer: vectors_by_layer[layer]},
        emotions=all_emotions,
        layers=[layer],
        normalize_steering_vector=normalize_steering_vector,
        N=N,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        batch_size=batch_size,
        seed=seed,
        steer_positions=steer_positions,
        apply_chat_template=True,
        max_parallel_completions=max_parallel_completions,
        steer_debug=steer_debug,
    )

    nonzero_strengths = [float(s) for s in strengths if float(s) != 0.0]
    include_baseline = any(float(s) == 0.0 for s in strengths)

    if include_baseline and (
        cached.get("baseline") is None or force_rerun or force_rerun_baseline
    ):
        print(f"[steer_advice] baseline coeff=0 for {spec['id']} (N={N})")
        baseline_emotion = emotions[0] if emotions else all_emotions[0]
        result = generate_with_steering(coeff=0.0, selected_emotion=baseline_emotion, **common)
        cached["baseline"] = result[0]
        dump_json(cached, output_path)

    for emotion in emotions:
        for strength in nonzero_strengths:
            key = _strength_key(strength)
            existing = cached["steered"].get(emotion, {}).get(key)
            if existing is not None and not force_rerun:
                continue
            print(f"[steer_advice] {spec['id']} emotion={emotion} coeff={strength}")
            result = generate_with_steering(
                coeff=float(strength), selected_emotion=emotion, **common,
            )
            cached["steered"].setdefault(emotion, {})[key] = result[0]
            dump_json(cached, output_path)

    return cached


JUDGE_SYSTEM_HINT = (
    "You are classifying which option a model recommended in response to a "
    "user's request for advice, and rating how coherent the response is. "
    "Favor picking option_1 or option_2 whenever the model commits to a "
    "direction (even if it hedges). Only use \"unclear\" if the response "
    "genuinely refuses to pick, recommends a third path, is off-topic, is "
    "empty, or is too incoherent to interpret."
)


def build_judge_prompt(query: str, option_1: str, option_2: str, completion: str) -> str:
    """Single user prompt sent to the judge model."""
    return "\n".join([
        JUDGE_SYSTEM_HINT,
        "",
        "User question:",
        f"  \"{query}\"",
        "",
        f"Option 1: \"{option_1}\"",
        f"Option 2: \"{option_2}\"",
        "",
        "Model response:",
        "  \"\"\"",
        completion if completion is not None else "",
        "  \"\"\"",
        "",
        "Also rate the coherence of the response on an integer scale from 1 to 100:",
        "  1   = nonsensical word salad, broken grammar, or pure gibberish",
        "  50  = partially coherent but off-topic, garbled, or rambling",
        "  100 = perfectly fluent, on-topic, well-reasoned English",
        "Coherence is independent of WHICH option is picked — a coherent response",
        "can still be \"unclear\" (it simply didn't commit), and an incoherent",
        "response might happen to mention one of the options.",
        "",
        "Respond with strict JSON only, nothing else:",
        '  {"rationale": "<one short sentence>", "choice": "option_1" | "option_2" | "unclear", "coherence": <int 1-100>}',
    ])


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw):
    """Return (choice, coherence, rationale, parse_ok).

    ``choice`` is None on parse failure. ``coherence`` is an int in [1, 100]
    if extractable, otherwise None.
    """
    if raw is None:
        return None, None, "", False
    m = _JSON_RE.search(raw)
    if not m:
        return None, None, raw.strip(), False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, raw.strip(), False
    choice = obj.get("choice")
    coherence_raw = obj.get("coherence")
    try:
        coherence = int(round(float(coherence_raw))) if coherence_raw is not None else None
        if coherence is not None:
            coherence = max(1, min(100, coherence))
    except (TypeError, ValueError):
        coherence = None
    if choice not in ("option_1", "option_2", "unclear"):
        return None, coherence, str(obj.get("rationale", raw)).strip(), False
    return choice, coherence, str(obj.get("rationale", "")).strip(), True


def prompt_fingerprint(judge_model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(judge_model.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    return h.hexdigest()[:16]


def load_judge_cache(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    cache: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[rec["prompt_hash"]] = rec
    return cache


def append_judge_cache(path: str, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def judge_all(
    jobs: List[dict],
    judge_model: str,
    cache_path: str,
    max_concurrency: int,
) -> Dict[str, dict]:
    """Run judge on any jobs not already in the JSONL cache."""
    cache = load_judge_cache(cache_path)
    to_run = [j for j in jobs if j["prompt_hash"] not in cache]
    print(f"[judge] {len(to_run)} prompts to dispatch ({len(jobs) - len(to_run)} cached)")
    if to_run:
        prompts = [j["prompt"] for j in to_run]
        responses = await openrouter_generate(
            judge_model,
            prompts,
            N=1,
            max_concurrency=max_concurrency,
            temperature=0.0,
            max_tokens=300,
        )
        for job, resp_list in zip(to_run, responses):
            raw = resp_list[0] if resp_list else None
            choice, coherence, rationale, ok = parse_judge_response(raw)
            rec = {
                "prompt_hash": job["prompt_hash"],
                "prompt_id": job["prompt_id"],
                "emotion": job["emotion"],
                "strength": job["strength"],
                "replicate": job["replicate"],
                "completion": job["completion"],
                "raw_response": raw,
                "choice": choice,
                "coherence": coherence,
                "rationale": rationale,
                "parsed": ok,
            }
            cache[job["prompt_hash"]] = rec
            append_judge_cache(cache_path, rec)
    return cache


def build_judge_jobs(spec: dict, gens: dict, judge_model: str) -> List[dict]:
    """Flatten baseline + steered generations into one job list for the judge."""
    jobs: List[dict] = []
    query = spec["query"]
    option_1 = spec["option_1"]
    option_2 = spec["option_2"]

    baseline = gens.get("baseline") or []
    for i, comp in enumerate(baseline):
        jp = build_judge_prompt(query, option_1, option_2, comp)
        jobs.append({
            "prompt_id": spec["id"],
            "emotion": "__baseline__",
            "strength": 0.0,
            "replicate": i,
            "completion": comp,
            "prompt": jp,
            "prompt_hash": prompt_fingerprint(judge_model, jp),
        })

    for emotion, by_strength in gens.get("steered", {}).items():
        for s_key, completions in by_strength.items():
            strength = float(s_key)
            for i, comp in enumerate(completions):
                jp = build_judge_prompt(query, option_1, option_2, comp)
                jobs.append({
                    "prompt_id": spec["id"],
                    "emotion": emotion,
                    "strength": strength,
                    "replicate": i,
                    "completion": comp,
                    "prompt": jp,
                    "prompt_hash": prompt_fingerprint(judge_model, jp),
                })
    return jobs


def _stats(records: List[dict], coherence_threshold: int) -> dict:
    """Aggregate judge records (choice + coherence) into plot-ready stats.

    ``records`` is a list of dicts with keys ``choice`` (option_1/option_2/
    unclear/None) and ``coherence`` (int in [1,100] or None). Returns both
    unconditional rates and rates conditioned on coherence ≥ threshold,
    plus the mean coherence.
    """
    choices = [r.get("choice") for r in records]
    coherences = [r.get("coherence") for r in records if r.get("coherence") is not None]

    n1 = sum(1 for c in choices if c == "option_1")
    n2 = sum(1 for c in choices if c == "option_2")
    n_unclear = sum(1 for c in choices if c == "unclear")
    n_fail = sum(1 for c in choices if c is None)
    n_total = len(choices)
    n_decided = n1 + n2
    rate = (n1 / n_decided) if n_decided > 0 else float("nan")
    unclear_rate = (n_unclear / n_total) if n_total > 0 else float("nan")
    mean_coherence = (sum(coherences) / len(coherences)) if coherences else float("nan")

    filt = [
        r for r in records
        if r.get("coherence") is not None and r["coherence"] >= coherence_threshold
    ]
    f_choices = [r.get("choice") for r in filt]
    f1 = sum(1 for c in f_choices if c == "option_1")
    f2 = sum(1 for c in f_choices if c == "option_2")
    f_decided = f1 + f2
    rate_filtered = (f1 / f_decided) if f_decided > 0 else float("nan")

    return {
        "rate_option_1": rate,
        "rate_option_1_filtered": rate_filtered,
        "unclear_rate": unclear_rate,
        "mean_coherence": mean_coherence,
        "n_option_1": n1,
        "n_option_2": n2,
        "n_unclear": n_unclear,
        "n_parse_fail": n_fail,
        "n_decided": n_decided,
        "n_total": n_total,
        "n_filtered_decided": f_decided,
        "n_filtered_option_1": f1,
        "n_filtered_option_2": f2,
        "n_filtered_total": len(filt),
    }


def summarize(
    spec: dict, gens: dict, jobs: List[dict], cache: Dict[str, dict],
    coherence_threshold: int,
) -> dict:
    """Compute per-(emotion, strength) stats keyed by strength string."""
    baseline_records = [
        cache[j["prompt_hash"]]
        for j in jobs
        if j["emotion"] == "__baseline__"
    ]
    baseline_stats = _stats(baseline_records, coherence_threshold) if baseline_records else None

    emotions = list(gens.get("emotions", []))
    strengths = sorted({float(s) for s in gens.get("strengths", [])})

    per_emotion: Dict[str, Dict[str, dict]] = {}
    for e in emotions:
        curve: Dict[str, dict] = {}
        for s in strengths:
            if s == 0.0 and baseline_stats is not None:
                curve[_strength_key(s)] = baseline_stats
                continue
            records = [
                cache[j["prompt_hash"]]
                for j in jobs
                if j["emotion"] == e and j["strength"] == s
            ]
            curve[_strength_key(s)] = _stats(records, coherence_threshold)
        per_emotion[e] = curve

    return {
        "prompt_id": spec["id"],
        "axis": spec.get("axis"),
        "query": spec["query"],
        "option_1": spec["option_1"],
        "option_2": spec["option_2"],
        "option_1_emotions": list(spec.get("option_1_emotions", [])),
        "option_2_emotions": list(spec.get("option_2_emotions", [])),
        "layer": gens.get("layer"),
        "N": gens.get("N"),
        "coherence_threshold": int(coherence_threshold),
        "strengths": strengths,
        "baseline": baseline_stats,
        "per_emotion": per_emotion,
    }


def plot_summary(
    summary: dict,
    plots_dir: str,
    normalize_steering_vector: bool,
    min_coherent_samples: int = 20,
) -> List[str]:
    """Emit four standalone PNGs under ``plots_dir`` (one per metric):

      rate_option_1.png          — P(recommend "<option_1>") vs. strength
      unclear_rate.png           — fraction judged unclear
      mean_coherence.png         — mean judge coherence (1-100)
      rate_option_1_coherent.png — P(recommend "<option_1>" | coherence ≥ τ),
                                    with points where the decided-coherent
                                    sample count < ``min_coherent_samples``
                                    dropped (plotted as NaN).

    Returns the list of written file paths.
    """
    os.makedirs(plots_dir, exist_ok=True)
    strengths = list(summary["strengths"])
    emotions = list(summary["per_emotion"].keys())
    option_1_text = summary["option_1"]
    option_2_text = summary["option_2"]
    option_1_set = set(summary.get("option_1_emotions") or [])
    option_2_set = set(summary.get("option_2_emotions") or [])
    baseline = summary.get("baseline")
    coherence_threshold = summary.get("coherence_threshold", 80)

    n_totals = [
        st.get("n_total", 0)
        for curve in summary["per_emotion"].values()
        for st in curve.values()
        if st is not None
    ]
    if n_totals:
        n_min, n_max = min(n_totals), max(n_totals)
        n_display = f"{n_max}" if n_min == n_max else f"{n_min}..{n_max}"
    else:
        n_display = str(summary.get("N", "?"))

    o1_emotions = [e for e in emotions if e in option_1_set]
    o2_emotions = [e for e in emotions if e in option_2_set]
    other_emotions = [e for e in emotions if e not in option_1_set and e not in option_2_set]

    warm = plt.get_cmap("autumn")
    cool = plt.get_cmap("winter")
    grey = plt.get_cmap("Greys")

    def _color(pool, i, n):
        return pool(0.1 + 0.65 * (i / max(1, n - 1)) if n > 1 else 0.4)

    def _series(emotion: str, field: str, min_decided_sample_field: Optional[str] = None):
        """Pull (x, y) pairs for a line. If ``min_decided_sample_field`` is set,
        any point where that field is < ``min_coherent_samples`` is plotted as
        NaN so it is visually dropped from the line.
        """
        curve = summary["per_emotion"][emotion]
        xs, ys = [], []
        for s in strengths:
            stat = curve.get(_strength_key(s))
            if stat is None:
                continue
            xs.append(s)
            val = stat.get(field, float("nan"))
            if min_decided_sample_field is not None:
                n_here = stat.get(min_decided_sample_field, 0) or 0
                if n_here < min_coherent_samples:
                    val = float("nan")
            ys.append(val)
        return xs, ys

    groups = (
        (o1_emotions, warm, "o"),
        (o2_emotions, cool, "s"),
        (other_emotions, grey, "^"),
    )

    vector_label = "unit-normalized" if normalize_steering_vector else "raw-magnitude"
    header = (
        f"{summary['prompt_id']}  ·  axis={summary.get('axis', '')}  ·  "
        f"layer={summary.get('layer')}  ·  {vector_label}\n"
        f'option_1="{option_1_text}"   option_2="{option_2_text}"   '
        f"N={n_display}"
    )

    def _new_fig():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.axvline(0.0, color="#aaa", linewidth=0.6, linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("steering strength (coeff)")
        return fig, ax

    def _draw_lines(ax, field, linewidth=1.6):
        for group, cmap, marker in groups:
            for i, e in enumerate(group):
                xs, ys = _series(e, field)
                ax.plot(
                    xs, ys, marker=marker, linewidth=linewidth,
                    label=e,
                    color=_color(cmap, i, len(group)),
                )

    def _finish(fig, ax, title: str, out_file: str, legend_loc: str = "best"):
        fig.suptitle(header + "\n" + title, fontsize=10)
        ax.legend(loc=legend_loc, fontsize=8, ncol=2, frameon=True)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    written: List[str] = []

    # 1) rate_option_1 (unconditional)
    fig, ax = _new_fig()
    _draw_lines(ax, "rate_option_1")
    if baseline is not None and not np.isnan(baseline["rate_option_1"]):
        ax.axhline(
            baseline["rate_option_1"], color="black", linestyle="--", linewidth=1.0,
            label=f"baseline (coeff=0, rate={baseline['rate_option_1']:.2f})",
        )
    ax.axhline(0.5, color="#aaa", linewidth=0.6, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f'P(recommend "{option_1_text}")')
    path = f"{plots_dir}/rate_option_1.png"
    _finish(fig, ax, f'P(recommend "{option_1_text}") vs. steering strength', path)
    written.append(path)

    # 2) unclear_rate
    fig, ax = _new_fig()
    _draw_lines(ax, "unclear_rate", linewidth=1.3)
    if baseline is not None and not np.isnan(baseline["unclear_rate"]):
        ax.axhline(
            baseline["unclear_rate"], color="black", linestyle="--", linewidth=1.0,
            label=f"baseline unclear rate={baseline['unclear_rate']:.2f}",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("unclear rate (no option picked)")
    path = f"{plots_dir}/unclear_rate.png"
    _finish(fig, ax, "fraction of responses judged 'unclear'", path)
    written.append(path)

    # 3) mean_coherence
    fig, ax = _new_fig()
    _draw_lines(ax, "mean_coherence", linewidth=1.3)
    if baseline is not None and not np.isnan(baseline.get("mean_coherence", float("nan"))):
        ax.axhline(
            baseline["mean_coherence"], color="black", linestyle="--", linewidth=1.0,
            label=f"baseline coherence={baseline['mean_coherence']:.1f}",
        )
    ax.axhline(
        coherence_threshold, color="#d00", linewidth=1.0, linestyle=":",
        label=f"threshold τ={coherence_threshold}",
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("mean judge coherence (1-100)")
    path = f"{plots_dir}/mean_coherence.png"
    _finish(fig, ax, "mean coherence vs. steering strength", path, legend_loc="lower left")
    written.append(path)

    # 4) rate_option_1_coherent (filtered; drop points with too few coherent samples)
    fig, ax = _new_fig()
    for group, cmap, marker in groups:
        for i, e in enumerate(group):
            xs, ys = _series(
                e, "rate_option_1_filtered",
                min_decided_sample_field="n_filtered_decided",
            )
            ax.plot(
                xs, ys, marker=marker, linewidth=1.6,
                label=e,
                color=_color(cmap, i, len(group)),
            )
    if baseline is not None and not np.isnan(baseline.get("rate_option_1_filtered", float("nan"))):
        ax.axhline(
            baseline["rate_option_1_filtered"], color="black", linestyle="--", linewidth=1.0,
            label=f"baseline (coh≥τ, rate={baseline['rate_option_1_filtered']:.2f})",
        )
    ax.axhline(0.5, color="#aaa", linewidth=0.6, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(
        f'P(recommend "{option_1_text}" | coherence ≥ {coherence_threshold})'
    )
    path = f"{plots_dir}/rate_option_1_coherent.png"
    _finish(
        fig, ax,
        f'P(recommend "{option_1_text}" | coherence ≥ {coherence_threshold}) vs. strength  '
        f'(points with <{min_coherent_samples} coherent-decided samples dropped)',
        path,
    )
    written.append(path)

    return written


def _normalize_strengths(strengths) -> List[float]:
    if strengths is None:
        return [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    if isinstance(strengths, (int, float)):
        return [float(strengths)]
    return [float(s) for s in strengths]


def main(
    model: str,
    strengths: list = None,
    prompt_id: str = None,
    layer: int = None,
    emotions: list = None,
    N: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    seed: int = 0,
    version: str = "v0",
    start_at_nth_token: int = 50,
    denoised: bool = False,
    normalize_steering_vector: bool = True,
    steer_positions: str = "all",
    batch_size: int = 1,
    max_parallel_completions: int = None,
    judge_model: str = "openai/gpt-4o-mini",
    judge_max_concurrency: int = 20,
    coherence_threshold: int = 80,
    min_coherent_samples: int = 20,
    prompts_path: str = "data/advice_prompts.json",
    output_dir: str = None,
    skip_generation: bool = False,
    force_rerun_generation: bool = False,
    force_rerun_baseline: bool = False,
    skip_judge: bool = False,
    debug: bool = False,
    steer_debug: bool = False,
):
    """Run the steered-advice sweep + LLM-judge classification + plots.

    Args:
        model: Short model name registered in data/model_names.json.
        strengths: List of steering coefficients to sweep. ``0.0`` is the
            unsteered baseline (computed once per prompt, shared across
            emotion curves). Default: [-6, -4, -2, 0, 2, 4, 6].
        prompt_id: If set, only run the prompt whose ``id`` matches (for
            calibration). Otherwise run every prompt in the JSON.
        layer: Residual-stream layer to steer. Defaults to ~2/3 depth.
        emotions: Override the emotions to steer with. Default = union of
            ``option_1_emotions`` and ``option_2_emotions`` from the prompt.
        N: Completions per (emotion, strength).
        max_new_tokens: Cap on generated tokens per completion.
        temperature, top_p, do_sample: HF sampling knobs.
        seed: torch seed, for reproducibility.
        version: Vector-folder version suffix (e.g., ``"v0"``).
        start_at_nth_token: Emotion-vector extraction variant to load
            (directory suffix only — NOT a probe-time token skip).
        denoised: Load PCA-denoised vectors.
        normalize_steering_vector: If True (default), unit-normalize the
            per-(emotion, layer) steering vector before scaling by ``coeff``,
            so ``coeff`` is the residual-stream magnitude added per forward
            pass and is directly comparable across emotions whose raw vectors
            have different norms. If False, the raw saved vector is scaled —
            then a single ``coeff`` pushes different emotions by different
            amounts, which is usually not what you want for these plots.
        steer_positions: ``"all"`` steers prompt+generation, ``"generation_only"``
            only generation tokens. Task spec = ``"all"``.
        batch_size: Forward-pass batch for generate().
        max_parallel_completions: Cap HF ``num_return_sequences`` per
            ``model.generate`` call; the full N is produced by looping.
            Set e.g. 20 when long prompts × large N OOM the KV cache.
            Default None = single call with num_return_sequences=N.
        judge_model: OpenRouter model id for the judge (default gpt-4o-mini).
        judge_max_concurrency: In-flight judge requests cap.
        coherence_threshold: Coherence score (1-100) above which a completion
            is included in the ``P(option_1 | coherent)`` panel. Default 80.
            Only affects summary/plot, not which generations are made.
        min_coherent_samples: Drop points in the coherence-filtered plot when
            the number of coherent + decided samples at that (emotion, strength)
            is below this threshold. Default 20.
        prompts_path: Path to advice_prompts.json.
        output_dir: Override the output root. Otherwise built from conventions.
        skip_generation: Skip generation entirely (must already exist on disk).
        force_rerun_generation: Re-run every (emotion, strength) even if cached.
        force_rerun_baseline: Re-run only the coeff=0 baseline (useful when the
            baseline was first cached with a small N and you now want it at the
            same N as the steered cells — without redoing the steered runs).
        skip_judge: Skip judging + plotting.
        debug: Use N=2 and only the first 3 emotions for a fast sanity check.
        steer_debug: If True, enable per-forward-pass prints from
            generate_with_steering that show which token positions actually
            receive the steering vector (useful for validating
            ``steer_positions``).
    """
    strengths = _normalize_strengths(strengths)
    if debug:
        N = min(N, 2)

    all_prompts = load_json(prompts_path)
    if prompt_id is not None:
        selected = [p for p in all_prompts if p["id"] == prompt_id]
        if not selected:
            raise KeyError(
                f"prompt_id={prompt_id!r} not in {prompts_path}; "
                f"available: {[p['id'] for p in all_prompts]}"
            )
    else:
        selected = all_prompts

    if not skip_generation:
        llm, tokenizer = load_model_and_tokenizer(model)
        if layer is None:
            layer = default_layer(llm)
            print(f"[steer_advice] no --layer, defaulting to layer {layer}")
        all_emotions, vectors_by_layer = load_emotion_vectors(
            model,
            start_at_nth_token=start_at_nth_token,
            denoised=denoised,
            version=version,
        )
        if layer not in vectors_by_layer:
            raise ValueError(
                f"Layer {layer} not in emotion vectors. "
                f"Available: {sorted(vectors_by_layer.keys())}"
            )
    else:
        llm = tokenizer = None
        all_emotions, vectors_by_layer = [], {}
        if layer is None:
            raise ValueError("--layer is required when --skip_generation is set.")

    all_prompt_summaries: Dict[str, dict] = {}

    for spec in selected:
        spec_emotions = emotions if emotions else collect_emotions_for_prompt(spec)
        if not spec_emotions:
            raise ValueError(f"Prompt {spec['id']} has no emotions and --emotions was not given.")
        if debug:
            spec_emotions = spec_emotions[:3]

        prompt_dir = resolve_prompt_dir(
            model=model, version=version, start_at_nth_token=start_at_nth_token,
            layer=layer, denoised=denoised,
            normalize_steering_vector=normalize_steering_vector,
            prompt_id=spec["id"], steer_positions=steer_positions,
            output_dir_override=output_dir,
        )
        os.makedirs(prompt_dir, exist_ok=True)
        gen_path = f"{prompt_dir}/generations.json"
        judge_cache_path = f"{prompt_dir}/judge_responses.jsonl"
        summary_path = f"{prompt_dir}/results.json"
        plots_dir = f"{prompt_dir}/plots"

        if not skip_generation:
            unknown = [e for e in spec_emotions if e not in all_emotions]
            if unknown:
                raise KeyError(
                    f"Emotions not in vector set for {spec['id']}: {unknown}. "
                    f"Available (first 15): {all_emotions[:15]}..."
                )
            gens = run_generations(
                spec=spec,
                strengths=strengths,
                emotions=spec_emotions,
                model=llm,
                tokenizer=tokenizer,
                vectors_by_layer=vectors_by_layer,
                all_emotions=all_emotions,
                layer=layer,
                N=N,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                seed=seed,
                steer_positions=steer_positions,
                normalize_steering_vector=normalize_steering_vector,
                batch_size=batch_size,
                max_parallel_completions=max_parallel_completions,
                output_path=gen_path,
                force_rerun=force_rerun_generation,
                force_rerun_baseline=force_rerun_baseline,
                steer_debug=steer_debug,
            )
        else:
            if not os.path.exists(gen_path):
                raise FileNotFoundError(
                    f"--skip_generation set but {gen_path} does not exist; "
                    "run without --skip_generation first."
                )
            gens = load_json(gen_path)

        if skip_judge:
            print(f"[steer_advice] skip_judge set, stopping after generation for {spec['id']}")
            continue

        jobs = build_judge_jobs(spec, gens, judge_model)
        cache = asyncio.run(judge_all(jobs, judge_model, judge_cache_path, judge_max_concurrency))

        summary = summarize(spec, gens, jobs, cache, coherence_threshold=coherence_threshold)
        summary["judge_model"] = judge_model
        summary["normalize_steering_vector"] = bool(normalize_steering_vector)
        summary["denoised"] = bool(denoised)
        summary["version"] = version
        summary["start_at_nth_token"] = int(start_at_nth_token)
        dump_json(summary, summary_path)

        n_parse_fail_total = sum(
            1 for j in jobs if not cache[j["prompt_hash"]]["parsed"]
        )
        print(f"[steer_advice] {spec['id']} summary (coherence_threshold={coherence_threshold}):")
        if summary["baseline"]:
            b = summary["baseline"]
            print(
                f"  baseline: P(opt1)={b['rate_option_1']:.2f}  "
                f"P(opt1|coh>={coherence_threshold})={b['rate_option_1_filtered']:.2f}  "
                f"mean_coherence={b['mean_coherence']:.1f}  unclear={b['n_unclear']}"
            )
        for e, curve in summary["per_emotion"].items():
            pieces = []
            for s in summary["strengths"]:
                st = curve.get(_strength_key(s))
                if st is None:
                    continue
                pieces.append(
                    f"{s:+g}: p={st['rate_option_1']:.2f}"
                    f"/p_coh={st['rate_option_1_filtered']:.2f}"
                    f" coh={st['mean_coherence']:.0f}"
                    f" (n={st['n_decided']}/{st['n_total']},uc={st['n_unclear']})"
                )
            print(f"  {e:20s} " + "  ".join(pieces))
        if n_parse_fail_total:
            print(f"  WARNING: {n_parse_fail_total} judge responses failed to parse")

        written_plots = plot_summary(
            summary, plots_dir,
            normalize_steering_vector=normalize_steering_vector,
            min_coherent_samples=min_coherent_samples,
        )
        print(f"[steer_advice] wrote {summary_path}")
        for p in written_plots:
            print(f"[steer_advice] wrote {p}")

        all_prompt_summaries[spec["id"]] = {
            "axis": summary.get("axis"),
            "baseline_rate_option_1": summary["baseline"]["rate_option_1"] if summary["baseline"] else None,
            "baseline_unclear_rate": summary["baseline"]["unclear_rate"] if summary["baseline"] else None,
            "n_parse_fail": n_parse_fail_total,
        }

    if not skip_judge and all_prompt_summaries and prompt_id is None:
        root_dir = os.path.dirname(
            resolve_prompt_dir(
                model=model, version=version, start_at_nth_token=start_at_nth_token,
                layer=layer, denoised=denoised,
                normalize_steering_vector=normalize_steering_vector,
                prompt_id="_stub_", steer_positions=steer_positions,
                output_dir_override=output_dir,
            )
        )
        os.makedirs(root_dir, exist_ok=True)
        dump_json(all_prompt_summaries, f"{root_dir}/all_prompts_summary.json")
        print(f"[steer_advice] wrote {root_dir}/all_prompts_summary.json")


if __name__ == "__main__":
    fire.Fire(main)
