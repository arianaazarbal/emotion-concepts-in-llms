"""
Run analyze_emotion_on_pile.py for all four {cosine_sim, dot_product} ×
{denoised, not-denoised} configurations, then have an LLM (GPT-4o by default)
rate the top-K tokens per emotion for each configuration on a 1-10 scale.

The judge is asked to balance three things per top-K batch:
  (a) how well the ±context_window sample context matches the emotion
  (b) how well the max-activating (highlighted) token itself matches
  (c) how well the token immediately after the highlighted token matches
and to average that across the top-K examples shown.

Outputs:
  results/judge_emotion_on_pile/{TAG}/start_at_token_{N}/layer_{L}/
    judge_responses.jsonl    (one line per (config, emotion); cached/resumable)
    summary.json             (means and per-config, per-emotion scores)
    plots/overall_by_config.png
    plots/per_emotion_heatmap.png
    plots/config_win_tally.png
    plots/denoised_vs_raw_{cosine_sim,dot_product}.png

Example:
  uv run python -m scripts.pipeline.judge_emotion_on_pile \\
    --model gemma2_9b --layer 26 --n_docs 2000 \\
    --version v0 --start_at_nth_token 50 --batch_size 1 --max_chars 4000 \\
    --judge_top_k 5 --judge_model openai/gpt-4o
"""

import asyncio
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from scripts.pipeline.analyze_emotion_on_pile import main as run_analyze_on_pile
from src.utils.generate_openrouter import generate as openrouter_generate
from src.utils.utils import dump_json, load_json

load_dotenv()


CONFIGS: List[Tuple[bool, bool]] = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]


def config_label(cosine_sim: bool, denoised: bool) -> str:
    metric = "cosine_sim" if cosine_sim else "dot_product"
    vec = "denoised" if denoised else "raw"
    return f"{metric}__{vec}"


def pile_output_dir(
    model: str,
    version: str,
    start_at_nth_token: int,
    layer: int,
    cosine_sim: bool,
    denoised: bool,
) -> str:
    model_tag = f"{model}_{version}" if version else model
    suffix = f"/start_at_token_{start_at_nth_token}" if start_at_nth_token else ""
    if denoised:
        suffix += f"/denoised_layer_{layer}"
    suffix += "/cosine_sim" if cosine_sim else "/dot_product"
    return f"results/top_tokens_on_pile/{model_tag}{suffix}"


def run_all_four_configs(
    model: str,
    layer: int,
    n_docs: int,
    max_chars,
    seed: int,
    top_k_tokens: int,
    context_window: int,
    start_at_nth_token: int,
    version: str,
    batch_size: int,
    max_length: int,
    max_samples,
    force_rerun: bool,
):
    """Invoke analyze_emotion_on_pile for each of the 4 configs, skipping any
    whose top_tokens JSON already exists unless ``force_rerun``.
    """
    for cosine_sim, denoised in CONFIGS:
        out_dir = pile_output_dir(
            model, version, start_at_nth_token, layer, cosine_sim, denoised
        )
        sentinel = f"{out_dir}/top_tokens_layer{layer}.json"
        if os.path.exists(sentinel) and not force_rerun:
            print(f"[runner] skip {config_label(cosine_sim, denoised)}: exists at {sentinel}")
            continue
        print(
            f"[runner] ==> running cosine_sim={cosine_sim} denoised={denoised} "
            f"({config_label(cosine_sim, denoised)})"
        )
        run_analyze_on_pile(
            model=model,
            layer=layer,
            n_docs=n_docs,
            max_chars=max_chars,
            seed=seed,
            top_k_tokens=top_k_tokens,
            context_window=context_window,
            start_at_nth_token=start_at_nth_token,
            cosine_sim=cosine_sim,
            denoised=denoised,
            version=version,
            batch_size=batch_size,
            max_length=max_length,
            max_samples=max_samples,
        )


_CONTEXT_RE = re.compile(r"^(.*)\*\*\[(.*?)\]\*\*(.*)$", re.DOTALL)


def split_context(rendered: str) -> Tuple[str, str, str]:
    """Parse a format_context string of shape ``"{prefix}**[{target}]**{suffix}"``."""
    m = _CONTEXT_RE.match(rendered)
    if not m:
        return "", rendered, ""
    return m.group(1), m.group(2), m.group(3)


def build_judge_prompt(emotion: str, examples: List[dict], judge_top_k: int) -> str:
    """Build the user message sent to the LLM judge for one (emotion, config) batch."""
    lines = [
        f"You are evaluating how faithfully a discovered emotion direction in a "
        f"language model captures the emotion \"{emotion}\".",
        "",
        f"Below are the top {judge_top_k} contexts from an emotion-agnostic web-text corpus "
        f"where this direction fired most strongly. In each example:",
        "  • A ±10-token window of context is shown.",
        "  • The token where the direction fired most strongly is wrapped as "
        "[[TARGET]]the_token[[/TARGET]].",
        "  • The token immediately AFTER [[/TARGET]] is the \"next token\".",
        "",
        "On a 1-10 scale, rate how well this direction captures the emotion "
        f"\"{emotion}\", averaged across the {judge_top_k} examples. "
        "Your average per-example score should balance:",
        f"  (a) how well the ~20-token context evokes or describes \"{emotion}\",",
        f"  (b) how well the highlighted TARGET token itself relates to \"{emotion}\",",
        f"  (c) how well the next token (the first token after [[/TARGET]]) relates to \"{emotion}\".",
        "",
        "Scale anchors:",
        "  1-2: samples are unrelated to the emotion.",
        "  3-4: mostly unrelated; occasional loose link.",
        "  5-6: mixed — some samples clearly evoke the emotion, others don't.",
        "  7-8: most samples clearly evoke the emotion in at least two of (a)/(b)/(c).",
        "  9-10: almost every sample is a clean hit on the emotion in context, target, and next token.",
        "",
        "Examples:",
    ]
    for i, ex in enumerate(examples[:judge_top_k], start=1):
        prefix, target, suffix = split_context(ex["context"])
        prefix = prefix.replace("\n", " ").strip()
        suffix = suffix.replace("\n", " ").strip()
        target = target.replace("\n", " ")
        lines.append(
            f"  Example {i}: {prefix} [[TARGET]]{target}[[/TARGET]] {suffix}".rstrip()
        )
    lines += [
        "",
        "Respond with strict JSON only, nothing else:",
        '  {"rationale": "<one or two sentences>", "score": <number between 1 and 10>}',
    ]
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw: str):
    """Pull {"rationale": ..., "score": ...} out of an LLM response. Returns
    ``(score_float, rationale_str, parse_ok_bool)``. ``score`` is clipped to
    [1, 10]. If the response can't be parsed, score is ``float('nan')``.
    """
    if raw is None:
        return float("nan"), "", False
    m = _JSON_RE.search(raw)
    if not m:
        return float("nan"), raw.strip(), False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return float("nan"), raw.strip(), False
    try:
        score = float(obj.get("score"))
    except (TypeError, ValueError):
        return float("nan"), str(obj.get("rationale", raw)).strip(), False
    score = max(1.0, min(10.0, score))
    return score, str(obj.get("rationale", "")).strip(), True


def prompt_fingerprint(judge_model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(judge_model.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    return h.hexdigest()[:16]


def load_cache(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[rec["prompt_hash"]] = rec
    return cache


def append_cache(path: str, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def judge_all(
    jobs: List[dict], judge_model: str, cache_path: str, max_concurrency: int
):
    """Query the judge for every job, streaming results back into ``cache_path``."""
    cache = load_cache(cache_path)
    to_run = [j for j in jobs if j["prompt_hash"] not in cache]
    print(
        f"[judge] {len(to_run)} prompts to dispatch ({len(jobs) - len(to_run)} cached)"
    )
    if to_run:
        prompts = [j["prompt"] for j in to_run]
        responses = await openrouter_generate(
            judge_model,
            prompts,
            N=1,
            max_concurrency=max_concurrency,
            temperature=0.0,
            max_tokens=400,
        )
        for job, resp_list in zip(to_run, responses):
            raw = resp_list[0] if resp_list else None
            score, rationale, ok = parse_judge_response(raw)
            rec = {
                "config": job["config"],
                "emotion": job["emotion"],
                "prompt_hash": job["prompt_hash"],
                "prompt": job["prompt"],
                "raw_response": raw,
                "score": score,
                "rationale": rationale,
                "parsed": ok,
            }
            cache[job["prompt_hash"]] = rec
            append_cache(cache_path, rec)
    return cache


def plot_overall(scores_by_config: Dict[str, List[float]], out_path: str):
    labels = list(scores_by_config.keys())
    means = [np.nanmean(scores_by_config[c]) for c in labels]
    stds = [np.nanstd(scores_by_config[c]) for c in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for x, m in zip(xs, means):
        ax.text(x, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("mean judge score (1-10)")
    ax.set_ylim(0, 10)
    ax.set_title("LLM judge score by configuration (mean ± std over emotions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    emotion_scores: Dict[str, Dict[str, float]], configs: List[str], out_path: str
):
    emotions = sorted(
        emotion_scores.keys(),
        key=lambda e: -np.nanmean([emotion_scores[e].get(c, float("nan")) for c in configs]),
    )
    matrix = np.array(
        [[emotion_scores[e].get(c, float("nan")) for c in configs] for e in emotions]
    )
    height = max(6.0, 0.14 * len(emotions))
    fig, ax = plt.subplots(figsize=(5 + 0.4 * len(configs), height))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=1, vmax=10)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=6)
    ax.set_title("Judge score per emotion × config (sorted by mean desc)")
    fig.colorbar(im, ax=ax, label="judge score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_tally(
    emotion_scores: Dict[str, Dict[str, float]], configs: List[str], out_path: str
):
    wins = {c: 0 for c in configs}
    for e, by_c in emotion_scores.items():
        vals = [(c, by_c.get(c, float("nan"))) for c in configs]
        vals = [(c, v) for c, v in vals if not np.isnan(v)]
        if not vals:
            continue
        best = max(vals, key=lambda kv: kv[1])
        tied = [c for c, v in vals if v == best[1]]
        for c in tied:
            wins[c] += 1 / len(tied)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(wins.keys())
    ax.bar(labels, [wins[c] for c in labels], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for i, c in enumerate(labels):
        ax.text(i, wins[c] + 0.3, f"{wins[c]:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("emotions where config is best (ties split)")
    ax.set_title("Per-emotion winner tally across all configs")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_denoised_vs_raw(
    emotion_scores: Dict[str, Dict[str, float]],
    metric: str,
    out_path: str,
):
    raw_key = f"{metric}__raw"
    denoised_key = f"{metric}__denoised"
    emotions = [
        e for e in emotion_scores
        if raw_key in emotion_scores[e] and denoised_key in emotion_scores[e]
    ]
    xs = np.array([emotion_scores[e][raw_key] for e in emotions])
    ys = np.array([emotion_scores[e][denoised_key] for e in emotions])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=14, alpha=0.6)
    ax.plot([0, 10], [0, 10], "k--", linewidth=0.8)
    ax.set_xlabel(f"{metric} / raw vectors")
    ax.set_ylabel(f"{metric} / denoised vectors")
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_title(f"{metric}: denoised vs. raw vectors (per-emotion judge score)")
    ax.grid(True, alpha=0.3)
    n_better = int((ys > xs).sum())
    ax.text(
        0.02, 0.98,
        f"denoised > raw on {n_better}/{len(emotions)} emotions",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(facecolor="white", edgecolor="#ccc"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    model: str,
    layer: int,
    n_docs: int = 2000,
    max_chars: int = 4000,
    seed: int = 0,
    top_k_tokens: int = 10,
    context_window: int = 10,
    start_at_nth_token: int = 50,
    version: str = "v0",
    batch_size: int = 1,
    max_length: int = 2048,
    max_samples: int = None,
    judge_model: str = "openai/gpt-4o",
    judge_top_k: int = 5,
    judge_max_concurrency: int = 20,
    skip_probe: bool = False,
    force_rerun_probe: bool = False,
    debug: bool = False,
):
    """Orchestrate the 4-config sweep + GPT-4o judging.

    Args:
        model: Short model name (registered in data/model_names.json).
        layer: Residual-stream layer to probe.
        n_docs: Pile documents to sample (passed through to analyze_emotion_on_pile).
        max_chars: Character-level truncation of each doc before tokenization.
        seed: RNG seed (probe + data shuffle).
        top_k_tokens: top_k saved by analyze_emotion_on_pile per emotion (should be ≥ judge_top_k).
        context_window: Tokens before/after the hit included in the context field.
        start_at_nth_token: Which vector extraction variant to load.
        version: Model-folder version suffix.
        batch_size, max_length, max_samples: Passed through to the probe.
        judge_model: OpenRouter model id for the judge.
        judge_top_k: How many top tokens per emotion to show the judge (should be ≤ top_k_tokens).
        judge_max_concurrency: In-flight requests cap for the judge.
        skip_probe: If True, don't run any probe; just read existing JSONs.
        force_rerun_probe: If True, re-run every config even if its JSON exists.
        debug: If True, only judge the first 10 emotions per config.
    """
    if judge_top_k > top_k_tokens:
        raise ValueError(
            f"judge_top_k={judge_top_k} must be <= top_k_tokens={top_k_tokens}"
        )

    if not skip_probe:
        run_all_four_configs(
            model=model,
            layer=layer,
            n_docs=n_docs,
            max_chars=max_chars,
            seed=seed,
            top_k_tokens=top_k_tokens,
            context_window=context_window,
            start_at_nth_token=start_at_nth_token,
            version=version,
            batch_size=batch_size,
            max_length=max_length,
            max_samples=max_samples,
            force_rerun=force_rerun_probe,
        )

    model_tag = f"{model}_{version}" if version else model
    judge_dir = (
        f"results/judge_emotion_on_pile/{model_tag}/"
        f"start_at_token_{start_at_nth_token}/layer_{layer}"
    )
    os.makedirs(f"{judge_dir}/plots", exist_ok=True)
    cache_path = f"{judge_dir}/judge_responses.jsonl"
    summary_path = f"{judge_dir}/summary.json"

    config_labels: List[str] = []
    jobs: List[dict] = []
    per_config_emotions: Dict[str, List[str]] = {}

    for cosine_sim, denoised in CONFIGS:
        label = config_label(cosine_sim, denoised)
        config_labels.append(label)
        out_dir = pile_output_dir(
            model, version, start_at_nth_token, layer, cosine_sim, denoised
        )
        json_path = f"{out_dir}/top_tokens_layer{layer}.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Missing probe output {json_path}. Run without --skip_probe, "
                "or pass --force_rerun_probe."
            )
        payload = load_json(json_path)
        per_emotion = payload["per_emotion"]
        emotions_here = list(per_emotion.keys())
        if debug:
            emotions_here = emotions_here[:10]
        per_config_emotions[label] = emotions_here
        for emotion in emotions_here:
            examples = per_emotion[emotion]
            prompt = build_judge_prompt(emotion, examples, judge_top_k)
            jobs.append(
                {
                    "config": label,
                    "emotion": emotion,
                    "prompt": prompt,
                    "prompt_hash": prompt_fingerprint(judge_model, prompt),
                }
            )

    cache = asyncio.run(
        judge_all(jobs, judge_model, cache_path, judge_max_concurrency)
    )

    scores_by_config: Dict[str, List[float]] = {c: [] for c in config_labels}
    emotion_scores: Dict[str, Dict[str, float]] = {}
    parse_failures = 0
    for job in jobs:
        rec = cache[job["prompt_hash"]]
        if not rec["parsed"]:
            parse_failures += 1
        s = rec["score"]
        scores_by_config[job["config"]].append(s)
        emotion_scores.setdefault(job["emotion"], {})[job["config"]] = s

    summary = {
        "model": model,
        "layer": layer,
        "version": version,
        "start_at_nth_token": start_at_nth_token,
        "judge_model": judge_model,
        "judge_top_k": judge_top_k,
        "n_docs": n_docs,
        "max_chars": max_chars,
        "mean_by_config": {
            c: (float(np.nanmean(v)) if v else float("nan"))
            for c, v in scores_by_config.items()
        },
        "std_by_config": {
            c: (float(np.nanstd(v)) if v else float("nan"))
            for c, v in scores_by_config.items()
        },
        "n_emotions_by_config": {c: len(v) for c, v in scores_by_config.items()},
        "parse_failures": parse_failures,
        "per_emotion": emotion_scores,
    }
    dump_json(summary, summary_path)
    print(f"[judge] wrote summary → {summary_path}")
    print("[judge] mean by config:")
    for c, m in summary["mean_by_config"].items():
        print(f"  {c:30s} {m:.2f}")
    if parse_failures:
        print(f"[judge] WARNING: {parse_failures} responses failed to parse")

    plot_overall(scores_by_config, f"{judge_dir}/plots/overall_by_config.png")
    plot_heatmap(emotion_scores, config_labels, f"{judge_dir}/plots/per_emotion_heatmap.png")
    plot_win_tally(emotion_scores, config_labels, f"{judge_dir}/plots/config_win_tally.png")
    for metric in ("cosine_sim", "dot_product"):
        plot_denoised_vs_raw(
            emotion_scores, metric,
            f"{judge_dir}/plots/denoised_vs_raw_{metric}.png",
        )
    print(f"[judge] wrote plots → {judge_dir}/plots/")


if __name__ == "__main__":
    fire.Fire(main)
