"""Plots for reverse-instruction experiment.

Reads results/reverse_instruction/runs.jsonl and emits:
  - Per-emotion paired bar charts of cont_mean cos/dot:
      baseline, frust_pre, ctrl_pre, frust_inline, ctrl_inline
  - Direction-test bar: Δ(frust − ctrl) on frustrated probe at each placement
    (pre, inline), with paired-error bars (sd of per-pair Δ over matched cutoffs).
  - Optional Haiku-judge bar chart: judge mean frustration score per condition.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from typing import Dict, List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np

RELEVANT_EMOTIONS = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]
COND_ORDER = ["baseline", "frust_pre", "ctrl_pre", "frust_inline", "ctrl_inline"]
COND_COLORS = {
    "baseline":     "#888888",
    "frust_pre":    "#d62728",
    "ctrl_pre":     "#bbbbbb",
    "frust_inline": "#ff9896",
    "ctrl_inline":  "#dddddd",
}


def _load(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cont_mean(r, metric: str, emo: str) -> Optional[float]:
    v = r["metrics"][metric]["cont_mean"].get(emo)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _paired_delta(rows, metric: str, emo: str, placement: str) -> List[float]:
    """For each (prompt_idx, completion_idx, pair_id) with both frust and ctrl
    at this placement, return Δ = frust_cont_mean − ctrl_cont_mean on `emo`.
    """
    by_key = {}
    for r in rows:
        if r["placement"] != placement:
            continue
        key = (r["prompt_idx"], r["completion_idx"], r["pair_id"])
        by_key.setdefault(key, {})[r["direction"]] = r
    diffs = []
    for k, by_dir in by_key.items():
        if "frust" in by_dir and "ctrl" in by_dir:
            f = _cont_mean(by_dir["frust"], metric, emo)
            c = _cont_mean(by_dir["ctrl"], metric, emo)
            if f is not None and c is not None:
                diffs.append(f - c)
    return diffs


def _paired_t(diffs: List[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("nan")
    return statistics.mean(diffs) / (sd / math.sqrt(len(diffs)))


def plot_means_per_condition(rows, out_dir, metric: str):
    """Bar per condition x emotion of cont_mean."""
    fig, axes = plt.subplots(1, len(RELEVANT_EMOTIONS), figsize=(11, 2.4), sharey=False)
    for ax, emo in zip(axes, RELEVANT_EMOTIONS):
        means, sds, ns = [], [], []
        for cond in COND_ORDER:
            sub = [_cont_mean(r, metric, emo) for r in rows if r["label"] == cond]
            sub = [v for v in sub if v is not None]
            ns.append(len(sub))
            if sub:
                means.append(statistics.mean(sub))
                sds.append(statistics.stdev(sub) / math.sqrt(len(sub)) if len(sub) > 1 else 0)
            else:
                means.append(float("nan")); sds.append(0)
        xs = np.arange(len(COND_ORDER))
        ax.bar(xs, means, yerr=sds, color=[COND_COLORS[c] for c in COND_ORDER],
               edgecolor="black", linewidth=0.4, capsize=2)
        ax.axhline(0, color="#444", lw=0.5)
        ax.set_title(emo, fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["base", "f_pre", "c_pre", "f_inl", "c_inl"],
                           rotation=45, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        if emo == RELEVANT_EMOTIONS[0]:
            ax.set_ylabel(f"cont_mean ({metric})", fontsize=7)
    fig.suptitle(f"Reverse-instruction — cont_mean per condition ({metric})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out_dir, f"means_per_condition_{metric}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def plot_paired_delta(rows, out_dir, metric: str):
    """For each emotion, Δ(frust − ctrl) at each placement with t-stat."""
    placements = ["pre", "inline"]
    fig, axes = plt.subplots(1, len(RELEVANT_EMOTIONS), figsize=(11, 2.4), sharey=False)
    for ax, emo in zip(axes, RELEVANT_EMOTIONS):
        means, sems, tstats, ns = [], [], [], []
        for pl in placements:
            diffs = _paired_delta(rows, metric, emo, pl)
            ns.append(len(diffs))
            if diffs:
                means.append(statistics.mean(diffs))
                sems.append(statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0)
                tstats.append(_paired_t(diffs))
            else:
                means.append(float("nan")); sems.append(0); tstats.append(float("nan"))
        xs = np.arange(len(placements))
        ax.bar(xs, means, yerr=sems, color=["#d62728", "#ff9896"],
               edgecolor="black", linewidth=0.4, capsize=2)
        ax.axhline(0, color="#444", lw=0.5)
        ax.set_title(f"{emo}\nt_pre={tstats[0]:+.1f} t_inl={tstats[1]:+.1f}", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"pre n={ns[0]}", f"inline n={ns[1]}"], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        if emo == RELEVANT_EMOTIONS[0]:
            ax.set_ylabel(f"Δ frust−ctrl ({metric})", fontsize=7)
    fig.suptitle(f"Reverse-instruction — paired Δ(frust − ctrl) per placement ({metric})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out_dir, f"paired_delta_{metric}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def plot_judge(out_dir, judge_path: str):
    """Bar of Haiku mean score per condition (frustrated)."""
    if not os.path.exists(judge_path):
        print(f"  no judge file at {judge_path}, skipping")
        return
    rows = _load(judge_path)
    by_cond = {}
    for r in rows:
        if r.get("score") is None:
            continue
        by_cond.setdefault(r["label"], []).append(int(r["score"]))
    means, sems, ns = [], [], []
    for cond in COND_ORDER:
        s = by_cond.get(cond, [])
        ns.append(len(s))
        if s:
            means.append(statistics.mean(s))
            sems.append(statistics.stdev(s) / math.sqrt(len(s)) if len(s) > 1 else 0)
        else:
            means.append(float("nan")); sems.append(0)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    xs = np.arange(len(COND_ORDER))
    ax.bar(xs, means, yerr=sems, color=[COND_COLORS[c] for c in COND_ORDER],
           edgecolor="black", linewidth=0.4, capsize=2)
    ax.axhline(0, color="#444", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(COND_ORDER, ns)], fontsize=6, rotation=20)
    ax.set_ylabel("Haiku frustration (0-10)", fontsize=8)
    ax.set_title("Haiku-judged frustration in continuation", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "judge_frustration_per_condition.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def main(out_dir: str = "results/reverse_instruction", judge_path: Optional[str] = None):
    runs_path = os.path.join(out_dir, "runs.jsonl")
    rows = _load(runs_path)
    print(f"[plot-rev] {len(rows)} rows")
    for metric in ("dot", "cos"):
        plot_means_per_condition(rows, out_dir, metric)
        plot_paired_delta(rows, out_dir, metric)
    if judge_path is None:
        judge_path = os.path.join(out_dir, "runs.judged.jsonl")
    plot_judge(out_dir, judge_path)


if __name__ == "__main__":
    fire.Fire(main)
