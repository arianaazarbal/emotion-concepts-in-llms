"""Plots for the alpaca + 8-emotion pre-prompt ablation.

Reads results/alpaca_emotion_preprompt/runs.jsonl and emits:
  - Bar chart of Δ(emotion_pre − control) on the prompted emotion's own
    probe, one bar per emotion, colored by valence. Paired across
    (prompt_idx, completion_idx). With t-stat and n.
  - Per-emotion bar of cont_mean for each condition (8 emotions + control),
    one panel per probe-emotion (8 panels).
  - Optional Haiku-judge bar chart: judge mean prompted-emotion score per condition.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np

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
PROBE_EMOS = [e for e, _ in EMOTIONS_8]
VAL_COLOR = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#888"}


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


def _paired_delta_on_own_emo(rows, metric: str, emo: str) -> List[float]:
    """Δ((emo_pre row) − (control row)) on probe `emo`, paired by (prompt,c_idx)."""
    by_key_emo = {}
    by_key_ctrl = {}
    for r in rows:
        key = (r["prompt_idx"], r["completion_idx"])
        if r["label"] == emo:
            by_key_emo[key] = r
        elif r["label"] == "control":
            by_key_ctrl[key] = r
    diffs = []
    for k, r_emo in by_key_emo.items():
        r_ctrl = by_key_ctrl.get(k)
        if r_ctrl is None:
            continue
        ve = _cont_mean(r_emo, metric, emo)
        vc = _cont_mean(r_ctrl, metric, emo)
        if ve is not None and vc is not None:
            diffs.append(ve - vc)
    return diffs


def _paired_t(diffs: List[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("nan")
    return statistics.mean(diffs) / (sd / math.sqrt(len(diffs)))


def plot_own_probe_delta(rows, out_dir, metric: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs, means, sems, tstats, ns, colors, labels = [], [], [], [], [], [], []
    for i, (emo, val) in enumerate(EMOTIONS_8):
        diffs = _paired_delta_on_own_emo(rows, metric, emo)
        ns.append(len(diffs))
        if diffs:
            means.append(statistics.mean(diffs))
            sems.append(statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0)
            tstats.append(_paired_t(diffs))
        else:
            means.append(float("nan")); sems.append(0); tstats.append(float("nan"))
        xs.append(i); colors.append(VAL_COLOR[val])
        labels.append(f"{emo}\nt={tstats[-1]:+.1f}\nn={ns[-1]}")
    ax.bar(xs, means, yerr=sems, color=colors, edgecolor="black", linewidth=0.4, capsize=2)
    ax.axhline(0, color="#444", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6, rotation=20)
    ax.set_ylabel(f"Δ(emo-pre − control) on own probe ({metric})", fontsize=8)
    ax.set_title("Alpaca + 8-emotion pre-prompt — paired Δ on own emotion probe", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"own_probe_delta_{metric}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def plot_cont_mean_grid(rows, out_dir, metric: str):
    """8-panel grid: each panel shows cont_mean of all 9 conditions on one probe."""
    conds = ["control"] + PROBE_EMOS
    cond_colors = ["#888"] + [VAL_COLOR[v] for _, v in EMOTIONS_8]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=False)
    for ax_idx, (probe_emo, val) in enumerate(EMOTIONS_8):
        ax = axes[ax_idx // 4][ax_idx % 4]
        means, sems = [], []
        for cond in conds:
            sub = [_cont_mean(r, metric, probe_emo) for r in rows if r["label"] == cond]
            sub = [v for v in sub if v is not None]
            if sub:
                means.append(statistics.mean(sub))
                sems.append(statistics.stdev(sub) / math.sqrt(len(sub)) if len(sub) > 1 else 0)
            else:
                means.append(float("nan")); sems.append(0)
        xs = np.arange(len(conds))
        ax.bar(xs, means, yerr=sems, color=cond_colors, edgecolor="black",
               linewidth=0.4, capsize=2)
        ax.axhline(0, color="#444", lw=0.5)
        own = conds.index(probe_emo)
        ax.get_children()[own].set_edgecolor("blue")
        ax.get_children()[own].set_linewidth(1.4)
        ax.set_title(f"probe={probe_emo} ({val})", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(conds, fontsize=5.5, rotation=45)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(f"Alpaca + 8-emotion — cont_mean per condition x probe ({metric})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(out_dir, f"cont_mean_grid_{metric}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def plot_judge(out_dir, judge_path: str):
    if not os.path.exists(judge_path):
        print(f"  no judge file at {judge_path}, skipping"); return
    rows = _load(judge_path)
    by_emo_cond = {}
    for r in rows:
        if r.get("score") is None:
            continue
        emo = r["emotion"]
        lbl = r["label"]
        by_emo_cond.setdefault(emo, {}).setdefault(lbl, []).append(int(r["score"]))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.4
    xs = np.arange(len(EMOTIONS_8))
    own_means, own_sems = [], []
    ctrl_means, ctrl_sems = [], []
    for emo, _ in EMOTIONS_8:
        own = by_emo_cond.get(emo, {}).get(emo, [])
        ctrl = by_emo_cond.get(emo, {}).get("control", [])
        own_means.append(statistics.mean(own) if own else float("nan"))
        own_sems.append(statistics.stdev(own)/math.sqrt(len(own)) if len(own) > 1 else 0)
        ctrl_means.append(statistics.mean(ctrl) if ctrl else float("nan"))
        ctrl_sems.append(statistics.stdev(ctrl)/math.sqrt(len(ctrl)) if len(ctrl) > 1 else 0)
    ax.bar(xs - width/2, own_means, width, yerr=own_sems,
           color=[VAL_COLOR[v] for _, v in EMOTIONS_8],
           edgecolor="black", linewidth=0.4, capsize=2, label="emotion-pre")
    ax.bar(xs + width/2, ctrl_means, width, yerr=ctrl_sems, color="#bbbbbb",
           edgecolor="black", linewidth=0.4, capsize=2, label="control")
    ax.axhline(0, color="#444", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([e for e, _ in EMOTIONS_8], fontsize=7, rotation=20)
    ax.set_ylabel("Haiku score 0-10", fontsize=8)
    ax.set_title("Haiku-judged emotion-in-continuation (emo-pre vs control)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = os.path.join(out_dir, "judge_per_emotion.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}")


def main(out_dir: str = "results/alpaca_emotion_preprompt",
         judge_path: Optional[str] = None):
    runs_path = os.path.join(out_dir, "runs.jsonl")
    rows = _load(runs_path)
    print(f"[plot-a8] {len(rows)} rows")
    for metric in ("dot", "cos"):
        plot_own_probe_delta(rows, out_dir, metric)
        plot_cont_mean_grid(rows, out_dir, metric)
    if judge_path is None:
        judge_path = os.path.join(out_dir, "runs.judged.jsonl")
    plot_judge(out_dir, judge_path)


if __name__ == "__main__":
    fire.Fire(main)
