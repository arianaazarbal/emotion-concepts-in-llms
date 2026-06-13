"""Plots for the alpaca 8-emotion pre-prompt experiment.

Two plots, both small:
  1. Per-emotion paired strip: (emotion-pre, control) on the emotion's own probe.
  2. 8-emotion bar chart of Δ_(emo - ctrl) with t-stat labels.

Additional: small histograms of Haiku judge scores per condition, if
runs.judged.jsonl exists.
"""
import json
import math
import os
import statistics
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np

EOT_MARKERS = ("<end_of_turn>", "<eos>", "<|endoftext|>", "<bos>")
VALENCE_COLOR = {"positive": "#1f77b4", "negative": "#d62728", "neutral": "#777"}


def is_eot_only(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return any(s.startswith(m) for m in EOT_MARKERS)


def paired_t(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return m, float("nan"), n
    t = m / (sd / math.sqrt(n))
    return m, t, n


def main(
    runs_path: str = "results/alpaca_emotion_preprompt/runs.jsonl",
    out_dir: str = "results/alpaca_emotion_preprompt",
    judged_path: str = "results/alpaca_emotion_preprompt/runs.judged.jsonl",
    eot_filter: bool = True,
):
    rows = [json.loads(l) for l in open(runs_path) if l.strip()]
    by_pc = defaultdict(dict)
    for r in rows:
        by_pc[(r["prompt_idx"], r["completion_idx"])][r["label"]] = r

    emos_pos = [e for e, r in {r["label"]: r["valence"] for r in rows if r["label"] != "control"}.items()
                if r == "positive"]
    emos_neg = [e for e, r in {r["label"]: r["valence"] for r in rows if r["label"] != "control"}.items()
                if r == "negative"]
    emos_ordered = emos_pos + emos_neg

    # ---------- Plot 1: 8-emotion bar chart ----------
    deltas = []
    tstats = []
    ns = []
    valences = []
    for emo in emos_ordered:
        ds = []
        for key, by_lbl in by_pc.items():
            if emo not in by_lbl or "control" not in by_lbl: continue
            er = by_lbl[emo]; cr = by_lbl["control"]
            if eot_filter and (is_eot_only(er["continued_text"]) or is_eot_only(cr["continued_text"])):
                continue
            ve = er["metrics"]["cos"]["cont_mean"].get(emo)
            vc = cr["metrics"]["cos"]["cont_mean"].get(emo)
            if ve is None or vc is None: continue
            if isinstance(ve, float) and math.isnan(ve): continue
            if isinstance(vc, float) and math.isnan(vc): continue
            ds.append(ve - vc)
        m, t, n = paired_t(ds)
        deltas.append(m); tstats.append(t); ns.append(n)
        valences.append({e: by_lbl[e].get("valence") for e in [emo] for by_lbl in [next(iter(by_pc.values()))]}.get(emo))

    # rebuild valences cleanly
    valences = []
    for emo in emos_ordered:
        v = next((r["valence"] for r in rows if r["label"] == emo), None)
        valences.append(v)

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    xs = np.arange(len(emos_ordered))
    colors = [VALENCE_COLOR.get(v, "#777") for v in valences]
    ax.bar(xs, deltas, color=colors, alpha=0.85, edgecolor="black", lw=0.6)
    ax.axhline(0, color="#888", lw=0.5)
    for i, (d, t, n) in enumerate(zip(deltas, tstats, ns)):
        if not math.isnan(t):
            ax.text(i, d, f"t={t:+.1f}\nn={n}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(emos_ordered, rotation=20, fontsize=8)
    ax.set_ylabel("Δ probe (emo − ctrl), cos · cont_mean", fontsize=8)
    ax.set_title(f"Alpaca 8-emotion pre-prompt — paired Δ on prompted emotion's own probe (n_groups≈{ns[0] if ns else 0}; eot-filt={eot_filter})", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out_bar = os.path.join(out_dir, f"bar_8emotion{'_eot' if eot_filter else '_all'}.png")
    fig.savefig(out_bar, dpi=140)
    plt.close(fig)
    print(f"saved {out_bar}")

    # ---------- Plot 2: paired strip plot, 8 emos × 1 row ----------
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.8), sharey=False)
    axes = axes.flatten()
    for ax, emo, val in zip(axes, emos_ordered, valences):
        d_e, d_c = [], []
        for key, by_lbl in by_pc.items():
            if emo not in by_lbl or "control" not in by_lbl: continue
            er = by_lbl[emo]; cr = by_lbl["control"]
            if eot_filter and (is_eot_only(er["continued_text"]) or is_eot_only(cr["continued_text"])):
                continue
            ve = er["metrics"]["cos"]["cont_mean"].get(emo)
            vc = cr["metrics"]["cos"]["cont_mean"].get(emo)
            if ve is None or vc is None: continue
            if math.isnan(ve) or math.isnan(vc): continue
            d_e.append(ve); d_c.append(vc)
        n = len(d_e)
        rng = np.random.default_rng(0)
        jit = 0.06 * rng.standard_normal(n)
        color = VALENCE_COLOR.get(val, "#777")
        ax.scatter(np.zeros(n) + jit, d_e, s=10, color=color, alpha=0.6)
        ax.scatter(np.ones(n) + jit, d_c, s=10, color="#888", alpha=0.6)
        for i in range(n):
            ax.plot([0 + jit[i], 1 + jit[i]], [d_e[i], d_c[i]], color="#cccccc", alpha=0.35, lw=0.4)
        if n:
            ax.scatter([0, 1], [np.mean(d_e), np.mean(d_c)], s=50, color="black", marker="_", zorder=5)
            ax.set_title(f"{emo}\nΔ={np.mean(d_e) - np.mean(d_c):+.3f}", fontsize=8)
        ax.axhline(0, color="#bbb", lw=0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["emo", "ctrl"], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(f"Alpaca 8-emotion: emo-pre vs ctrl on each emotion's own probe (cos · cont_mean)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_strip = os.path.join(out_dir, f"strip_8emotion{'_eot' if eot_filter else '_all'}.png")
    fig.savefig(out_strip, dpi=140)
    plt.close(fig)
    print(f"saved {out_strip}")

    # ---------- Optional: judge histograms ----------
    if os.path.exists(judged_path):
        judges = [json.loads(l) for l in open(judged_path) if l.strip()]
        # group: keyed (prompt_idx, completion_idx, emotion) → {emo_score, ctrl_score}
        # judge rows are (prompt_idx, completion_idx, label, emotion)
        # for label != control: emotion == label
        # for label == control: emotion enumerates all emos (emit_per_label_control)
        d_per_emo = defaultdict(lambda: {"emo": [], "ctrl": []})
        for j in judges:
            if j["score"] is None: continue
            if j["label"] == "control":
                d_per_emo[j["emotion"]]["ctrl"].append(j["score"])
            else:
                d_per_emo[j["emotion"]]["emo"].append(j["score"])

        fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.8), sharey=False)
        axes = axes.flatten()
        for ax, emo, val in zip(axes, emos_ordered, valences):
            data = d_per_emo[emo]
            e_mean = float(np.mean(data["emo"])) if data["emo"] else float("nan")
            c_mean = float(np.mean(data["ctrl"])) if data["ctrl"] else float("nan")
            bins = np.arange(-0.5, 11.5, 1)
            color = VALENCE_COLOR.get(val, "#777")
            ax.hist(data["emo"], bins=bins, color=color, alpha=0.7, label=f"emo (m={e_mean:.1f})")
            ax.hist(data["ctrl"], bins=bins, color="#888", alpha=0.5, label=f"ctrl (m={c_mean:.1f})")
            ax.set_title(f"{emo}", fontsize=8)
            ax.set_xlim(-0.5, 10.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)
        fig.suptitle("Alpaca 8-emotion · Haiku judge score (0-10) on continuation per prompted emotion", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_judge = os.path.join(out_dir, "judge_histograms.png")
        fig.savefig(out_judge, dpi=140)
        plt.close(fig)
        print(f"saved {out_judge}")


if __name__ == "__main__":
    fire.Fire(main)
