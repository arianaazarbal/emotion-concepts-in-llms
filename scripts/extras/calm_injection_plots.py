"""Plot calm-injection experiment results.

Reads results/calm_injection/{exp1_runs,exp2_runs,exp2_validation}.jsonl and
produces small paired comparison plots of mean per-emotion probe values:

- For each metric (dot, cos):
  - Per emotion: paired strip plot of (baseline, with-injection) means over the
    "rest of completion" region.
  - Per emotion: same for the "next token after injection" position.
- Exp2 validation: paired comparison of avg frustration (plain vs reminder).

All plots are kept deliberately small (matches Ariana's preference).
"""
from __future__ import annotations

import json
import os
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np

RELEVANT_EMOTIONS = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]


def _load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_pairs(runs: list, metric: str, region_key_base: str, region_key_inj: str):
    """Return {emotion: ([baseline_values], [injection_values])} aligned by row."""
    out = {e: ([], []) for e in RELEVANT_EMOTIONS}
    for r in runs:
        m = r["metrics"][metric]
        for e in RELEVANT_EMOTIONS:
            bv = m[region_key_base].get(e)
            iv = m[region_key_inj].get(e)
            if bv is None or iv is None:
                continue
            if np.isnan(bv) or np.isnan(iv):
                continue
            out[e][0].append(bv)
            out[e][1].append(iv)
    return out


def _paired_strip(ax, base: List[float], inj: List[float], emo: str, ylabel: str):
    """Tiny paired strip plot: two columns (base / inj), one line per pair."""
    base_a = np.array(base, dtype=float)
    inj_a = np.array(inj, dtype=float)
    n = len(base_a)
    rng = np.random.default_rng(0)
    jitter = 0.04 * rng.standard_normal(n)
    ax.scatter(np.zeros(n) + jitter, base_a, s=12, color="#888", alpha=0.7, label="baseline")
    ax.scatter(np.ones(n) + jitter, inj_a, s=12, color="#1f77b4", alpha=0.7, label="injection")
    for i in range(n):
        ax.plot([0 + jitter[i], 1 + jitter[i]], [base_a[i], inj_a[i]],
                color="#cccccc", alpha=0.5, lw=0.6)
    if n:
        ax.scatter([0, 1], [base_a.mean(), inj_a.mean()], s=40, color="black", marker="_", zorder=5)
        delta = inj_a.mean() - base_a.mean()
        ax.set_title(f"{emo}\nΔ={delta:+.3f} (n={n})", fontsize=8)
    else:
        ax.set_title(f"{emo}\n(no data)", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base", "inj"], fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def _region_pairs_available(rows: list, metric: str) -> list:
    """Discover which (base, inj) region-name pairs exist in the data."""
    if not rows:
        return []
    keys = set(rows[0]["metrics"][metric].keys())
    pairs = []
    for k in sorted(keys):
        if k.startswith("base_"):
            inj_k = "inj_" + k[len("base_"):]
            if inj_k in keys:
                pairs.append((k, inj_k))
    return pairs


def plot_exp(runs_path: str, out_dir: str, tag: str):
    runs = _load_jsonl(runs_path)
    if not runs:
        print(f"[plot] {tag}: empty runs file, skipping")
        return
    print(f"[plot] {tag}: {len(runs)} rows")

    labels = sorted(set(r.get("injection_label", "calm") for r in runs))
    has_control = "control" in labels

    for metric in ("dot", "cos"):
        pairs = _region_pairs_available(runs, metric)
        for base_key, inj_key in pairs:
            region_name = base_key[len("base_"):]
            for label in labels:
                sub = [r for r in runs if r.get("injection_label", "calm") == label]
                paired = _extract_pairs(sub, metric, base_key, inj_key)
                fig, axes = plt.subplots(1, len(RELEVANT_EMOTIONS), figsize=(9.2, 2.0), sharey=False)
                for ax, emo in zip(axes, RELEVANT_EMOTIONS):
                    base, inj = paired[emo]
                    _paired_strip(ax, base, inj, emo, ylabel=metric if emo == RELEVANT_EMOTIONS[0] else "")
                fig.suptitle(f"{tag} ({label}) — {region_name} ({metric})", fontsize=9)
                fig.tight_layout(rect=(0, 0, 1, 0.92))
                suffix = f"_{label}" if has_control else ""
                out_path = os.path.join(out_dir, f"{tag}{suffix}_{region_name}_{metric}.png")
                fig.savefig(out_path, dpi=140)
                plt.close(fig)
                print(f"  saved {out_path}")

            if has_control:
                _plot_calm_vs_control(runs, out_dir, tag, metric, base_key, inj_key, region_name)


def _plot_calm_vs_control(runs, out_dir, tag, metric, base_key, inj_key, region_name):
    """Plot Δ = inj - base for calm vs control on the SAME (prompt, completion, j_pair) cutoff.

    With matched cutoffs, calm[j] and control[j] share base_rest. So we plot
    Δ_calm vs Δ_control side-by-side per emotion.
    """
    by_key = {}
    for r in runs:
        key = (r["prompt_idx"], r.get("completion_idx", 0), r["injection_id"])
        by_key.setdefault(key, {})[r.get("injection_label", "calm")] = r
    deltas = {e: ([], []) for e in RELEVANT_EMOTIONS}
    for key, by_lbl in by_key.items():
        if "calm" not in by_lbl or "control" not in by_lbl:
            continue
        for label in ("calm", "control"):
            row = by_lbl[label]
            m = row["metrics"][metric]
            bvs = m.get(base_key, {})
            ivs = m.get(inj_key, {})
            for e in RELEVANT_EMOTIONS:
                bv = bvs.get(e); iv = ivs.get(e)
                if bv is None or iv is None or np.isnan(bv) or np.isnan(iv):
                    continue
                deltas[e][0 if label == "calm" else 1].append(iv - bv)
    fig, axes = plt.subplots(1, len(RELEVANT_EMOTIONS), figsize=(9.2, 2.0), sharey=False)
    for ax, emo in zip(axes, RELEVANT_EMOTIONS):
        d_calm, d_ctrl = deltas[emo]
        ax.axhline(0, color="#999", lw=0.5)
        rng = np.random.default_rng(0)
        for x_idx, (vals, color) in enumerate([(d_calm, "#1f77b4"), (d_ctrl, "#aaaaaa")]):
            n = len(vals)
            jit = 0.04 * rng.standard_normal(n)
            ax.scatter(np.full(n, x_idx) + jit, vals, s=12, color=color, alpha=0.7)
            if n:
                ax.scatter([x_idx], [np.mean(vals)], s=40, color="black", marker="_", zorder=5)
        diff = np.mean(d_calm) - np.mean(d_ctrl) if d_calm and d_ctrl else float("nan")
        ax.set_title(f"{emo}\nΔΔ={diff:+.3f}", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["calm", "ctrl"], fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        if emo == RELEVANT_EMOTIONS[0]:
            ax.set_ylabel(f"Δ {metric}", fontsize=7)
    fig.suptitle(f"{tag} — Δ(inj−base) calm vs control · {region_name} ({metric})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = os.path.join(out_dir, f"{tag}_calmVScontrol_{region_name}_{metric}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_validation(val_path: str, out_dir: str):
    rows = _load_jsonl(val_path)
    if not rows:
        return
    for metric in ("dot", "cos"):
        sub = [r for r in rows if r["metric"] == metric]
        fig, axes = plt.subplots(1, len(RELEVANT_EMOTIONS), figsize=(9.2, 2.0), sharey=False)
        for ax, emo in zip(axes, RELEVANT_EMOTIONS):
            plain = [r["plain_mean"].get(emo) for r in sub]
            frust = [r["frust_mean"].get(emo) for r in sub]
            plain = [v for v in plain if v is not None and not np.isnan(v)]
            frust = [v for v in frust if v is not None and not np.isnan(v)]
            _paired_strip(ax, plain, frust, emo, ylabel=metric if emo == RELEVANT_EMOTIONS[0] else "")
            ax.set_xticklabels(["plain", "frust"], fontsize=7)
        fig.suptitle(f"exp2 validation — plain vs +frustrated reminder ({metric})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_path = os.path.join(out_dir, f"exp2_validation_{metric}.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"  saved {out_path}")


def main(out_dir: str = "results/calm_injection"):
    exp1 = os.path.join(out_dir, "exp1_runs.jsonl")
    exp2 = os.path.join(out_dir, "exp2_runs.jsonl")
    val = os.path.join(out_dir, "exp2_validation.jsonl")
    if os.path.exists(exp1):
        plot_exp(exp1, out_dir, "exp1")
    if os.path.exists(exp2):
        plot_exp(exp2, out_dir, "exp2")
    if os.path.exists(val):
        plot_validation(val, out_dir)


if __name__ == "__main__":
    fire.Fire(main)
