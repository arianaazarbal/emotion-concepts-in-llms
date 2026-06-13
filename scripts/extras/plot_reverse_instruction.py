"""Plot reverse-instruction results: paired strip per emotion, per placement.

Compact figs (Ariana prefers small)."""
import json
import math
import os
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np

EOT_MARKERS = ("<end_of_turn>", "<eos>", "<|endoftext|>", "<bos>")
EMOS = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]


def is_eot_only(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return any(s.startswith(m) for m in EOT_MARKERS)


def main(
    runs_path: str = "results/reverse_instruction/runs.jsonl",
    out_dir: str = "results/reverse_instruction",
    eot_filter: bool = True,
):
    rows = [json.loads(l) for l in open(runs_path) if l.strip()]
    by_pcj = defaultdict(dict)
    for r in rows:
        if r["label"] == "baseline":
            continue
        key = (r["prompt_idx"], r["completion_idx"], r["pair_id"])
        by_pcj[key][r["label"]] = r

    for placement, pos_lbl, neg_lbl in [("pre", "frust_pre", "ctrl_pre"),
                                         ("inline", "frust_inline", "ctrl_inline")]:
        pairs = []
        for key, by_lbl in by_pcj.items():
            if pos_lbl not in by_lbl or neg_lbl not in by_lbl: continue
            f, c = by_lbl[pos_lbl], by_lbl[neg_lbl]
            if eot_filter and (is_eot_only(f["continued_text"]) or is_eot_only(c["continued_text"])):
                continue
            pairs.append((f, c))

        fig, axes = plt.subplots(1, len(EMOS), figsize=(10.5, 2.2), sharey=False)
        for ax, emo in zip(axes, EMOS):
            d_f, d_c = [], []
            for f, c in pairs:
                vf = f["metrics"]["cos"]["cont_mean"].get(emo)
                vc = c["metrics"]["cos"]["cont_mean"].get(emo)
                if vf is None or vc is None: continue
                if math.isnan(vf) or math.isnan(vc): continue
                d_f.append(vf); d_c.append(vc)
            n = len(d_f)
            rng = np.random.default_rng(0)
            jit = 0.06 * rng.standard_normal(n)
            ax.scatter(np.zeros(n) + jit, d_f, s=10, color="#d62728", alpha=0.55, label="frust")
            ax.scatter(np.ones(n) + jit, d_c, s=10, color="#888", alpha=0.55, label="ctrl")
            for i in range(n):
                ax.plot([0 + jit[i], 1 + jit[i]], [d_f[i], d_c[i]],
                        color="#cccccc", alpha=0.35, lw=0.4)
            if n:
                mf = float(np.mean(d_f)); mc = float(np.mean(d_c))
                ax.scatter([0, 1], [mf, mc], s=50, color="black", marker="_", zorder=5)
                delta = mf - mc
                ax.set_title(f"{emo}\nΔ={delta:+.3f}", fontsize=8)
            else:
                ax.set_title(f"{emo}\n(no data)", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["frust", "ctrl"], fontsize=7)
            ax.tick_params(axis="y", labelsize=6)
            ax.axhline(0, color="#bbb", lw=0.4)
        suffix = "_eot_filt" if eot_filter else ""
        title = f"Exp R — {placement}-prompt frust-instr vs neutral-ctrl · cont_mean (cos) · n_pairs={len(pairs)}"
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = os.path.join(out_dir, f"strip_{placement}{suffix}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"saved {path} (n={len(pairs)})")


if __name__ == "__main__":
    fire.Fire(main)
