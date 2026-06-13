"""Aggregate the reverse-instruction experiment.

Per (prompt_idx, completion_idx, pair_id) we have 4 paired conditions
(frust_pre, ctrl_pre, frust_inline, ctrl_inline) sharing one cutoff, plus
a baseline cell per (prompt_idx, completion_idx).

We report Δ_frust − Δ_ctrl on the relevant emotion (frustrated) at each
window, paired by (prompt_idx, completion_idx, pair_id). Also EOT-filtered.
"""
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Optional

import fire

EOT_MARKERS = ("<end_of_turn>", "<eos>", "<|endoftext|>", "<bos>")
RELEVANT = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]


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
    runs_path: str = "results/reverse_instruction/runs.jsonl",
    out_path: Optional[str] = None,
):
    rows = [json.loads(l) for l in open(runs_path) if l.strip()]
    print(f"loaded {len(rows)} rows")

    by_pcj = defaultdict(dict)
    for r in rows:
        if r["label"] == "baseline":
            continue
        key = (r["prompt_idx"], r["completion_idx"], r["pair_id"])
        by_pcj[key][r["label"]] = r

    print(f"unique (prompt, completion, pair_id) groups: {len(by_pcj)}")

    pairs_pre_all = []
    pairs_inline_all = []
    pairs_pre_kept = []
    pairs_inline_kept = []

    for key, by_lbl in by_pcj.items():
        if "frust_pre" in by_lbl and "ctrl_pre" in by_lbl:
            f = by_lbl["frust_pre"]; c = by_lbl["ctrl_pre"]
            pairs_pre_all.append((f, c))
            if not (is_eot_only(f["continued_text"]) or is_eot_only(c["continued_text"])):
                pairs_pre_kept.append((f, c))
        if "frust_inline" in by_lbl and "ctrl_inline" in by_lbl:
            f = by_lbl["frust_inline"]; c = by_lbl["ctrl_inline"]
            pairs_inline_all.append((f, c))
            if not (is_eot_only(f["continued_text"]) or is_eot_only(c["continued_text"])):
                pairs_inline_kept.append((f, c))

    def report(pairs, region: str = "cont_mean"):
        out = {}
        for emo in RELEVANT:
            ds = []
            ms_f = []; ms_c = []
            for f, c in pairs:
                vf = f["metrics"]["cos"][region].get(emo)
                vc = c["metrics"]["cos"][region].get(emo)
                if vf is None or vc is None: continue
                if isinstance(vf, float) and math.isnan(vf): continue
                if isinstance(vc, float) and math.isnan(vc): continue
                ds.append(vf - vc)
                ms_f.append(vf); ms_c.append(vc)
            if not ds:
                out[emo] = (float("nan"),)*5
                continue
            m_f = statistics.mean(ms_f); m_c = statistics.mean(ms_c)
            m, t, n = paired_t(ds)
            out[emo] = (m_f, m_c, m, t, n)
        return out

    for placement, pairs_all, pairs_kept in [
        ("PRE", pairs_pre_all, pairs_pre_kept),
        ("INLINE", pairs_inline_all, pairs_inline_kept),
    ]:
        print(f"\n=== {placement} — ALL pairs (n_groups={len(pairs_all)}) — cont_mean cos ===")
        print(f"{'emotion':12s}  {'M_frust':>9s}  {'M_ctrl':>9s}  {'Δ(f-c)':>10s}  {'t':>7s}  {'n':>5s}")
        for emo, (mf, mc, m, t, n) in report(pairs_all).items():
            print(f"{emo:12s}  {mf:+.4f}    {mc:+.4f}    {m:+.4f}     {t:+6.2f}   {n:5d}")
        print(f"\n=== {placement} — EOT-FILTERED (n_groups={len(pairs_kept)}) — cont_mean cos ===")
        print(f"{'emotion':12s}  {'M_frust':>9s}  {'M_ctrl':>9s}  {'Δ(f-c)':>10s}  {'t':>7s}  {'n':>5s}")
        for emo, (mf, mc, m, t, n) in report(pairs_kept).items():
            print(f"{emo:12s}  {mf:+.4f}    {mc:+.4f}    {m:+.4f}     {t:+6.2f}   {n:5d}")

    if out_path:
        summary = {"placements": {}}
        for placement, pairs_all, pairs_kept in [
            ("pre", pairs_pre_all, pairs_pre_kept),
            ("inline", pairs_inline_all, pairs_inline_kept),
        ]:
            summary["placements"][placement] = {
                "all": {emo: list(t) for emo, t in report(pairs_all).items()},
                "eot_filtered": {emo: list(t) for emo, t in report(pairs_kept).items()},
                "n_all": len(pairs_all), "n_kept": len(pairs_kept),
            }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
