"""Aggregate the alpaca-neutral 8-emotion pre-prompt experiment.

For each (prompt_idx, completion_idx) we have:
  - 1 control row ("Please respond thoroughly and clearly.")
  - 8 emotion rows ("Respond as though you are {emo}.")

All share the same neutral prefill and cutoff. We compute per-emotion
paired Δ = (emotion-row probe on its own emotion) − (matched control-row
probe on the same emotion).

EOT/EOS filter: drop pairs where either continuation begins with
<end_of_turn>/<eos>.
"""
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Optional

import fire

EOT_MARKERS = ("<end_of_turn>", "<eos>", "<|endoftext|>", "<bos>")


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
    out_path: Optional[str] = None,
):
    rows = [json.loads(l) for l in open(runs_path) if l.strip()]
    print(f"loaded {len(rows)} rows")

    by_pc = defaultdict(dict)
    for r in rows:
        key = (r["prompt_idx"], r["completion_idx"])
        by_pc[key][r["label"]] = r

    emos = sorted({r["label"] for r in rows if r["label"] != "control"})
    print(f"emotions: {emos}")
    print(f"groups: {len(by_pc)}")

    def report(region: str = "cont_mean", eot_filter: bool = True):
        out = {}
        for emo in emos:
            ds = []
            ms_e = []; ms_c = []
            for key, by_lbl in by_pc.items():
                if emo not in by_lbl or "control" not in by_lbl: continue
                er = by_lbl[emo]; cr = by_lbl["control"]
                if eot_filter and (is_eot_only(er["continued_text"]) or is_eot_only(cr["continued_text"])):
                    continue
                ve = er["metrics"]["cos"][region].get(emo)
                vc = cr["metrics"]["cos"][region].get(emo)
                if ve is None or vc is None: continue
                if isinstance(ve, float) and math.isnan(ve): continue
                if isinstance(vc, float) and math.isnan(vc): continue
                ds.append(ve - vc)
                ms_e.append(ve); ms_c.append(vc)
            if not ds:
                out[emo] = (float("nan"),)*5
                continue
            m, t, n = paired_t(ds)
            out[emo] = (statistics.mean(ms_e), statistics.mean(ms_c), m, t, n)
        return out

    for region in ("cont_mean", "cont_next5"):
        for eot in (True, False):
            r = report(region, eot_filter=eot)
            tag = "EOT-filt" if eot else "all"
            print(f"\n=== region={region}, {tag} — cos ===")
            print(f"{'emotion':12s}  {'M_emo':>9s}  {'M_ctrl':>9s}  {'Δ(e-c)':>10s}  {'t':>7s}  {'n':>5s}")
            for emo, (me, mc, m, t, n) in r.items():
                print(f"{emo:12s}  {me:+.4f}    {mc:+.4f}    {m:+.4f}     {t:+6.2f}   {n:5d}")

    if out_path:
        summary = {}
        for region in ("cont_mean", "cont_next1", "cont_next5", "cont_next10"):
            for eot in (True, False):
                tag = f"{region}_{'eot_filt' if eot else 'all'}"
                summary[tag] = {emo: list(t) for emo, t in report(region, eot_filter=eot).items()}
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
