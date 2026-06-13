"""Probe unspoken emotions in Bob/Amy conversations conditioned on backstories.

For every (backstory, conversation) pair in data/bob_amy_backstories.json x
data/bob_amy_chats.json, build the text ``backstory + "\\n\\n" + "\\n".join(
f"{speaker}: {text}")`` and run the emotion probe over it. For each of four
emotions (happy, sad, at ease, worried) we compute:

- per-speaker averages: mean probe score over tokens whose char-span falls
  inside an "Amy: ..." or "Bob: ..." line (speaker prefix inclusive,
  trailing newline exclusive).
- per-backstory averages: mean probe score over backstory tokens (the prefix
  of the text up to the blank line), averaged across conversations.

Outputs go to ``results/unspoken_emotions/{model_tag}/``:
- ``scores.json`` — full per-pair, per-emotion line-level averages
- ``heatmap_{emotion}_{amy|bob}.png`` — 4 emotions x 2 speakers = 8 heatmaps
- ``backstory_bars.png`` — grouped bar chart (4 emotions per backstory)
- ``token_mapping_check.txt`` — reconstruction sanity check for one pair
"""

import json
import os
from typing import List, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.emotion_probe import (
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import load_json


EMOTIONS = ["happy", "sad", "at ease", "worried"]


def build_text(backstory: str, chat: List[dict]) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Return ``(text, line_spans)``.

    ``line_spans`` is a list of ``(speaker, char_start, char_end)`` tuples
    describing each chat line's character range inside ``text``. The range is
    inclusive of the speaker prefix (``Amy:`` / ``Bob:``) and excludes the
    trailing newline.
    """
    joined_lines = [f"{turn['speaker']}: {turn['text']}" for turn in chat]
    chat_text = "\n".join(joined_lines)
    sep = "\n\n"
    text = backstory + sep + chat_text
    spans: List[Tuple[str, int, int]] = []
    cursor = len(backstory) + len(sep)
    for line, turn in zip(joined_lines, chat):
        start = cursor
        end = cursor + len(line)
        spans.append((turn["speaker"], start, end))
        cursor = end + 1
    return text, spans


def tokens_in_range(
    offsets: List[Tuple[int, int]], lo: int, hi: int
) -> List[int]:
    """Indices of tokens whose char span lies within ``[lo, hi)``.

    Uses character start as the membership key so a token spanning a newline
    is attributed to the line it starts in (matching the user's request to
    end "at the last token before the \\n").
    """
    idx = []
    for i, (ts, te) in enumerate(offsets):
        if ts == te:
            continue
        if ts >= lo and te <= hi:
            idx.append(i)
    return idx


def sanity_check_mapping(
    text: str,
    token_offsets: List[Tuple[int, int]],
    backstory_end: int,
    line_spans: List[Tuple[str, int, int]],
    out_path: str,
) -> None:
    """Write a human-readable reconstruction of the token-to-line mapping.

    For each line (and the backstory) we list the tokens attributed to it by
    joining their char slices. If reconstruction diverges from the source
    string we raise so the caller sees the failure immediately.
    """
    def _norm(s: str) -> str:
        return "".join(s.split())

    mismatches = 0
    lines = ["=== backstory ==="]
    bs_idx = tokens_in_range(token_offsets, 0, backstory_end)
    recon = "".join(text[token_offsets[i][0] : token_offsets[i][1]] for i in bs_idx)
    ok = _norm(recon) == _norm(text[:backstory_end])
    lines.append(f"match={ok}  n_tokens={len(bs_idx)}")
    lines.append(f"recon: {recon!r}")
    lines.append(f"orig : {text[:backstory_end]!r}")
    if not ok:
        mismatches += 1

    for speaker, lo, hi in line_spans:
        tok_idx = tokens_in_range(token_offsets, lo, hi)
        recon = "".join(text[token_offsets[i][0] : token_offsets[i][1]] for i in tok_idx)
        ok = _norm(recon) == _norm(text[lo:hi])
        lines.append(f"--- {speaker} [{lo}:{hi}] n_tokens={len(tok_idx)} match={ok} ---")
        lines.append(f"recon: {recon!r}")
        lines.append(f"orig : {text[lo:hi]!r}")
        if not ok:
            mismatches += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"mismatches={mismatches} (whitespace-normalized)\n\n")
        f.write("\n".join(lines))
    if mismatches:
        print(f"[check] WARNING: {mismatches} line(s) had token/char reconstruction mismatch; see {out_path}")


def main(
    model: str = "gemma2_9b",
    layer: int = None,
    version: str = "v0",
    start_at_nth_token: int = 50,
    cosine_sim: bool = False,
    denoised: bool = False,
    max_length: int = 2048,
    chats_path: str = None,
    backstories_path: str = None,
    output_dir: str = None,
    seed: int = 0,
    focus_chat: str = "cafe_invite",
    v1: bool = False,
):
    """Run the Bob/Amy unspoken-emotion probe and produce heatmaps + bar chart.

    Args:
        model: Short model name from data/model_names.json.
        layer: Probe layer; defaults to the model's primary layer (or ~2/3).
        version: Emotion-vector version tag (e.g. ``v0``).
        start_at_nth_token: Which saved emotion-vector variant to load
            (mirrors ``start_at_nth_token`` in extract_emotion_vectors). Does
            not affect which story tokens are probed — all tokens are scored.
        cosine_sim: Use cosine similarity instead of raw dot product.
        denoised: Load PCA-denoised emotion vectors.
        max_length: Tokenizer truncation cap.
        chats_path: JSON of chats. Defaults to v0 or v1 file depending on ``v1``.
        backstories_path: JSON of backstories. Defaults as above.
        output_dir: Override the default ``results/unspoken_emotions/{model_tag}[_v1]``.
        seed: Reproducibility seed (no stochasticity in probe itself, but kept
            for parity with other scripts).
        v1: Use the v1 data files (equal-length backstories, 30-turn chats).
    """
    torch.manual_seed(seed)

    data_tag = "v1" if v1 else "v0"
    if chats_path is None:
        chats_path = f"data/bob_amy_chats{'_v1' if v1 else ''}.json"
    if backstories_path is None:
        backstories_path = f"data/bob_amy_backstories{'_v1' if v1 else ''}.json"
    model_tag = f"{model}_{version}" if version else model
    out_tag = f"{model_tag}_{data_tag}"
    out_dir = output_dir or f"results/unspoken_emotions/{out_tag}"
    os.makedirs(out_dir, exist_ok=True)

    chats_raw = load_json(chats_path)
    backstories_raw = load_json(backstories_path)
    chat_names = [c["name"] for c in chats_raw]
    chats = [c["messages"] for c in chats_raw]
    backstory_names = [b["name"] for b in backstories_raw]
    backstories = [b["text"] for b in backstories_raw]
    print(f"[load] {len(chats)} chats ({chat_names}), {len(backstories)} backstories ({backstory_names})")

    print(f"[load] model + tokenizer: {model}")
    hf_model, tokenizer = load_model_and_tokenizer(model)
    emotions, vec_by_layer = load_emotion_vectors(
        model,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        version=version,
    )
    missing = [e for e in EMOTIONS if e not in emotions]
    if missing:
        raise ValueError(
            f"Emotions {missing} missing from vectors for {model_tag}. "
            f"Available: {sorted(emotions)[:10]}..."
        )

    if layer is None:
        from src.utils.emotion_probe import default_layer
        layer = default_layer(hf_model)
    print(f"[probe] layer={layer} emotions={EMOTIONS}")

    nB, nC = len(backstories), len(chats)
    buckets = ["full", "name", "colon", "rest"]
    amy_by_bucket = {b: np.zeros((len(EMOTIONS), nB, nC), dtype=np.float32) for b in buckets}
    bob_by_bucket = {b: np.zeros((len(EMOTIONS), nB, nC), dtype=np.float32) for b in buckets}
    amy_scores = amy_by_bucket["full"]
    bob_scores = bob_by_bucket["full"]
    backstory_scores = np.zeros((len(EMOTIONS), nB, nC), dtype=np.float32)

    mapping_check_written = False
    per_pair_records = []
    per_pair_lines: List[List[List[dict]]] = [
        [[] for _ in range(nC)] for _ in range(nB)
    ]

    for bi, backstory in enumerate(backstories):
        for ci, chat in enumerate(chats):
            text, line_spans = build_text(backstory, chat)
            backstory_end = len(backstory)

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            offsets = enc["offset_mapping"][0].tolist()

            if not mapping_check_written:
                sanity_check_mapping(
                    text,
                    offsets,
                    backstory_end,
                    line_spans,
                    f"{out_dir}/token_mapping_check.txt",
                )
                mapping_check_written = True
                print(f"[check] token->char mapping verified; wrote {out_dir}/token_mapping_check.txt")

            result = emotion_probe(
                texts=[text],
                model=hf_model,
                tokenizer=tokenizer,
                emotion_vectors_by_layer=vec_by_layer,
                emotions=emotions,
                layers=[layer],
                selected_emotions=EMOTIONS,
                aggregation="none",
                batch_size=1,
                max_length=max_length,
                cosine_sim=cosine_sim,
            )
            per_token = result.scores[layer][0].numpy()  # [T_valid, E]
            offsets_used = offsets[: per_token.shape[0]]

            bs_idx = tokens_in_range(offsets_used, 0, backstory_end)
            if bs_idx:
                bs_avg = per_token[bs_idx].mean(axis=0)
            else:
                bs_avg = np.zeros(len(EMOTIONS), dtype=np.float32)

            per_speaker_bucket: dict = {
                "Amy": {b: [] for b in buckets},
                "Bob": {b: [] for b in buckets},
            }
            line_records = []
            for speaker, lo, hi in line_spans:
                idx_full = tokens_in_range(offsets_used, lo, hi)
                if not idx_full:
                    continue
                name_end = lo + len(speaker)
                idx_name = tokens_in_range(offsets_used, lo, name_end)
                idx_colon = tokens_in_range(offsets_used, name_end, name_end + 1)
                idx_rest = tokens_in_range(offsets_used, name_end + 1, hi)
                line_avg = per_token[idx_full].mean(axis=0)
                name_avg = per_token[idx_name].mean(axis=0) if idx_name else None
                colon_avg = per_token[idx_colon].mean(axis=0) if idx_colon else None
                rest_avg = per_token[idx_rest].mean(axis=0) if idx_rest else None
                line_records.append({
                    "speaker": speaker,
                    "char_start": lo,
                    "char_end": hi,
                    "n_tokens": len(idx_full),
                    "scores": {e: float(line_avg[k]) for k, e in enumerate(EMOTIONS)},
                })
                if speaker in per_speaker_bucket:
                    per_speaker_bucket[speaker]["full"].append(line_avg)
                    if name_avg is not None:
                        per_speaker_bucket[speaker]["name"].append(name_avg)
                    if colon_avg is not None:
                        per_speaker_bucket[speaker]["colon"].append(colon_avg)
                    if rest_avg is not None:
                        per_speaker_bucket[speaker]["rest"].append(rest_avg)

            per_pair_lines[bi][ci] = line_records
            for bucket in buckets:
                a = per_speaker_bucket["Amy"][bucket]
                b = per_speaker_bucket["Bob"][bucket]
                a_mean = np.stack(a).mean(axis=0) if a else np.zeros(len(EMOTIONS))
                b_mean = np.stack(b).mean(axis=0) if b else np.zeros(len(EMOTIONS))
                for k in range(len(EMOTIONS)):
                    amy_by_bucket[bucket][k, bi, ci] = a_mean[k]
                    bob_by_bucket[bucket][k, bi, ci] = b_mean[k]
            amy_vals = per_speaker_bucket["Amy"]["full"]
            bob_vals = per_speaker_bucket["Bob"]["full"]
            amy_mean = np.stack(amy_vals).mean(axis=0) if amy_vals else np.zeros(len(EMOTIONS))
            bob_mean = np.stack(bob_vals).mean(axis=0) if bob_vals else np.zeros(len(EMOTIONS))
            for k in range(len(EMOTIONS)):
                backstory_scores[k, bi, ci] = bs_avg[k]

            per_pair_records.append({
                "backstory_idx": bi,
                "backstory_name": backstory_names[bi],
                "chat_idx": ci,
                "chat_name": chat_names[ci],
                "amy_mean": {e: float(amy_mean[k]) for k, e in enumerate(EMOTIONS)},
                "bob_mean": {e: float(bob_mean[k]) for k, e in enumerate(EMOTIONS)},
                "backstory_mean": {e: float(bs_avg[k]) for k, e in enumerate(EMOTIONS)},
                "lines": line_records,
            })
            print(f"[probe] backstory={backstory_names[bi]} chat={chat_names[ci]} done (|Amy|={len(amy_vals)} |Bob|={len(bob_vals)})")

    scores_path = f"{out_dir}/scores.json"
    with open(scores_path, "w") as f:
        json.dump(
            {
                "model": model,
                "model_tag": model_tag,
                "data_version": data_tag,
                "layer": layer,
                "emotions": EMOTIONS,
                "metric": "cosine_sim" if cosine_sim else "dot_product",
                "pairs": per_pair_records,
            },
            f,
            indent=2,
        )
    print(f"[save] wrote {scores_path}")

    for bucket in buckets:
        _plot_heatmaps(
            amy_by_bucket[bucket],
            bob_by_bucket[bucket],
            backstory_names,
            chat_names,
            out_dir,
            bucket=bucket,
        )
    for nth in [1, 7, 14]:
        amy_t, bob_t = _per_speaker_turn_arrays(per_pair_lines, nth, nB, nC)
        _plot_heatmaps(
            amy_t, bob_t, backstory_names, chat_names, out_dir,
            bucket=f"speaker_turn_{nth}",
        )
    _plot_backstory_bars(backstory_scores, backstory_names, out_dir)
    if focus_chat:
        if focus_chat not in chat_names:
            print(f"[plot] focus_chat={focus_chat!r} not in {chat_names}; skipping focus plot")
        else:
            _plot_focus_chat(
                amy_scores, bob_scores, backstory_names, chat_names, focus_chat, out_dir
            )
            for emo in ["happy", "sad"]:
                _plot_focus_per_turn(
                    per_pair_lines, backstory_names, chat_names, focus_chat, emo, out_dir
                )
    print(f"[done] outputs in {out_dir}")


def _per_speaker_turn_arrays(
    per_pair_lines: List[List[List[dict]]],
    nth: int,
    nB: int,
    nC: int,
) -> tuple:
    """Return (amy, bob) arrays shape [E, nB, nC] for each speaker's n-th turn.

    ``nth`` is 1-indexed (1 = first time that speaker talks). If a speaker
    doesn't have that many turns in a given conversation, their cell is NaN.
    """
    amy = np.full((len(EMOTIONS), nB, nC), np.nan, dtype=np.float32)
    bob = np.full((len(EMOTIONS), nB, nC), np.nan, dtype=np.float32)
    for bi in range(nB):
        for ci in range(nC):
            recs = per_pair_lines[bi][ci]
            amy_turns = [r for r in recs if r["speaker"] == "Amy"]
            bob_turns = [r for r in recs if r["speaker"] == "Bob"]
            if len(amy_turns) >= nth:
                r = amy_turns[nth - 1]
                amy[:, bi, ci] = [r["scores"][e] for e in EMOTIONS]
            if len(bob_turns) >= nth:
                r = bob_turns[nth - 1]
                bob[:, bi, ci] = [r["scores"][e] for e in EMOTIONS]
    return amy, bob


def _plot_heatmaps(
    amy_scores: np.ndarray,
    bob_scores: np.ndarray,
    backstory_names: List[str],
    chat_names: List[str],
    out_dir: str,
    bucket: str = "full",
) -> None:
    """Grid of heatmaps: rows = emotions, cols = speaker (Amy, Bob).

    ``bucket`` restricts which tokens contributed:
    - ``full``: all tokens in the line (speaker prefix + colon + message)
    - ``name``: only tokens covering the literal ``Amy``/``Bob`` chars
    - ``colon``: only the token covering the ``:`` char
    - ``rest``: tokens after the colon (message body)
    """
    nE, nB, nC = amy_scores.shape
    fig, axes = plt.subplots(nE, 2, figsize=(6.5, 1.8 * nE + 1.0), squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    for k, emo in enumerate(EMOTIONS):
        for j, (speaker, data) in enumerate([("Amy", amy_scores[k]), ("Bob", bob_scores[k])]):
            ax = axes[k, j]
            masked = np.ma.masked_invalid(data)
            im = ax.imshow(masked, aspect="auto", cmap=cmap)
            ax.set_title(f"{emo} / {speaker}", fontsize=9)
            ax.set_xticks(range(nC))
            ax.set_xticklabels(chat_names, fontsize=7, rotation=30, ha="right")
            ax.set_yticks(range(nB))
            ax.set_yticklabels(backstory_names, fontsize=7)
            if k == nE - 1:
                ax.set_xlabel("conversation", fontsize=8)
            if j == 0:
                ax.set_ylabel("backstory", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"bucket = {bucket}", fontsize=10)
    fig.tight_layout()
    suffix = "" if bucket == "full" else f"_{bucket}"
    path = f"{out_dir}/heatmaps{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_focus_per_turn(
    per_pair_lines: List[List[List[dict]]],
    backstory_names: List[str],
    chat_names: List[str],
    focus_chat: str,
    emotion: str,
    out_dir: str,
) -> None:
    """For ``focus_chat``, one subplot per speaker (Amy, Bob) with one line
    per backstory showing ``emotion`` probe scores across conversation turns."""
    ci = chat_names.index(focus_chat)
    nB = len(backstory_names)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True, squeeze=False)
    for si, speaker in enumerate(["Amy", "Bob"]):
        ax = axes[0, si]
        n_turns = 0
        for bi in range(nB):
            recs = per_pair_lines[bi][ci]
            n_turns = max(n_turns, len(recs))
            xs, ys = [], []
            for ti, rec in enumerate(recs):
                if rec["speaker"] == speaker:
                    xs.append(ti)
                    ys.append(rec["scores"][emotion])
            ax.plot(
                xs, ys, "o-",
                label=backstory_names[bi],
                color=colors[bi % len(colors)],
                markersize=4,
            )
        ax.set_title(speaker, fontsize=10)
        ax.set_xlabel("turn", fontsize=8)
        ax.set_xticks(range(n_turns))
        if si == 0:
            ax.set_ylabel(f"{emotion} probe", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="k", lw=0.4)
        ax.legend(fontsize=7, title="backstory", title_fontsize=7)
    fig.suptitle(f"chat = {focus_chat}: {emotion} across turns", fontsize=10)
    fig.tight_layout()
    safe_emo = emotion.replace(" ", "_")
    path = f"{out_dir}/focus_{focus_chat}_{safe_emo}_per_turn.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_focus_chat(
    amy_scores: np.ndarray,
    bob_scores: np.ndarray,
    backstory_names: List[str],
    chat_names: List[str],
    focus_chat: str,
    out_dir: str,
) -> None:
    """For a single conversation, plot Bob vs Amy per emotion across backstories.

    One subplot per emotion; x-axis = backstory; two bars per backstory (Amy, Bob).
    """
    ci = chat_names.index(focus_chat)
    nE, nB = len(EMOTIONS), len(backstory_names)
    ncols = 2
    nrows = (nE + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 2.4 * nrows), squeeze=False)
    x = np.arange(nB)
    width = 0.38
    for k, emo in enumerate(EMOTIONS):
        ax = axes[k // ncols, k % ncols]
        amy_vals = amy_scores[k, :, ci]
        bob_vals = bob_scores[k, :, ci]
        ax.bar(x - width / 2, amy_vals, width, label="Amy", color="#c65b7c")
        ax.bar(x + width / 2, bob_vals, width, label="Bob", color="#3f7cac")
        ax.set_title(emo, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(backstory_names, fontsize=7, rotation=20, ha="right")
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="k", lw=0.5)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(nE, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(f"chat = {focus_chat}: Amy vs Bob by backstory", fontsize=10)
    fig.tight_layout()
    path = f"{out_dir}/focus_{focus_chat}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_backstory_bars(
    backstory_scores: np.ndarray, backstory_names: List[str], out_dir: str
) -> None:
    """Grouped bars: one x-tick per backstory, one bar per emotion (avg over chats)."""
    nE, nB, _ = backstory_scores.shape
    per_bs = backstory_scores.mean(axis=2)  # [E, B]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(nB)
    width = 0.8 / nE
    for k, emo in enumerate(EMOTIONS):
        ax.bar(x + (k - (nE - 1) / 2) * width, per_bs[k], width, label=emo)
    ax.set_xticks(x)
    ax.set_xticklabels(backstory_names, fontsize=8)
    ax.set_ylabel("avg probe score over backstory tokens", fontsize=8)
    ax.legend(fontsize=7, ncol=len(EMOTIONS), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = f"{out_dir}/backstory_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    fire.Fire(main)
