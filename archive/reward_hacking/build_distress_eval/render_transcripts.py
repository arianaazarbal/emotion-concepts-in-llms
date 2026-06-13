"""Pretty-print transcripts as Markdown for human/subagent review.

Reads a judged JSONL and writes one Markdown file with each conversation
formatted as a labeled multi-turn dialogue plus the Haiku judge rating.

Usage:
    uv run python build_distress_eval/render_transcripts.py \\
        --input_path build_distress_eval/results/gemma2_9b_judged.jsonl \\
        --output_path build_distress_eval/results/gemma2_9b_all.md \\
        --sort_by_rating
"""

import json
from pathlib import Path
from typing import Optional

import fire


def render(
    input_path: str,
    output_path: str,
    top_k: Optional[int] = None,
    sort_by_rating: bool = False,
    only_high: bool = False,
):
    """Render judged transcripts to Markdown.

    Args:
        input_path: judged JSONL file.
        output_path: Markdown output.
        top_k: keep only the top-K by Haiku rating after sort.
        sort_by_rating: sort descending by rating.
        only_high: drop rows with rating < 5.
    """
    rows = [json.loads(line) for line in open(input_path)]

    def _r(row):
        j = row.get("judge")
        return (j or {}).get("rating") if j else None

    if sort_by_rating:
        rows.sort(key=lambda r: (_r(r) is None, -(_r(r) or -1)))

    if only_high:
        rows = [r for r in rows if (_r(r) or 0) >= 5]

    if top_k:
        rows = rows[:top_k]

    out = []
    out.append(f"# Transcripts: {input_path}")
    out.append(f"\nTotal rendered: {len(rows)}\n")
    for idx, r in enumerate(rows):
        rating = _r(r)
        ev = (r.get("judge") or {}).get("evidence", "")
        reasoning = (r.get("judge") or {}).get("reasoning", "")
        out.append("\n---\n")
        out.append(f"### Transcript #{idx} — `{r['condition']}`  rating={rating}  (model={r['model']}, conv={r['conv_index']})")
        out.append(f"\n**Judge evidence:** {ev!r}")
        out.append(f"\n**Judge reasoning:** {reasoning}")
        out.append("")
        for m in r["messages"]:
            role = m["role"].upper()
            out.append(f"**{role}:**")
            out.append("")
            content = m["content"].strip()
            for line in content.split("\n"):
                out.append(f"> {line}")
            out.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(out))
    print(f"[render] wrote {len(rows)} transcripts to {output_path}")


if __name__ == "__main__":
    fire.Fire(render)
