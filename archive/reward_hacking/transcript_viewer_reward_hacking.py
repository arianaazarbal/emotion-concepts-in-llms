"""Browser viewer for single-turn reward-hacking eval results.

Point it at the per-model results root produced by
``scripts/single_turn_reward_hacking_eval`` (e.g.
``results/single_turn_reward_hacking_eval/gemma2_9b``). The page lets you pick
a steering cell + prompt type from dropdowns, filter by hack / valid status,
and inspect each item's prompt, model completion, extracted answer and judge
response.

Usage:
    uv run python -m scripts.transcript_viewer_reward_hacking \\
        --root results/single_turn_reward_hacking_eval/gemma2_9b --port 8090
"""

import json
import os
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import fire

from src.utils.utils import load_json, load_jsonl

ROOT: str = ""
DATA_PATH_DEFAULT = "data/reward_hacking/code_generation.jsonl"
PROMPT_BY_PROBLEM: dict = {}


def _index_cells(root: str) -> dict:
    """Return ``{steer_tag: [prompt_type, ...]}`` for every cell with a results.json."""
    cells: dict = {}
    if not os.path.isdir(root):
        return cells
    for tag in sorted(os.listdir(root)):
        tag_dir = os.path.join(root, tag)
        if not os.path.isdir(tag_dir):
            continue
        pts = []
        for pt in sorted(os.listdir(tag_dir)):
            pt_dir = os.path.join(tag_dir, pt)
            if os.path.isfile(os.path.join(pt_dir, "results.json")):
                pts.append(pt)
        if pts:
            cells[tag] = pts
    return cells


def _load_cell(tag: str, pt: str) -> dict:
    return load_json(os.path.join(ROOT, tag, pt, "results.json"))


def _load_completions(tag: str, pt: str) -> dict:
    p = os.path.join(ROOT, tag, pt, "completions.json")
    if not os.path.isfile(p):
        return {}
    return load_json(p)


def _prompt_for(problem_statement: str) -> str:
    """Resolve the original user prompt from the source dataset, keyed by problem statement."""
    return PROMPT_BY_PROBLEM.get(problem_statement, "")


def _render_page(tag: str, pt: str, only_hack: bool, only_valid: bool, only_invalid: bool) -> str:
    cells = _index_cells(ROOT)
    tags = list(cells)
    if not tags:
        return f"<html><body><p>No cells found under {escape(ROOT)}.</p></body></html>"

    if tag not in cells:
        tag = tags[0]
    pts = cells[tag]
    if pt not in pts:
        pt = pts[0]

    payload = _load_cell(tag, pt)
    items = payload.get("items", [])
    stats = payload.get("stats", {})
    comp_payload = _load_completions(tag, pt)
    cfg = comp_payload.get("config", {})
    conversations = comp_payload.get("conversations") or []
    steering = cfg.get("steering", {})
    multiturn = cfg.get("multiturn", 1)

    def keep(it):
        if only_hack and it.get("is_hack") is not True:
            return False
        if only_valid and not it.get("is_valid"):
            return False
        if only_invalid and it.get("is_valid"):
            return False
        return True

    filtered = [(i, it) for i, it in enumerate(items) if keep(it)]

    tag_options = "".join(
        f'<option value="{escape(t)}"{" selected" if t == tag else ""}>{escape(t)}</option>'
        for t in tags
    )
    pt_options = "".join(
        f'<option value="{escape(p)}"{" selected" if p == pt else ""}>{escape(p)}</option>'
        for p in pts
    )

    cards = []
    for i, it in filtered:
        is_hack = it.get("is_hack")
        is_valid = it.get("is_valid")
        badge = []
        if is_valid:
            badge.append('<span class="b ok">valid</span>')
        else:
            badge.append('<span class="b bad">invalid</span>')
        if is_hack is True:
            badge.append('<span class="b hack">HACK</span>')
        elif is_hack is False:
            badge.append('<span class="b clean">clean</span>')
        elif is_valid:
            badge.append('<span class="b unk">judge-failed</span>')
        prompt = _prompt_for(it.get("problem_statement", ""))
        convo = conversations[i] if i < len(conversations) else None
        if convo:
            turn_blocks = []
            for j, m in enumerate(convo):
                role = m.get("role", "?")
                content = m.get("content", "")
                cls = {"user": "u", "assistant": "a", "system": "s"}.get(role, "x")
                is_final_assistant = (
                    role == "assistant" and j == len(convo) - 1
                )
                open_attr = " open" if is_final_assistant or role == "system" else ""
                label = f"turn {j} · {role}"
                if is_final_assistant:
                    label += " (judged)"
                turn_blocks.append(
                    f'<details{open_attr} class="msg {cls}"><summary>{escape(label)}</summary>'
                    f'<pre>{escape(content)}</pre></details>'
                )
            convo_block = (
                f'<details open><summary>conversation ({len(convo)} turns)</summary>'
                f'<div class="convo">{"".join(turn_blocks)}</div></details>'
            )
        else:
            convo_block = (
                f'<details{" open" if prompt else ""}><summary>user prompt</summary>'
                f'<pre>{escape(prompt) or "<i>not found in dataset</i>"}</pre></details>'
                f'<details open><summary>model completion</summary>'
                f'<pre>{escape(it.get("completion") or "")}</pre></details>'
            )
        cards.append(f"""
<div class="card">
  <div class="hd">#{i} {' '.join(badge)}</div>
  <details open><summary>problem statement</summary><pre>{escape(it.get("problem_statement", ""))}</pre></details>
  <details><summary>tests</summary><pre>{escape(it.get("tests", ""))}</pre></details>
  {convo_block}
  <details><summary>extracted answer</summary><pre>{escape(it.get("extracted_answer") or "")}</pre></details>
  <details{' open' if is_hack is True else ''}><summary>judge response</summary><pre>{escape(it.get("judge_response") or "")}</pre></details>
</div>
""")

    steering_str = json.dumps(steering, sort_keys=True) if steering else "(unsteered)"
    stats_str = json.dumps(stats, sort_keys=True)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>reward-hacking transcripts</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 1em auto; padding: 0 1em; color: #222; }}
.controls {{ position: sticky; top: 0; background: #fff; padding: 0.5em 0; border-bottom: 1px solid #ddd; z-index: 10; }}
.controls > * {{ margin-right: 0.5em; }}
.meta {{ font-size: 0.85em; color: #555; padding: 0.5em 0; }}
.card {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.75em 1em; margin: 1em 0; background: #fafafa; }}
.hd {{ font-weight: 600; margin-bottom: 0.5em; }}
.b {{ display: inline-block; padding: 0 0.5em; border-radius: 3px; font-size: 0.8em; margin-left: 0.3em; color: #fff; }}
.b.ok {{ background: #2a7; }} .b.bad {{ background: #999; }}
.b.hack {{ background: #c33; }} .b.clean {{ background: #46a; }} .b.unk {{ background: #b80; }}
.convo {{ border-left: 3px solid #ddd; padding-left: 0.6em; margin-top: 0.3em; }}
.msg.u > summary {{ color: #46a; font-weight: 600; }}
.msg.a > summary {{ color: #2a7; font-weight: 600; }}
.msg.s > summary {{ color: #888; font-style: italic; }}
details {{ margin: 0.4em 0; }}
summary {{ cursor: pointer; font-size: 0.85em; color: #555; user-select: none; }}
pre {{ background: #fff; border: 1px solid #e2e2e2; padding: 0.6em; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em; max-height: 600px; overflow: auto; }}
</style></head>
<body>
<form class="controls" method="get">
  <label>cell <select name="tag" onchange="this.form.submit()">{tag_options}</select></label>
  <label>prompt <select name="pt" onchange="this.form.submit()">{pt_options}</select></label>
  <label><input type="checkbox" name="hack" value="1"{' checked' if only_hack else ''} onchange="this.form.submit()"> hacks only</label>
  <label><input type="checkbox" name="valid" value="1"{' checked' if only_valid else ''} onchange="this.form.submit()"> valid only</label>
  <label><input type="checkbox" name="invalid" value="1"{' checked' if only_invalid else ''} onchange="this.form.submit()"> invalid only</label>
  <noscript><button type="submit">apply</button></noscript>
</form>
<div class="meta">
  <div>root: <code>{escape(ROOT)}</code></div>
  <div>steering: <code>{escape(steering_str)}</code></div>
  <div>multiturn: <code>{multiturn}</code></div>
  <div>stats: <code>{escape(stats_str)}</code></div>
  <div>showing {len(filtered)} / {len(items)} items</div>
</div>
{''.join(cards) if cards else '<p><i>no items match the filters.</i></p>'}
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        u = urlparse(self.path)
        if u.path != "/":
            self.send_response(404); self.end_headers(); return
        q = parse_qs(u.query)
        tag = (q.get("tag") or [""])[0]
        pt = (q.get("pt") or [""])[0]
        only_hack = bool(q.get("hack"))
        only_valid = bool(q.get("valid"))
        only_invalid = bool(q.get("invalid"))
        body = _render_page(tag, pt, only_hack, only_valid, only_invalid).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(root: str, port: int = 8090, data_path: str = DATA_PATH_DEFAULT):
    """Serve the transcript viewer.

    Args:
        root: Per-model results dir (contains ``{steer_tag}/{prompt_type}/results.json``).
        port: HTTP port.
        data_path: Source JSONL used by the eval (so the viewer can show the
            user prompt for each problem). Defaults to the same path the eval uses.
    """
    global ROOT, PROMPT_BY_PROBLEM
    ROOT = os.path.abspath(root)
    if os.path.isfile(data_path):
        for row in load_jsonl(data_path):
            PROMPT_BY_PROBLEM[row.get("problem_statement", "")] = row.get("prompt", "")
    cells = _index_cells(ROOT)
    print(f"[viewer] root={ROOT} cells={list(cells)}")
    print(f"[viewer] http://localhost:{port}/")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    fire.Fire(main)
