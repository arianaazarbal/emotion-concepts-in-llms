"""Browser-viewable emotion story browser.

Serves a single page that, for a randomly sampled set of N topics, displays one
random story per (emotion, topic) drawn from
``results/stories/{model}/emotion_to_topic_to_stories.json``. A Regenerate
button reloads the page with a fresh random seed to re-sample topics and
stories. The current seed is visible and can be passed back as a query param
for reproducibility.
"""

import html
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import fire


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Emotion Story Browser</title>
<style>
  :root {{ --bg:#faf7f2; --ink:#1c1c1c; --muted:#666; --card:#ffffff;
           --line:#e3e0db; --accent:#2b7a4b; --accent-dark:#245f3d;
           --topic:#4a4a4a; --summary-bg:#f4ece0; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
          "Iowan Old Style", Georgia, serif; background: var(--bg); color: var(--ink);
          line-height: 1.65; }}
  .layout {{ display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }}
  nav.sidebar {{ position: sticky; top: 0; align-self: start; height: 100vh;
                 overflow-y: auto; border-right: 1px solid var(--line);
                 padding: 16px 14px; background: var(--card); }}
  nav.sidebar h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .1em;
                    color: var(--muted); margin: 0 0 10px; }}
  nav.sidebar input {{ width: 100%; padding: 6px 8px; margin-bottom: 10px;
                       border: 1px solid var(--line); border-radius: 6px;
                       font-size: 13px; }}
  nav.sidebar ul {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
  nav.sidebar a {{ color: #333; text-decoration: none; display: block;
                   padding: 3px 6px; border-radius: 4px; }}
  nav.sidebar a:hover {{ background: #f1ede5; color: #000; }}
  main {{ max-width: 820px; margin: 0 auto; padding: 24px 28px 80px; }}
  header.top {{ background: var(--card); border: 1px solid var(--line);
                padding: 18px 22px; border-radius: 12px; margin-bottom: 20px; }}
  header.top h1 {{ margin: 0 0 6px; font-size: 22px; }}
  .meta {{ display: flex; gap: 16px; align-items: center; font-size: 13px;
           color: var(--muted); flex-wrap: wrap; }}
  button.regen {{ background: var(--accent); color: white; border: none;
                  padding: 8px 16px; border-radius: 6px; cursor: pointer;
                  font-size: 14px; font-weight: 600; }}
  button.regen:hover {{ background: var(--accent-dark); }}
  .topics-preview {{ margin: 10px 0 0 18px; font-size: 13.5px; color: #333; }}
  .topics-preview li {{ margin: 3px 0; }}
  .controls {{ margin-top: 8px; font-size: 12px; color: var(--muted); }}
  .controls a {{ color: var(--muted); }}
  section.emotion {{ background: var(--card); border: 1px solid var(--line);
                     border-radius: 12px; margin-bottom: 14px; overflow: hidden; }}
  section.emotion > summary {{ padding: 13px 20px; cursor: pointer;
                               font-size: 17px; font-weight: 700;
                               list-style: none; background: var(--summary-bg);
                               outline: none; }}
  section.emotion > summary::-webkit-details-marker {{ display: none; }}
  section.emotion > summary::before {{ content: "▸"; display: inline-block;
                                        width: 14px; color: #999;
                                        transition: transform 0.15s; }}
  section.emotion[open] > summary::before {{ content: "▾"; }}
  .topic {{ padding: 14px 22px 18px; border-top: 1px solid #eee; }}
  .topic h3 {{ margin: 0 0 8px; font-size: 14px; color: var(--topic);
               font-weight: 600; font-style: italic; }}
  .story {{ white-space: pre-wrap; font-size: 15.5px; color: #222; }}
  @media (max-width: 820px) {{
    .layout {{ grid-template-columns: 1fr; }}
    nav.sidebar {{ position: static; height: auto; border-right: none;
                    border-bottom: 1px solid var(--line); }}
  }}
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar">
    <h2>Emotions ({n_emotions})</h2>
    <input id="filter" type="search" placeholder="Filter…" oninput="filterEmotions()">
    <ul id="toc">
      {toc_items}
    </ul>
  </nav>
  <main>
    <header class="top">
      <h1>Emotion Story Browser</h1>
      <div class="meta">
        <span><strong>{n_emotions}</strong> emotions · <strong>{n_topics}</strong> topics · seed <code>{seed}</code></span>
        <button class="regen" onclick="regenerate()">Regenerate</button>
      </div>
      <div class="controls">
        <a href="#" onclick="toggleAll(true);return false">Expand all</a> ·
        <a href="#" onclick="toggleAll(false);return false">Collapse all</a> ·
        <span>Data: <code>{data_path}</code></span>
      </div>
      <ol class="topics-preview">
        {topic_list}
      </ol>
    </header>
    {emotion_sections}
  </main>
</div>
<script>
function regenerate() {{
  const s = Math.floor(Math.random() * 1e9);
  window.location.href = '/?seed=' + s;
}}
function filterEmotions() {{
  const q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('#toc li').forEach(li => {{
    li.style.display = li.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function toggleAll(open) {{
  document.querySelectorAll('section.emotion').forEach(d => d.open = open);
}}
document.querySelectorAll('nav.sidebar a').forEach(a => {{
  a.addEventListener('click', () => {{
    const el = document.getElementById(a.getAttribute('href').slice(1));
    if (el) el.open = true;
  }});
}});
window.addEventListener('hashchange', () => {{
  const el = document.getElementById(location.hash.slice(1));
  if (el) el.open = true;
}});
</script>
</body>
</html>
"""


def render_page(data: dict, emotions: list, all_topics: list, n_topics: int,
                seed: int, data_path: str) -> str:
    """Sample topics & stories deterministically from ``seed`` and return HTML."""
    rng = random.Random(seed)
    topics = rng.sample(all_topics, n_topics)

    toc_items = "\n".join(
        f'<li><a href="#emo-{i}">{html.escape(e)}</a></li>'
        for i, e in enumerate(emotions)
    )
    topic_list = "\n".join(f"<li>{html.escape(t)}</li>" for t in topics)

    emotion_blocks = []
    for i, emotion in enumerate(emotions):
        topic_html = []
        for topic in topics:
            stories = data[emotion][topic]
            story = rng.choice(stories)
            topic_html.append(
                f'<div class="topic">'
                f'<h3>{html.escape(topic)}</h3>'
                f'<div class="story">{html.escape(story)}</div>'
                f'</div>'
            )
        emotion_blocks.append(
            f'<details class="emotion" id="emo-{i}">'
            f'<summary>{html.escape(emotion)}</summary>'
            f'{"".join(topic_html)}'
            f'</details>'
        )

    return PAGE_TEMPLATE.format(
        n_emotions=len(emotions),
        n_topics=n_topics,
        seed=seed,
        data_path=html.escape(data_path),
        toc_items=toc_items,
        topic_list=topic_list,
        emotion_sections="\n".join(emotion_blocks),
    )


def _build_handler(data: dict, emotions: list, all_topics: list,
                   n_topics: int, data_path: str, default_seed: int):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[view_stories] {self.address_string()} - " + fmt % args)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path not in ("/", "/index.html"):
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            seed_str = qs.get("seed", [str(default_seed)])[0]
            try:
                seed = int(seed_str)
            except ValueError:
                seed = random.randint(0, 10**9)
            html_body = render_page(
                data, emotions, all_topics, n_topics, seed, data_path
            )
            payload = html_body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main(
    model: str = "gemma2_9b",
    version: str = "v0",
    data_path: str | None = None,
    stories_root: str = "results/stories",
    host: str = "127.0.0.1",
    port: int = 5000,
    n_topics: int = 5,
    max_emotions: int | None = None,
    seed: int = 0,
):
    """Launch the emotion-story browser.

    By default loads ``{stories_root}/{model}_{version}/emotion_to_topic_to_stories.json``.
    Pass ``--data-path`` to override that with an explicit file path.

    Args:
        model: Model folder stem (e.g. ``gemma2_9b``).
        version: Version suffix; resolves to folder ``{model}_{version}``.
        data_path: Explicit path override; bypasses model/version resolution.
        stories_root: Root directory containing per-(model,version) story folders.
        host: Interface to bind. Use ``0.0.0.0`` for LAN access.
        port: TCP port to bind.
        n_topics: Number of topics to sample per page.
        max_emotions: Optional cap on emotions rendered (useful for debugging).
        seed: Default seed used on first page load (regenerate picks a new one).
    """
    if data_path is None:
        path = Path(stories_root) / f"{model}_{version}" / "emotion_to_topic_to_stories.json"
    else:
        path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Story file not found: {path.resolve()}")

    print(f"[view_stories] loading {path} ...")
    with path.open() as f:
        data = json.load(f)

    emotions = sorted(data.keys())
    if max_emotions is not None:
        emotions = emotions[:max_emotions]
    all_topics = sorted(data[emotions[0]].keys())
    print(f"[view_stories] {len(emotions)} emotions, {len(all_topics)} topics")

    handler_cls = _build_handler(
        data, emotions, all_topics, n_topics, str(path), seed
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}/"
    print(f"[view_stories] serving at {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[view_stories] shutting down")
        server.server_close()


if __name__ == "__main__":
    fire.Fire(main)
