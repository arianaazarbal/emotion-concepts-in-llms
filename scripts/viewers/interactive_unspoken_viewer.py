"""Interactive web viewer for Bob/Amy unspoken-emotion per-token scores.

Companion to :mod:`scripts.experiments.probe_unspoken_emotions`. Pick a (backstory, chat)
pair and a probe emotion; the page renders each token as a color chip whose
shade encodes the probe score at the given layer.

The vector settings (version, layer, start_at_nth_token, denoised, metric)
are locked in at launch via CLI flags — they're not adjustable in the UI.

Usage:
    uv run python -m scripts.viewers.interactive_unspoken_viewer \\
        --model gemma2_9b --port 8080
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import fire
import torch

from src.utils.emotion_probe import (
    default_layer,
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import load_json


STATE = {
    "model": None,
    "tokenizer": None,
    "model_short": None,
    "version": "",
    "layer": None,
    "cosine_sim": False,
    "start_at_nth_token": 50,
    "denoised": False,
    "emotions": [],
    "vec_by_layer": None,
    "backstories": [],
    "chats": [],
    "backstory_names": [],
    "chat_names": [],
    "texts": {},
    "score_cache": {},
}
PROBE_LOCK = threading.Lock()


def build_text_and_spans(backstory: str, chat_messages: list):
    """Construct the concatenated prompt and per-line char spans. Mirrors probe_unspoken_emotions."""
    joined = [f"{t['speaker']}: {t['text']}" for t in chat_messages]
    chat_text = "\n".join(joined)
    text = backstory + "\n\n" + chat_text
    spans = []
    cursor = len(backstory) + 2
    for line, turn in zip(joined, chat_messages):
        start = cursor
        end = cursor + len(line)
        spans.append({"speaker": turn["speaker"], "start": start, "end": end})
        cursor = end + 1
    return text, spans


def score_pair(bi: int, ci: int, probe_emotion: str):
    """Return per-token scores for (backstory bi, chat ci) at the probe emotion."""
    key = (bi, ci)
    cache = STATE["score_cache"]
    if key not in cache:
        text, spans = STATE["texts"][key]
        enc = STATE["tokenizer"](
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = enc["offset_mapping"][0].tolist()
        with PROBE_LOCK:
            result = emotion_probe(
                texts=[text],
                model=STATE["model"],
                tokenizer=STATE["tokenizer"],
                emotion_vectors_by_layer=STATE["vec_by_layer"],
                emotions=STATE["emotions"],
                layers=[STATE["layer"]],
                aggregation="none",
                batch_size=1,
                max_length=4096,
                cosine_sim=STATE["cosine_sim"],
            )
        per_token = result.scores[STATE["layer"]][0].numpy()  # [T, E]
        raw_tokens = result.tokens[0]
        display = [
            STATE["tokenizer"].convert_tokens_to_string([t]) or t for t in raw_tokens
        ]
        cache[key] = {
            "text": text,
            "spans": spans,
            "offsets": offsets[: per_token.shape[0]],
            "tokens": display,
            "raw_tokens": raw_tokens,
            "per_token": per_token,  # np array [T, E]
        }
    entry = cache[key]
    e_idx = STATE["emotions"].index(probe_emotion)
    scores = entry["per_token"][:, e_idx].tolist()
    return {
        "tokens": entry["tokens"],
        "raw_tokens": entry["raw_tokens"],
        "scores": scores,
        "offsets": entry["offsets"],
        "spans": entry["spans"],
        "backstory_end": len(STATE["backstories"][bi]),
    }


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Unspoken Emotion Viewer</title>
<style>
 body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #222; max-width: 1100px; }
 h1 { font-size: 20px; margin-bottom: 4px; }
 .sub { color: #666; font-size: 13px; margin-bottom: 20px; }
 .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
 label { font-size: 13px; }
 select, input[type=text] { padding: 4px 6px; font-size: 14px; font-family: inherit; }
 button { padding: 6px 14px; font-size: 14px; cursor: pointer; }
 .tokens { padding: 14px; background: #fff; border: 1px solid #ddd; border-radius: 4px;
           font-size: 15px; line-height: 2.3; min-height: 60px; white-space: pre-wrap; }
 .tok { padding: 2px 5px; border-radius: 4px; margin: 1px; display: inline-block; cursor: default; }
 .backstory-sep { display: block; height: 8px; }
 .meta { color: #555; font-size: 12px; margin: 6px 0 10px 0; font-family: monospace; }
 .err { color: #b00; font-family: monospace; }
 fieldset { border: 1px solid #eee; border-radius: 4px; margin-bottom: 16px; padding: 10px 14px; }
 legend { font-size: 13px; font-weight: bold; padding: 0 6px; color: #444; }
 .speaker-tag { font-weight: bold; color: #444; }
</style>
</head>
<body>
<h1>Unspoken Emotion Viewer (Bob / Amy)</h1>
<div class="sub">model: <b id="model_name"></b> · layer: <b id="layer_name"></b> · metric: <b id="metric_name"></b> · version: <b id="version_name"></b></div>

<fieldset>
<legend>Pick context + emotion</legend>
<div class="row">
 <label>Backstory: <select id="backstory"></select></label>
 <label>Chat: <select id="chat"></select></label>
 <label>Emotion: <select id="emotion"></select></label>
 <button id="score_btn">Render</button>
</div>
<div class="meta" id="score_meta">&nbsp;</div>
</fieldset>

<fieldset>
<legend>Per-token scores</legend>
<div class="tokens" id="tokens_view">(pick context + emotion and click Render)</div>
</fieldset>

<script>
const CFG = __CFG_JSON__;
document.getElementById("model_name").textContent = CFG.model;
document.getElementById("layer_name").textContent = CFG.layer;
document.getElementById("metric_name").textContent = CFG.metric;
document.getElementById("version_name").textContent = CFG.version || "(bare)";

function fillIdx(id, items) {
  const el = document.getElementById(id);
  items.forEach((name, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = name;
    el.appendChild(o);
  });
}
function fillName(id, items) {
  const el = document.getElementById(id);
  items.forEach(name => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    el.appendChild(o);
  });
}
fillIdx("backstory", CFG.backstory_names);
fillIdx("chat", CFG.chat_names);
fillName("emotion", CFG.emotions);

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function diverging(t) {
  const mid = [247, 247, 247];
  const pos = [178, 24, 43];
  const neg = [33, 102, 172];
  t = Math.max(-1, Math.min(1, t));
  const a = Math.abs(t);
  const target = t >= 0 ? pos : neg;
  return [
    Math.round(mid[0] + a * (target[0] - mid[0])),
    Math.round(mid[1] + a * (target[1] - mid[1])),
    Math.round(mid[2] + a * (target[2] - mid[2])),
  ];
}

document.getElementById("score_btn").onclick = async () => {
  const body = {
    backstory_idx: Number(document.getElementById("backstory").value),
    chat_idx: Number(document.getElementById("chat").value),
    emotion: document.getElementById("emotion").value,
  };
  const meta = document.getElementById("score_meta");
  meta.textContent = "Scoring...";
  try {
    const r = await postJSON("/api/score", body);
    const view = document.getElementById("tokens_view");
    view.innerHTML = "";
    const vmax = r.scores.reduce((m, s) => Math.max(m, Math.abs(s)), 1e-9);
    for (let i = 0; i < r.tokens.length; i++) {
      const tokStart = r.offsets[i] ? r.offsets[i][0] : 0;
      if (i > 0 && r.offsets[i-1] && r.offsets[i-1][1] <= r.backstory_end && tokStart >= r.backstory_end) {
        const br = document.createElement("span");
        br.className = "backstory-sep";
        br.innerHTML = "<br><hr style='border:none;border-top:1px dashed #bbb;margin:6px 0;'/>";
        view.appendChild(br);
      }
      const tok = r.tokens[i];
      const score = r.scores[i];
      const [rr, gg, bb] = diverging(score / vmax);
      const span = document.createElement("span");
      span.className = "tok";
      span.style.background = `rgb(${rr},${gg},${bb})`;
      const lum = 0.299 * rr + 0.587 * gg + 0.114 * bb;
      span.style.color = lum > 160 ? "#111" : "#fff";
      span.textContent = tok === "" ? " " : tok;
      span.title = `raw="${r.raw_tokens[i]}"  score=${score.toFixed(4)}  char=[${r.offsets[i][0]},${r.offsets[i][1]})`;
      view.appendChild(span);
    }
    const minS = Math.min(...r.scores), maxS = Math.max(...r.scores);
    meta.textContent =
      `emotion="${body.emotion}"  range=[${minS.toFixed(3)}, ${maxS.toFixed(3)}]  |vmax|=${vmax.toFixed(3)}  tokens=${r.tokens.length}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, ctype, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw or b"{}")

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                cfg = {
                    "model": STATE["model_short"],
                    "layer": STATE["layer"],
                    "metric": "cosine_sim" if STATE["cosine_sim"] else "dot_product",
                    "version": STATE["version"],
                    "backstory_names": STATE["backstory_names"],
                    "chat_names": STATE["chat_names"],
                    "emotions": STATE["emotions"],
                }
                html = INDEX_HTML.replace("__CFG_JSON__", json.dumps(cfg))
                return self._send(200, "text/html; charset=utf-8", html)
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] GET error: {e}")
            return self._send(400, "text/plain", str(e))

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
            if parsed.path == "/api/score":
                bi = int(body["backstory_idx"])
                ci = int(body["chat_idx"])
                emo = body["emotion"]
                return self._send(
                    200, "application/json", json.dumps(score_pair(bi, ci, emo))
                )
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] POST error: {e}")
            return self._send(400, "text/plain", str(e))


def main(
    model: str = "gemma2_9b",
    port: int = 8080,
    host: str = "127.0.0.1",
    version: str = "v0",
    layer: int = None,
    start_at_nth_token: int = 50,
    cosine_sim: bool = False,
    denoised: bool = False,
    chats_path: str = None,
    backstories_path: str = None,
    seed: int = 0,
    v1: bool = False,
):
    """Launch the Bob/Amy per-token emotion viewer.

    Args:
        model: Short model name in data/model_names.json.
        port: HTTP port.
        host: Bind interface. Use 0.0.0.0 for remote access.
        version: Emotion-vector version tag.
        layer: Probe layer; defaults to the model's ~2/3 layer.
        start_at_nth_token: Which saved vector variant to load.
        cosine_sim: Use cosine similarity instead of dot product.
        denoised: Load PCA-denoised vectors.
        chats_path: Path to the chats JSON.
        backstories_path: Path to the backstories JSON.
        seed: Torch RNG seed (parity).
    """
    torch.manual_seed(seed)

    if chats_path is None:
        chats_path = f"data/bob_amy_chats{'_v1' if v1 else ''}.json"
    if backstories_path is None:
        backstories_path = f"data/bob_amy_backstories{'_v1' if v1 else ''}.json"
    chats_raw = load_json(chats_path)
    backstories_raw = load_json(backstories_path)
    STATE["chat_names"] = [c["name"] for c in chats_raw]
    STATE["chats"] = [c["messages"] for c in chats_raw]
    STATE["backstory_names"] = [b["name"] for b in backstories_raw]
    STATE["backstories"] = [b["text"] for b in backstories_raw]

    print(f"[viewer] loading model={model}")
    mdl, tok = load_model_and_tokenizer(model)
    emotions, vec_by_layer = load_emotion_vectors(
        model,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        version=version,
    )
    if layer is None:
        layer = default_layer(mdl)
    if layer not in vec_by_layer:
        raise ValueError(
            f"Layer {layer} not in loaded vectors. Available: {sorted(vec_by_layer.keys())}"
        )

    STATE.update(
        {
            "model": mdl,
            "tokenizer": tok,
            "model_short": model,
            "version": version,
            "layer": int(layer),
            "cosine_sim": bool(cosine_sim),
            "start_at_nth_token": int(start_at_nth_token),
            "denoised": bool(denoised),
            "emotions": list(emotions),
            "vec_by_layer": vec_by_layer,
        }
    )

    for bi, bs in enumerate(STATE["backstories"]):
        for ci, chat in enumerate(STATE["chats"]):
            STATE["texts"][(bi, ci)] = build_text_and_spans(bs, chat)

    print(
        f"[viewer] layer={layer} version={version!r} start_at={start_at_nth_token} "
        f"denoised={denoised} cosine_sim={cosine_sim} emotions={len(emotions)}"
    )
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[viewer] serving on http://{host}:{port}   (ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] bye")


if __name__ == "__main__":
    fire.Fire(main)
