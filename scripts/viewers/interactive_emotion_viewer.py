"""Interactive web viewer for per-token emotion activations.

Launches a local HTTP server that lets you:
 1. Sample a random story for a given story-emotion (and version on disk).
 2. Probe that text with any emotion vector and render each token as a
    colored chip whose color encodes dot product or cosine similarity.

The extraction variant (version, layer, start_at_nth_token, denoised) is
switchable in the UI so a fixed text + fixed probe-emotion can be compared
across versions without resampling.

Usage:
    python -m scripts.viewers.interactive_emotion_viewer --model gemma2_9b --port 8080
"""

import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv(override=True)

import fire  # noqa: E402
import torch  # noqa: E402

from src.utils.emotion_probe import (  # noqa: E402
    default_layer,
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import load_json  # noqa: E402

MODEL = None
TOKENIZER = None
MODEL_SHORT = None
DEFAULT_METRIC = "dot"
DEFAULT_LAYER = None
VERSION_LIST: list = []

STORIES_CACHE: dict = {}
VECTORS_CACHE: dict = {}
LOAD_LOCK = threading.Lock()
PROBE_LOCK = threading.Lock()


def discover_versions(model_short: str) -> list:
    """Scan the cached emotion-vectors root and return version suffixes that actually contain vector files."""
    from src.utils.vector_cache import cached_emotion_vectors_root
    root = cached_emotion_vectors_root()
    if not os.path.isdir(root):
        return []
    prefix = f"{model_short}_"
    found = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        has_vecs = False
        for dirpath, _, files in os.walk(full):
            if any(f.endswith("_vectors.pt") for f in files):
                has_vecs = True
                break
        if not has_vecs:
            continue
        if name == model_short:
            found.append("")
        elif name.startswith(prefix):
            found.append(name[len(prefix) :])
    return found


def get_stories(version: str) -> dict:
    """Lazy-load + cache the emotion-to-stories dict for a given version."""
    with LOAD_LOCK:
        if version in STORIES_CACHE:
            return STORIES_CACHE[version]
        tag = f"{MODEL_SHORT}_{version}" if version else MODEL_SHORT
        path = f"results/stories/{tag}/emotion_to_stories.json"
        STORIES_CACHE[version] = load_json(path) if os.path.exists(path) else {}
        return STORIES_CACHE[version]


def get_vectors(version: str, start_at_nth_token: int, denoised: bool):
    """Lazy-load + cache emotion vectors for a (version, start_at, denoised) combo."""
    key = (version, int(start_at_nth_token), bool(denoised))
    with LOAD_LOCK:
        if key in VECTORS_CACHE:
            return VECTORS_CACHE[key]
        emotions, vectors_by_layer = load_emotion_vectors(
            MODEL_SHORT,
            start_at_nth_token=int(start_at_nth_token),
            denoised=bool(denoised),
            version=version,
        )
        VECTORS_CACHE[key] = (emotions, vectors_by_layer)
        return emotions, vectors_by_layer


def score_text(
    text, probe_emotion, version, metric, layer, start_at_nth_token, denoised,
):
    """Run the emotion probe on a single text for a single emotion and return tokens + scores."""
    emotions, vecs_by_layer = get_vectors(version, start_at_nth_token, denoised)
    if probe_emotion not in emotions:
        raise ValueError(
            f"Emotion '{probe_emotion}' not in vectors (version='{version}'). "
            f"Available ({len(emotions)}): {emotions[:10]}..."
        )
    if layer is None:
        layer = DEFAULT_LAYER if DEFAULT_LAYER is not None else default_layer(MODEL)
    if layer not in vecs_by_layer:
        raise ValueError(
            f"Layer {layer} not available. Have: {sorted(vecs_by_layer.keys())}"
        )

    cosine = metric == "cosine"

    def _run():
        return emotion_probe(
            texts=text,
            model=MODEL,
            tokenizer=TOKENIZER,
            emotion_vectors_by_layer={layer: vecs_by_layer[layer]},
            emotions=emotions,
            layers=[layer],
            selected_emotions=[probe_emotion],
            aggregation="none",
            cosine_sim=cosine,
        )

    with PROBE_LOCK:
        result = _run()
    raw_tokens = result.tokens[0]
    scores = result.scores[layer][0][:, 0].tolist()
    display_tokens = [TOKENIZER.convert_tokens_to_string([t]) or t for t in raw_tokens]
    return {
        "tokens": display_tokens,
        "raw_tokens": raw_tokens,
        "scores": scores,
        "layer": int(layer),
        "metric": metric,
        "version": version,
        "start_at_nth_token": int(start_at_nth_token),
        "denoised": bool(denoised),
    }


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Emotion Activation Viewer</title>
<style>
 body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #222; max-width: 1100px; }
 h1 { font-size: 20px; margin-bottom: 4px; }
 .sub { color: #666; font-size: 13px; margin-bottom: 20px; }
 .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
 label { font-size: 13px; }
 input[type=text], input[type=number], select, textarea {
   padding: 4px 6px; font-size: 14px; font-family: inherit;
 }
 textarea { width: 100%; box-sizing: border-box; min-height: 140px; }
 button { padding: 6px 14px; font-size: 14px; cursor: pointer; }
 .story-box { margin-bottom: 20px; }
 .tokens { padding: 14px; background: #fff; border: 1px solid #ddd; border-radius: 4px;
           font-size: 15px; line-height: 2.3; min-height: 60px; }
 .tok { padding: 2px 5px; border-radius: 4px; margin: 1px; display: inline-block; cursor: default; }
 .meta { color: #555; font-size: 12px; margin: 6px 0 10px 0; font-family: monospace; }
 .err { color: #b00; font-family: monospace; }
 fieldset { border: 1px solid #eee; border-radius: 4px; margin-bottom: 16px; padding: 10px 14px; }
 legend { font-size: 13px; font-weight: bold; padding: 0 6px; color: #444; }
</style>
</head>
<body>
<h1>Emotion Activation Viewer</h1>
<div class="sub">
  model: <b id="model_name"></b>
  · versions on disk: <span id="version_info"></span>
</div>

<fieldset>
<legend>1. Sample a story</legend>
<div class="row">
 <label>Story emotion: <input id="story_emotion" type="text" value="angry" list="story_emotions"/></label>
 <datalist id="story_emotions"></datalist>
 <label>Stories version:
  <select id="story_version"></select>
 </label>
 <label>Seed (blank = random): <input id="story_seed" type="number" value="" style="width:90px"/></label>
 <button id="sample_btn">Sample story</button>
</div>
<div class="meta" id="story_meta">&nbsp;</div>
<div class="story-box">
 <textarea id="story_text" placeholder="Sampled story will appear here — or type/paste your own text..."></textarea>
</div>
</fieldset>

<fieldset>
<legend>2. Probe + visualize</legend>
<div class="row">
 <label>Probe emotion: <input id="probe_emotion" type="text" value="angry" list="probe_emotions"/></label>
 <datalist id="probe_emotions"></datalist>
 <label>Vectors version: <select id="probe_version"></select></label>
 <label>Metric:
  <select id="metric">
   <option value="dot">dot product</option>
   <option value="cosine">cosine similarity</option>
  </select>
 </label>
 <label>Layer (blank=default): <input id="layer" type="number" value="" style="width:80px"/></label>
 <label>start_at_nth_token: <input id="start_at" type="number" value="0" style="width:80px"/></label>
 <label><input id="denoised" type="checkbox"/> denoised</label>
 <button id="score_btn">Score</button>
</div>
<div class="meta" id="score_meta">&nbsp;</div>
<div class="tokens" id="tokens_view">(no scores yet)</div>
</fieldset>

<script>
const VERSIONS = __VERSIONS_JSON__;
const DEFAULT_METRIC = __DEFAULT_METRIC__;
const MODEL_NAME = __MODEL_NAME__;
document.getElementById("model_name").textContent = MODEL_NAME;
document.getElementById("version_info").textContent =
  VERSIONS.map(v => v === "" ? "(bare)" : v).join(", ");

function fillSelect(id) {
  const el = document.getElementById(id);
  VERSIONS.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v === "" ? "(bare)" : v;
    el.appendChild(o);
  });
}
fillSelect("story_version");
fillSelect("probe_version");
document.getElementById("metric").value = DEFAULT_METRIC;

async function populateStoryEmotions() {
  const version = document.getElementById("story_version").value;
  const r = await fetch(`/api/story_emotions?version=${encodeURIComponent(version)}`);
  const data = await r.json();
  const dl = document.getElementById("story_emotions");
  dl.innerHTML = "";
  (data.emotions || []).forEach(e => {
    const o = document.createElement("option"); o.value = e; dl.appendChild(o);
  });
}

async function populateProbeEmotions() {
  const version = document.getElementById("probe_version").value;
  const startAt = document.getElementById("start_at").value || "0";
  const denoised = document.getElementById("denoised").checked;
  const url = `/api/vector_emotions?version=${encodeURIComponent(version)}` +
              `&start_at=${encodeURIComponent(startAt)}&denoised=${denoised}`;
  const r = await fetch(url);
  const data = await r.json();
  const dl = document.getElementById("probe_emotions");
  dl.innerHTML = "";
  (data.emotions || []).forEach(e => {
    const o = document.createElement("option"); o.value = e; dl.appendChild(o);
  });
}

document.getElementById("story_version").addEventListener("change", populateStoryEmotions);
["probe_version", "start_at", "denoised"].forEach(id =>
  document.getElementById(id).addEventListener("change", populateProbeEmotions));
populateStoryEmotions();
populateProbeEmotions();

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

document.getElementById("sample_btn").onclick = async () => {
  const body = {
    emotion: document.getElementById("story_emotion").value.trim(),
    version: document.getElementById("story_version").value,
    seed: document.getElementById("story_seed").value === "" ? null
          : Number(document.getElementById("story_seed").value),
  };
  const meta = document.getElementById("story_meta");
  meta.textContent = "Sampling...";
  try {
    const r = await postJSON("/api/sample", body);
    document.getElementById("story_text").value = r.story;
    meta.textContent = `emotion="${r.emotion}"  version="${r.version || "(bare)"}"  idx=${r.idx}/${r.total}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
};

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
  const text = document.getElementById("story_text").value;
  if (!text.trim()) {
    document.getElementById("score_meta").innerHTML =
      '<span class="err">Text is empty — sample or paste a story first.</span>';
    return;
  }
  const body = {
    text: text,
    emotion: document.getElementById("probe_emotion").value.trim(),
    version: document.getElementById("probe_version").value,
    metric: document.getElementById("metric").value,
    layer: document.getElementById("layer").value === "" ? null
           : Number(document.getElementById("layer").value),
    start_at_nth_token: Number(document.getElementById("start_at").value),
    denoised: document.getElementById("denoised").checked,
  };
  const meta = document.getElementById("score_meta");
  meta.textContent = "Scoring...";
  try {
    const r = await postJSON("/api/score", body);
    const view = document.getElementById("tokens_view");
    view.innerHTML = "";
    const vmax = r.scores.reduce((m, s) => Math.max(m, Math.abs(s)), 1e-9);
    r.tokens.forEach((tok, i) => {
      const span = document.createElement("span");
      const score = r.scores[i];
      const [rr, gg, bb] = diverging(score / vmax);
      span.className = "tok";
      span.style.background = `rgb(${rr},${gg},${bb})`;
      const lum = 0.299 * rr + 0.587 * gg + 0.114 * bb;
      span.style.color = lum > 160 ? "#111" : "#fff";
      span.textContent = tok === "" ? "\u00A0" : tok;
      span.title = `raw="${r.raw_tokens[i]}"  ${r.metric}=${score.toFixed(4)}`;
      view.appendChild(span);
    });
    const minS = Math.min(...r.scores), maxS = Math.max(...r.scores);
    meta.textContent =
      `probe="${body.emotion}"  version="${r.version || "(bare)"}"  metric=${r.metric}  ` +
      `layer=${r.layer}  start_at=${r.start_at_nth_token}  denoised=${r.denoised}  ` +
      `range=[${minS.toFixed(3)}, ${maxS.toFixed(3)}]  |vmax|=${vmax.toFixed(3)}`;
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
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                html = (
                    INDEX_HTML.replace("__VERSIONS_JSON__", json.dumps(VERSION_LIST))
                    .replace("__DEFAULT_METRIC__", json.dumps(DEFAULT_METRIC))
                    .replace("__MODEL_NAME__", json.dumps(MODEL_SHORT))
                )
                return self._send(200, "text/html; charset=utf-8", html)
            if path == "/api/story_emotions":
                version = qs.get("version", [""])[0]
                stories = get_stories(version)
                return self._send(
                    200,
                    "application/json",
                    json.dumps({"emotions": sorted(stories.keys())}),
                )
            if path == "/api/vector_emotions":
                version = qs.get("version", [""])[0]
                start_at = int(qs.get("start_at", ["0"])[0])
                denoised = qs.get("denoised", ["false"])[0].lower() == "true"
                try:
                    emotions, _ = get_vectors(version, start_at, denoised)
                except FileNotFoundError:
                    emotions = []
                return self._send(
                    200, "application/json", json.dumps({"emotions": emotions})
                )
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] GET {path} error: {e}")
            return self._send(400, "text/plain", str(e))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json()
            if path == "/api/sample":
                emotion = body["emotion"]
                version = body.get("version", "")
                seed = body.get("seed", None)
                stories = get_stories(version)
                if emotion not in stories or not stories[emotion]:
                    raise ValueError(
                        f"No stories for emotion='{emotion}' at version='{version}'. "
                        f"Available: {sorted(stories.keys())[:20]}"
                    )
                pool = stories[emotion]
                rng = random.Random(seed) if seed is not None else random
                idx = rng.randrange(len(pool))
                return self._send(
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "story": pool[idx],
                            "emotion": emotion,
                            "version": version,
                            "idx": idx,
                            "total": len(pool),
                        }
                    ),
                )
            if path == "/api/score":
                result = score_text(
                    text=body["text"],
                    probe_emotion=body["emotion"],
                    version=body.get("version", ""),
                    metric=body.get("metric", DEFAULT_METRIC),
                    layer=body.get("layer", None),
                    start_at_nth_token=body.get("start_at_nth_token", 0),
                    denoised=body.get("denoised", False),
                )
                return self._send(200, "application/json", json.dumps(result))
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] POST {path} error: {e}")
            return self._send(400, "text/plain", str(e))


def main(
    model: str = "gemma2_9b",
    port: int = 8080,
    host: str = "127.0.0.1",
    metric: str = "dot",
    layer: int = None,
    seed: int = 0,
):
    """Launch the interactive emotion activation viewer.

    Args:
        model: Short model name registered in data/model_names.json.
        port: HTTP port to bind.
        host: Bind interface. Use 0.0.0.0 for remote access.
        metric: Default metric: "dot" or "cosine". Overridable in the UI.
        layer: Default probe layer. None = ~2/3 through the model. Overridable in the UI.
        seed: RNG seed for initial randomness in "Sample story" when no per-request seed is set.
    """
    global MODEL, TOKENIZER, MODEL_SHORT
    global DEFAULT_METRIC, DEFAULT_LAYER, VERSION_LIST
    if metric not in ("dot", "cosine"):
        raise ValueError(f"--metric must be 'dot' or 'cosine', got {metric!r}")

    random.seed(seed)
    torch.manual_seed(seed)

    MODEL_SHORT = model
    DEFAULT_METRIC = metric
    VERSION_LIST = discover_versions(model)
    if not VERSION_LIST:
        raise FileNotFoundError(
            f"No emotion-vector dirs found under results/emotion_vectors/ for model='{model}'."
        )
    print(f"[viewer] discovered versions for {model}: {VERSION_LIST}")
    print(f"[viewer] loading model={model} (this may take a moment)...")
    MODEL, TOKENIZER = load_model_and_tokenizer(model)
    DEFAULT_LAYER = layer if layer is not None else default_layer(MODEL)
    print(f"[viewer] default layer={DEFAULT_LAYER}, default metric={DEFAULT_METRIC}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[viewer] serving on http://{host}:{port}   (ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] bye")


if __name__ == "__main__":
    fire.Fire(main)
