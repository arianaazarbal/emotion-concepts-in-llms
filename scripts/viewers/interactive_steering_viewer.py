"""Interactive web viewer for emotion-vector steered generation.

Launches a local HTTP server that lets you:
 1. Sample a random story for a given story-emotion (and version on disk),
    or type / paste your own text as the user prompt.
 2. Generate continuations from the model with a scaled emotion vector added
    to the residual stream at chosen layer(s). Optionally also generate a
    coeff=0 baseline with the same seed for side-by-side comparison.

Vector source knobs (version, layer, start_at_nth_token, denoised, normalize_steering_vector)
match interactive_emotion_viewer; steering knobs (coeff, steer_positions) plus
generation knobs (max_new_tokens, temperature, top_p, do_sample, N, seed) are
added. Everything is switchable in the UI so a fixed prompt + steering vector
can be compared across versions / layers / coefficients without reloading.

Usage:
    python -m scripts.viewers.interactive_steering_viewer --model gemma2_9b --port 8081
"""

import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import fire
import torch
from dotenv import load_dotenv

load_dotenv()

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.generate_with_steering import generate_with_steering
from src.utils.utils import load_json

MODEL = None
TOKENIZER = None
MODEL_SHORT = None
DEFAULT_LAYER = None
VERSION_LIST: list = []

STORIES_CACHE: dict = {}
VECTORS_CACHE: dict = {}
LOAD_LOCK = threading.Lock()
GEN_LOCK = threading.Lock()


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
            found.append(name[len(prefix):])
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


def run_generation(
    text,
    steer_emotion,
    version,
    layer,
    start_at_nth_token,
    denoised,
    normalize_steering_vector,
    coeff,
    steer_positions,
    max_new_tokens,
    temperature,
    top_p,
    do_sample,
    N,
    seed,
    apply_chat_template,
    include_baseline,
):
    """Run steered generation (+ optional baseline) on a single user prompt."""
    emotions, vecs_by_layer = get_vectors(version, start_at_nth_token, denoised)
    if steer_emotion not in emotions:
        raise ValueError(
            f"Emotion '{steer_emotion}' not in vectors (version='{version}'). "
            f"Available ({len(emotions)}): {emotions[:10]}..."
        )
    if layer is None:
        layer = DEFAULT_LAYER if DEFAULT_LAYER is not None else default_layer(MODEL)
    if layer not in vecs_by_layer:
        raise ValueError(f"Layer {layer} not available. Have: {sorted(vecs_by_layer.keys())}")

    common = dict(
        model=MODEL,
        tokenizer=TOKENIZER,
        prompts=[text],
        emotion_vectors_by_layer={layer: vecs_by_layer[layer]},
        emotions=emotions,
        selected_emotion=steer_emotion,
        layers=[layer],
        normalize_steering_vector=bool(normalize_steering_vector),
        N=int(N),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=bool(do_sample),
        batch_size=1,
        seed=int(seed),
        steer_positions=steer_positions,
        apply_chat_template=bool(apply_chat_template),
    )

    with GEN_LOCK:
        steered = generate_with_steering(coeff=float(coeff), **common)[0]
        baseline = generate_with_steering(coeff=0.0, **common)[0] if include_baseline else None

    return {
        "steered": steered,
        "baseline": baseline,
        "layer": int(layer),
        "version": version,
        "start_at_nth_token": int(start_at_nth_token),
        "denoised": bool(denoised),
        "normalize_steering_vector": bool(normalize_steering_vector),
        "coeff": float(coeff),
        "steer_positions": steer_positions,
        "N": int(N),
        "seed": int(seed),
    }


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Emotion Steering Viewer</title>
<style>
 body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #222; max-width: 1100px; }
 h1 { font-size: 20px; margin-bottom: 4px; }
 .sub { color: #666; font-size: 13px; margin-bottom: 20px; }
 .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
 label { font-size: 13px; }
 input[type=text], input[type=number], select, textarea {
   padding: 4px 6px; font-size: 14px; font-family: inherit;
 }
 textarea { width: 100%; box-sizing: border-box; min-height: 120px; }
 button { padding: 6px 14px; font-size: 14px; cursor: pointer; }
 .story-box { margin-bottom: 20px; }
 .meta { color: #555; font-size: 12px; margin: 6px 0 10px 0; font-family: monospace; }
 .err { color: #b00; font-family: monospace; }
 fieldset { border: 1px solid #eee; border-radius: 4px; margin-bottom: 16px; padding: 10px 14px; }
 legend { font-size: 13px; font-weight: bold; padding: 0 6px; color: #444; }
 .outputs { display: flex; gap: 14px; align-items: stretch; }
 .col { flex: 1 1 0; min-width: 0; }
 .col h3 { font-size: 14px; margin: 0 0 6px 0; color: #444; }
 .gens { display: flex; flex-direction: column; gap: 8px; }
 .gen { background: #fff; border: 1px solid #ddd; border-radius: 4px; padding: 10px 12px;
        font-size: 14px; line-height: 1.45; white-space: pre-wrap; }
 .gen.steered { border-left: 4px solid #b2182b; }
 .gen.baseline { border-left: 4px solid #2166ac; }
 .tag { display: inline-block; font-size: 11px; font-family: monospace; color: #666; margin-bottom: 4px; }
 .num { display: inline-block; min-width: 22px; font-weight: bold; color: #999; }
</style>
</head>
<body>
<h1>Emotion Steering Viewer</h1>
<div class="sub">model: <b id="model_name"></b> · versions on disk: <span id="version_info"></span></div>

<fieldset>
<legend>1. Choose source text (sample a story, or type your own)</legend>
<div class="row">
 <label>Story emotion: <input id="story_emotion" type="text" value="angry" list="story_emotions"/></label>
 <datalist id="story_emotions"></datalist>
 <label>Stories version: <select id="story_version"></select></label>
 <label>Seed (blank=random): <input id="story_seed" type="number" value="" style="width:90px"/></label>
 <button id="sample_btn">Sample story</button>
</div>
<div class="meta" id="story_meta">&nbsp;</div>
<div class="story-box">
 <textarea id="story_text" placeholder="Sampled story will appear here — or type/paste your own user prompt..."></textarea>
</div>
</fieldset>

<fieldset>
<legend>2. Steering vector</legend>
<div class="row">
 <label>Steer emotion: <input id="steer_emotion" type="text" value="angry" list="steer_emotions"/></label>
 <datalist id="steer_emotions"></datalist>
 <label>Vectors version: <select id="steer_version"></select></label>
 <label>Layer (blank=default): <input id="layer" type="number" value="" style="width:80px"/></label>
 <label>start_at_nth_token: <input id="start_at" type="number" value="0" style="width:80px"/></label>
 <label><input id="denoised" type="checkbox"/> denoised</label>
 <label><input id="normalize_steering_vector" type="checkbox" checked/> normalize_steering_vector (unit-length before coeff)</label>
</div>
<div class="row">
 <label>Coefficient: <input id="coeff" type="number" value="5" step="0.5" style="width:90px"/></label>
 <label>Positions:
  <select id="steer_positions">
   <option value="all">all (prompt + generation)</option>
   <option value="generation_only">generation_only</option>
  </select>
 </label>
 <label><input id="apply_chat_template" type="checkbox" checked/> wrap in chat template</label>
</div>
</fieldset>

<fieldset>
<legend>3. Generation</legend>
<div class="row">
 <label>N: <input id="N" type="number" value="2" min="1" max="8" style="width:60px"/></label>
 <label>max_new_tokens: <input id="max_new_tokens" type="number" value="160" style="width:80px"/></label>
 <label>temperature: <input id="temperature" type="number" value="1.0" step="0.1" style="width:70px"/></label>
 <label>top_p: <input id="top_p" type="number" value="1.0" step="0.05" style="width:70px"/></label>
 <label><input id="do_sample" type="checkbox" checked/> sample</label>
 <label>seed: <input id="gen_seed" type="number" value="0" style="width:80px"/></label>
 <label><input id="baseline" type="checkbox" checked/> also show unsteered baseline</label>
 <button id="gen_btn">Generate</button>
</div>
<div class="meta" id="gen_meta">&nbsp;</div>
<div class="outputs">
 <div class="col">
  <h3 id="steered_header">Steered</h3>
  <div class="gens" id="steered_view"><div class="gen">(no generations yet)</div></div>
 </div>
 <div class="col" id="baseline_col">
  <h3>Baseline (coeff=0, same seed)</h3>
  <div class="gens" id="baseline_view"><div class="gen">(no generations yet)</div></div>
 </div>
</div>
</fieldset>

<script>
const VERSIONS = __VERSIONS_JSON__;
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
fillSelect("steer_version");

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

async function populateSteerEmotions() {
  const version = document.getElementById("steer_version").value;
  const startAt = document.getElementById("start_at").value || "0";
  const denoised = document.getElementById("denoised").checked;
  const url = `/api/vector_emotions?version=${encodeURIComponent(version)}` +
              `&start_at=${encodeURIComponent(startAt)}&denoised=${denoised}`;
  const r = await fetch(url);
  const data = await r.json();
  const dl = document.getElementById("steer_emotions");
  dl.innerHTML = "";
  (data.emotions || []).forEach(e => {
    const o = document.createElement("option"); o.value = e; dl.appendChild(o);
  });
}

document.getElementById("story_version").addEventListener("change", populateStoryEmotions);
["steer_version", "start_at", "denoised"].forEach(id =>
  document.getElementById(id).addEventListener("change", populateSteerEmotions));
populateStoryEmotions();
populateSteerEmotions();

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

function renderGens(containerId, gens, klass) {
  const view = document.getElementById(containerId);
  view.innerHTML = "";
  if (!gens || gens.length === 0) {
    const d = document.createElement("div");
    d.className = "gen";
    d.textContent = "(no generations)";
    view.appendChild(d);
    return;
  }
  gens.forEach((g, i) => {
    const d = document.createElement("div");
    d.className = "gen " + klass;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.innerHTML = `<span class="num">#${i + 1}</span>`;
    d.appendChild(tag);
    const body = document.createElement("div");
    body.textContent = g == null ? "(null)" : g;
    d.appendChild(body);
    view.appendChild(d);
  });
}

document.getElementById("gen_btn").onclick = async () => {
  const text = document.getElementById("story_text").value;
  if (!text.trim()) {
    document.getElementById("gen_meta").innerHTML =
      '<span class="err">Text is empty — sample or paste a prompt first.</span>';
    return;
  }
  const body = {
    text: text,
    steer_emotion: document.getElementById("steer_emotion").value.trim(),
    version: document.getElementById("steer_version").value,
    layer: document.getElementById("layer").value === "" ? null
           : Number(document.getElementById("layer").value),
    start_at_nth_token: Number(document.getElementById("start_at").value),
    denoised: document.getElementById("denoised").checked,
    normalize_steering_vector: document.getElementById("normalize_steering_vector").checked,
    coeff: Number(document.getElementById("coeff").value),
    steer_positions: document.getElementById("steer_positions").value,
    apply_chat_template: document.getElementById("apply_chat_template").checked,
    max_new_tokens: Number(document.getElementById("max_new_tokens").value),
    temperature: Number(document.getElementById("temperature").value),
    top_p: Number(document.getElementById("top_p").value),
    do_sample: document.getElementById("do_sample").checked,
    N: Number(document.getElementById("N").value),
    seed: Number(document.getElementById("gen_seed").value),
    include_baseline: document.getElementById("baseline").checked,
  };
  const meta = document.getElementById("gen_meta");
  meta.textContent = "Generating...";
  document.getElementById("steered_view").innerHTML = '<div class="gen">...</div>';
  document.getElementById("baseline_view").innerHTML = '<div class="gen">...</div>';
  document.getElementById("baseline_col").style.display = body.include_baseline ? "" : "none";
  try {
    const r = await postJSON("/api/generate", body);
    document.getElementById("steered_header").textContent =
      `Steered  (emotion="${body.steer_emotion}"  coeff=${r.coeff})`;
    renderGens("steered_view", r.steered, "steered");
    if (r.baseline != null) {
      renderGens("baseline_view", r.baseline, "baseline");
    } else {
      document.getElementById("baseline_view").innerHTML = "";
    }
    meta.textContent =
      `emotion="${body.steer_emotion}"  version="${r.version || "(bare)"}"  layer=${r.layer}  ` +
      `start_at=${r.start_at_nth_token}  denoised=${r.denoised}  normalize_steering_vector=${r.normalize_steering_vector}  ` +
      `coeff=${r.coeff}  positions=${r.steer_positions}  N=${r.N}  seed=${r.seed}`;
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
                    INDEX_HTML
                    .replace("__VERSIONS_JSON__", json.dumps(VERSION_LIST))
                    .replace("__MODEL_NAME__", json.dumps(MODEL_SHORT))
                )
                return self._send(200, "text/html; charset=utf-8", html)
            if path == "/api/story_emotions":
                version = qs.get("version", [""])[0]
                stories = get_stories(version)
                return self._send(200, "application/json",
                                  json.dumps({"emotions": sorted(stories.keys())}))
            if path == "/api/vector_emotions":
                version = qs.get("version", [""])[0]
                start_at = int(qs.get("start_at", ["0"])[0])
                denoised = qs.get("denoised", ["false"])[0].lower() == "true"
                try:
                    emotions, _ = get_vectors(version, start_at, denoised)
                except FileNotFoundError:
                    emotions = []
                return self._send(200, "application/json",
                                  json.dumps({"emotions": emotions}))
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
                return self._send(200, "application/json", json.dumps({
                    "story": pool[idx], "emotion": emotion, "version": version,
                    "idx": idx, "total": len(pool),
                }))
            if path == "/api/generate":
                result = run_generation(
                    text=body["text"],
                    steer_emotion=body["steer_emotion"],
                    version=body.get("version", ""),
                    layer=body.get("layer", None),
                    start_at_nth_token=body.get("start_at_nth_token", 0),
                    denoised=body.get("denoised", False),
                    normalize_steering_vector=body.get("normalize_steering_vector", True),
                    coeff=body.get("coeff", 0.0),
                    steer_positions=body.get("steer_positions", "all"),
                    max_new_tokens=body.get("max_new_tokens", 160),
                    temperature=body.get("temperature", 1.0),
                    top_p=body.get("top_p", 1.0),
                    do_sample=body.get("do_sample", True),
                    N=body.get("N", 1),
                    seed=body.get("seed", 0),
                    apply_chat_template=body.get("apply_chat_template", True),
                    include_baseline=body.get("include_baseline", False),
                )
                return self._send(200, "application/json", json.dumps(result))
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] POST {path} error: {e}")
            return self._send(400, "text/plain", str(e))


def main(
    model: str = "gemma2_9b",
    port: int = 8081,
    host: str = "127.0.0.1",
    layer: int = None,
    seed: int = 0,
):
    """Launch the interactive emotion steering viewer.

    Args:
        model: Short model name registered in data/model_names.json.
        port: HTTP port to bind. Defaulting to 8081 so the probe viewer at 8080 can run too.
        host: Bind interface. Use 0.0.0.0 for remote access.
        layer: Default steering layer. None = ~2/3 through the model. Overridable in the UI.
        seed: RNG seed for initial randomness in "Sample story" when no per-request seed is set.
    """
    global MODEL, TOKENIZER, MODEL_SHORT, DEFAULT_LAYER, VERSION_LIST

    random.seed(seed)
    torch.manual_seed(seed)

    MODEL_SHORT = model
    VERSION_LIST = discover_versions(model)
    if not VERSION_LIST:
        raise FileNotFoundError(
            f"No emotion-vector dirs found under results/emotion_vectors/ for model='{model}'."
        )
    print(f"[steerview] discovered versions for {model}: {VERSION_LIST}")
    print(f"[steerview] loading model={model} (this may take a moment)...")
    MODEL, TOKENIZER = load_model_and_tokenizer(model)
    DEFAULT_LAYER = layer if layer is not None else default_layer(MODEL)
    print(f"[steerview] default steering layer={DEFAULT_LAYER}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[steerview] serving on http://{host}:{port}   (ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[steerview] bye")


if __name__ == "__main__":
    fire.Fire(main)
