"""Interactive web viewer for inline assistant-turn interventions.

Two-panel local HTTP tool:

 1. **Generate + probe.** Give a user prompt (sample a story or type your
    own). The model generates one unsteered response. Each assistant token is
    rendered as a color chip whose shade encodes the emotion-probe score at a
    chosen layer. The probe is *context-aware*: a single forward pass is run
    over the full chat-formatted sequence (prompt + assistant) and only the
    assistant-generated tokens are colored.

 2. **Intervene + continue.** Click any token in the Panel-1 response to set a
    cutoff. Type a message to inject. The assistant turn is rebuilt as
    ``prompt + response[:cutoff+1] + injected_text`` (inline — as if the
    assistant wrote the injected text itself), generation continues from
    there, and the whole assistant turn is re-probed and re-colored. The kept
    / injected / continued segments are visually delimited.

The probe knobs (version, layer, metric, start_at_nth_token, denoised) mirror
``interactive_emotion_viewer``; generation knobs (max_new_tokens, temperature,
top_p, do_sample, seed) mirror ``interactive_steering_viewer``. A "Re-color
(no regen)" action re-probes a stored generation with a different emotion /
layer / metric without resampling tokens. An optional fixed color-scale vmax
lets Panel 1 and Panel 2 be compared on the same scale.

Usage:
    uv run python -m scripts.viewers.interactive_intervention_viewer \\
        --model gemma2_9b --port 8082
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
SESSIONS: dict = {}
SESSION_SEQ = [0]

LOAD_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()


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
        for _, _, files in os.walk(full):
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


def _resolve_layer(layer, vecs_by_layer):
    if layer is None:
        layer = DEFAULT_LAYER if DEFAULT_LAYER is not None else default_layer(MODEL)
    if layer not in vecs_by_layer:
        raise ValueError(
            f"Layer {layer} not available. Have: {sorted(vecs_by_layer.keys())}"
        )
    return int(layer)


def _trim_eos(ids: list) -> list:
    """Drop everything from the first eos/pad token onward."""
    stop = set()
    for t in (TOKENIZER.eos_token_id, TOKENIZER.pad_token_id):
        if isinstance(t, int):
            stop.add(t)
        elif isinstance(t, (list, tuple)):
            stop.update(int(x) for x in t)
    out = []
    for t in ids:
        if t in stop:
            break
        out.append(int(t))
    return out


def _safe_chat_tokenize(formatted: str) -> list:
    """Tokenize a chat-templated (or raw) string into ids, avoiding double-BOS.

    Many chat templates (gemma2, llama-3, ...) emit a literal ``<bos>`` at the
    very start. The fast tokenizer is *also* willing to prepend a BOS when
    ``add_special_tokens=True``, which produces a spurious double-BOS that
    shifts the first-token hidden state off-distribution. We detect the
    template-emitted BOS by ``startswith(tokenizer.bos_token)`` and disable
    the tokenizer's own special-token insertion in that case. For templates
    that don't include a literal BOS, behaviour is unchanged.
    """
    bos = TOKENIZER.bos_token
    add_special = not (isinstance(bos, str) and bos and formatted.startswith(bos))
    ids = TOKENIZER(formatted, add_special_tokens=add_special)["input_ids"]
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError(
            f"Expected list[int] from tokenizer; got {type(ids).__name__} "
            f"with first elem type {type(ids[0]).__name__ if ids else 'empty'}"
        )
    return list(ids)


def _build_prompt_ids(prompt: str, apply_chat_template: bool) -> list:
    """Tokenize the user prompt, optionally wrapped as a user-turn chat message.

    Two-step (templated string -> tokenizer) to match generate_with_steering's
    pattern and avoid the BatchEncoding-vs-list ambiguity of
    apply_chat_template(tokenize=True) across transformers versions.
    """
    if apply_chat_template:
        formatted = TOKENIZER.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt
    return _safe_chat_tokenize(formatted)


@torch.no_grad()
def _generate(prefix_ids: list, max_new_tokens, temperature, top_p, do_sample, seed):
    """Continue generation from a flat list of token ids; return the new ids (eos-trimmed)."""
    device = next(MODEL.parameters()).device
    inp = torch.tensor([prefix_ids], device=device)
    attn = torch.ones_like(inp)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    out = MODEL.generate(
        input_ids=inp,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else 1.0,
        top_p=float(top_p) if do_sample else 1.0,
        num_return_sequences=1,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    new_ids = out[0, len(prefix_ids):].tolist()
    return _trim_eos(new_ids)


def _decode_full(full_ids: list) -> str:
    """Decode the full chat-formatted token sequence to plaintext (special tokens kept)."""
    return TOKENIZER.decode(full_ids, skip_special_tokens=False)


@torch.no_grad()
def _probe_span(full_ids: list, span_start: int, probe_emotion, version,
                metric, layer, start_at_nth_token, denoised):
    """Forward-pass full_ids once; return per-token scores for ids[span_start:]."""
    emotions, vecs_by_layer = get_vectors(version, start_at_nth_token, denoised)
    if probe_emotion not in emotions:
        raise ValueError(
            f"Emotion '{probe_emotion}' not in vectors (version='{version}'). "
            f"Available ({len(emotions)}): {emotions[:10]}..."
        )
    layer = _resolve_layer(layer, vecs_by_layer)
    e_idx = emotions.index(probe_emotion)
    cosine = metric == "cosine"

    device = next(MODEL.parameters()).device
    dtype = next(MODEL.parameters()).dtype
    v = vecs_by_layer[layer][e_idx].to(device=device, dtype=dtype)
    if cosine:
        v = torch.nn.functional.normalize(v, dim=-1)

    inp = torch.tensor([full_ids], device=device)
    with MODEL_LOCK:
        out = MODEL(input_ids=inp, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # [S, H]
    if cosine:
        h = torch.nn.functional.normalize(h, dim=-1)
    scores = (h @ v).float().cpu()  # [S]

    span_ids = full_ids[span_start:]
    span_scores = scores[span_start:].tolist()
    raw_tokens = TOKENIZER.convert_ids_to_tokens(span_ids)
    display = [TOKENIZER.convert_tokens_to_string([t]) or t for t in raw_tokens]
    return {
        "tokens": display,
        "raw_tokens": raw_tokens,
        "scores": span_scores,
        "layer": int(layer),
        "metric": metric,
        "version": version,
        "start_at_nth_token": int(start_at_nth_token),
        "denoised": bool(denoised),
    }


def _new_session(payload: dict) -> str:
    with SESSION_LOCK:
        SESSION_SEQ[0] += 1
        sid = str(SESSION_SEQ[0])
        SESSIONS[sid] = payload
    return sid


def do_generate(body: dict) -> dict:
    """Panel 1: generate one unsteered response and probe-color the assistant tokens."""
    prompt = body["prompt"]
    apply_ct = bool(body.get("apply_chat_template", True))
    with MODEL_LOCK:
        prompt_ids = _build_prompt_ids(prompt, apply_ct)
        gen_ids = _generate(
            prompt_ids,
            max_new_tokens=body.get("max_new_tokens", 200),
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            do_sample=body.get("do_sample", True),
            seed=body.get("seed", 0),
        )
    if not gen_ids:
        raise ValueError("Model produced no tokens before EOS.")
    full_ids = prompt_ids + gen_ids
    P = len(prompt_ids)
    segments = [
        {"label": "prompt", "start": 0, "end": P},
        {"label": "response", "start": P, "end": P + len(gen_ids)},
    ]
    sid = _new_session(
        {
            "full_ids": full_ids,
            "span_start": 0,
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "apply_chat_template": apply_ct,
            "segments": segments,
        }
    )
    scored = _probe_span(
        full_ids, 0,
        probe_emotion=body["emotion"],
        version=body.get("version", ""),
        metric=body.get("metric", DEFAULT_METRIC),
        layer=body.get("layer", None),
        start_at_nth_token=body.get("start_at_nth_token", 0),
        denoised=body.get("denoised", False),
    )
    scored["session_id"] = sid
    scored["segments"] = segments
    scored["n_gen_tokens"] = len(gen_ids)
    scored["plaintext"] = _decode_full(full_ids)
    return scored


def do_intervene(body: dict) -> dict:
    """Panel 2: splice injected text inline after the cutoff token, continue, re-probe."""
    parent = SESSIONS.get(str(body["session_id"]))
    if parent is None:
        raise ValueError(f"Unknown session_id {body['session_id']!r} (regenerate Panel 1).")
    prompt_ids = parent["prompt_ids"]
    gen_ids = parent["gen_ids"]

    cutoff = body.get("cutoff", None)
    if cutoff is None:
        cutoff = len(gen_ids) - 1
    cutoff = int(cutoff)
    if cutoff < -1 or cutoff >= len(gen_ids):
        raise ValueError(
            f"cutoff {cutoff} out of range [-1, {len(gen_ids) - 1}] "
            f"(-1 keeps none of the original response)."
        )
    kept_ids = gen_ids[: cutoff + 1]

    injected_text = body.get("injected_text", "")
    injected_ids = (
        list(TOKENIZER(injected_text, add_special_tokens=False)["input_ids"])
        if injected_text
        else []
    )

    prefix_ids = prompt_ids + kept_ids + injected_ids
    with MODEL_LOCK:
        cont_ids = _generate(
            prefix_ids,
            max_new_tokens=body.get("max_new_tokens", 200),
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            do_sample=body.get("do_sample", True),
            seed=body.get("seed", 0),
        )

    full_ids = prefix_ids + cont_ids
    P = len(prompt_ids)
    n_kept, n_inj, n_cont = len(kept_ids), len(injected_ids), len(cont_ids)
    segments = [
        {"label": "prompt", "start": 0, "end": P},
        {"label": "kept", "start": P, "end": P + n_kept},
        {"label": "injected", "start": P + n_kept, "end": P + n_kept + n_inj},
        {"label": "continued", "start": P + n_kept + n_inj, "end": P + n_kept + n_inj + n_cont},
    ]
    sid = _new_session(
        {
            "full_ids": full_ids,
            "span_start": 0,
            "prompt_ids": prompt_ids,
            "gen_ids": kept_ids + injected_ids + cont_ids,
            "apply_chat_template": parent["apply_chat_template"],
            "segments": segments,
        }
    )
    scored = _probe_span(
        full_ids, 0,
        probe_emotion=body["emotion"],
        version=body.get("version", ""),
        metric=body.get("metric", DEFAULT_METRIC),
        layer=body.get("layer", None),
        start_at_nth_token=body.get("start_at_nth_token", 0),
        denoised=body.get("denoised", False),
    )
    scored["session_id"] = sid
    scored["segments"] = segments
    scored["cutoff"] = cutoff
    scored["n_injected_tokens"] = n_inj
    scored["plaintext"] = _decode_full(full_ids)
    return scored


def do_recolor(body: dict) -> dict:
    """Re-probe a stored session with (possibly) different emotion / layer / metric — no regen."""
    sess = SESSIONS.get(str(body["session_id"]))
    if sess is None:
        raise ValueError(f"Unknown session_id {body['session_id']!r}.")
    scored = _probe_span(
        sess["full_ids"], sess["span_start"],
        probe_emotion=body["emotion"],
        version=body.get("version", ""),
        metric=body.get("metric", DEFAULT_METRIC),
        layer=body.get("layer", None),
        start_at_nth_token=body.get("start_at_nth_token", 0),
        denoised=body.get("denoised", False),
    )
    scored["session_id"] = str(body["session_id"])
    scored["segments"] = sess["segments"]
    scored["plaintext"] = _decode_full(sess["full_ids"])
    return scored


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Intervention Viewer</title>
<style>
 body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #222; max-width: 1180px; }
 h1 { font-size: 20px; margin-bottom: 4px; }
 .sub { color: #666; font-size: 13px; margin-bottom: 20px; }
 .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
 label { font-size: 13px; }
 input[type=text], input[type=number], select, textarea {
   padding: 4px 6px; font-size: 14px; font-family: inherit;
 }
 textarea { width: 100%; box-sizing: border-box; min-height: 90px; }
 button { padding: 6px 14px; font-size: 14px; cursor: pointer; }
 .tokens { padding: 14px; background: #fff; border: 1px solid #ddd; border-radius: 4px;
           font-size: 15px; line-height: 2.4; min-height: 50px; }
 .tok { padding: 2px 5px; border-radius: 4px; margin: 1px; display: inline-block; cursor: default; }
 .tok.clickable { cursor: pointer; }
 .tok.clickable:hover { outline: 2px solid #444; }
 .tok.cutoff { outline: 3px solid #111; }
 .tok.seg-prompt { font-size: 12px; opacity: 0.9; }
 .tok.seg-injected { text-decoration: underline; text-decoration-style: wavy;
   text-decoration-color: #111; text-underline-offset: 3px; }
 .seg-sep { display: inline-block; border-left: 2px dashed #999; margin: 0 5px;
   height: 1.1em; vertical-align: middle; }
 .seg-lab { font-size: 10px; font-family: monospace; color: #666; margin: 0 3px;
   text-transform: uppercase; letter-spacing: .5px; }
 .meta { color: #555; font-size: 12px; margin: 6px 0 10px 0; font-family: monospace; }
 .err { color: #b00; font-family: monospace; }
 fieldset { border: 1px solid #eee; border-radius: 4px; margin-bottom: 16px; padding: 10px 14px; }
 legend { font-size: 13px; font-weight: bold; padding: 0 6px; color: #444; }
 .panel { border-left: 4px solid #b2182b; }
 .hint { font-size: 12px; color: #777; margin: 4px 0 8px 0; }
 details.plaintext { margin-top: 8px; font-size: 12px; }
 details.plaintext summary { cursor: pointer; color: #555; user-select: none; }
 details.plaintext pre { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px;
   padding: 8px 10px; white-space: pre-wrap; word-break: break-word;
   font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
   margin: 6px 0 0 0; max-height: 480px; overflow: auto; }
</style>
</head>
<body>
<h1>Inline Intervention Viewer</h1>
<div class="sub">
  model: <b id="model_name"></b> · versions on disk: <span id="version_info"></span>
  · context-aware probe (colors = emotion score on the full chat-formatted sequence)
</div>

<fieldset>
<legend>Probe (shared by both panels)</legend>
<div class="row">
 <label>Visualize emotion: <input id="probe_emotion" type="text" value="angry" list="probe_emotions"/></label>
 <datalist id="probe_emotions"></datalist>
 <label>Vectors version: <select id="probe_version"></select></label>
 <label>Metric:
  <select id="metric">
   <option value="dot">dot product</option>
   <option value="cosine">cosine similarity</option>
  </select>
 </label>
 <label>Layer (blank=default): <input id="layer" type="number" value="" style="width:80px"/></label>
 <label>start_at_nth_token: <input id="start_at" type="number" value="50" style="width:80px"/></label>
 <label><input id="denoised" type="checkbox"/> denoised</label>
 <label>Fixed |vmax| (blank=auto/panel): <input id="vmax" type="number" value="" step="0.5" style="width:90px"/></label>
</div>
<div class="row">
 <button id="recolor1_btn">Re-color Panel 1 (no regen)</button>
 <button id="recolor2_btn">Re-color Panel 2 (no regen)</button>
</div>
</fieldset>

<fieldset class="panel">
<legend>Panel 1 · Prompt &rarr; unsteered response</legend>
<div class="row">
 <label>Story emotion: <input id="story_emotion" type="text" value="angry" list="story_emotions"/></label>
 <datalist id="story_emotions"></datalist>
 <label>Stories version: <select id="story_version"></select></label>
 <label>Seed (blank=random): <input id="story_seed" type="number" value="" style="width:90px"/></label>
 <button id="sample_btn">Sample story</button>
</div>
<div class="meta" id="story_meta">&nbsp;</div>
<textarea id="prompt_text" placeholder="Sampled story will appear here — or type/paste your own user prompt..."></textarea>
<div class="row" style="margin-top:10px">
 <label>max_new_tokens: <input id="g1_max" type="number" value="200" style="width:80px"/></label>
 <label>temperature: <input id="g1_temp" type="number" value="1.0" step="0.1" style="width:70px"/></label>
 <label>top_p: <input id="g1_topp" type="number" value="1.0" step="0.05" style="width:70px"/></label>
 <label><input id="g1_sample" type="checkbox" checked/> sample</label>
 <label>seed: <input id="g1_seed" type="number" value="0" style="width:80px"/></label>
 <label><input id="g1_ct" type="checkbox" checked/> wrap in chat template</label>
 <button id="gen_btn">Generate &amp; probe</button>
</div>
<div class="hint">Click any token below to set the intervention cutoff (injection goes <i>after</i> it).</div>
<div class="meta" id="gen_meta">&nbsp;</div>
<div class="tokens" id="p1_view">(no generation yet)</div>
<details class="plaintext" id="p1_plain_details">
 <summary>Show plaintext transcript</summary>
 <pre id="p1_plain"></pre>
</details>
</fieldset>

<fieldset class="panel">
<legend>Panel 2 · Inline intervention &rarr; continued response</legend>
<div class="meta" id="cutoff_meta">Cutoff: (none — defaults to end of response, i.e. pure continuation)</div>
<textarea id="injected_text" placeholder="Message to splice inline into the assistant turn after the cutoff token..."></textarea>
<div class="row" style="margin-top:10px">
 <label>max_new_tokens: <input id="g2_max" type="number" value="200" style="width:80px"/></label>
 <label>temperature: <input id="g2_temp" type="number" value="1.0" step="0.1" style="width:70px"/></label>
 <label>top_p: <input id="g2_topp" type="number" value="1.0" step="0.05" style="width:70px"/></label>
 <label><input id="g2_sample" type="checkbox" checked/> sample</label>
 <label>seed: <input id="g2_seed" type="number" value="0" style="width:80px"/></label>
 <button id="intervene_btn">Intervene &amp; continue</button>
</div>
<div class="meta" id="interv_meta">&nbsp;</div>
<div class="tokens" id="p2_view">(no intervention yet)</div>
<details class="plaintext" id="p2_plain_details">
 <summary>Show plaintext transcript</summary>
 <pre id="p2_plain"></pre>
</details>
</fieldset>

<script>
const VERSIONS = __VERSIONS_JSON__;
const DEFAULT_METRIC = __DEFAULT_METRIC__;
const MODEL_NAME = __MODEL_NAME__;
document.getElementById("model_name").textContent = MODEL_NAME;
document.getElementById("version_info").textContent =
  VERSIONS.map(v => v === "" ? "(bare)" : v).join(", ");

let P1_SESSION = null, P2_SESSION = null, CUTOFF = null, P1_NTOK = 0;

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

function probeBody() {
  return {
    emotion: document.getElementById("probe_emotion").value.trim(),
    version: document.getElementById("probe_version").value,
    metric: document.getElementById("metric").value,
    layer: document.getElementById("layer").value === "" ? null
           : Number(document.getElementById("layer").value),
    start_at_nth_token: Number(document.getElementById("start_at").value),
    denoised: document.getElementById("denoised").checked,
  };
}

function diverging(t) {
  const mid = [247, 247, 247], pos = [178, 24, 43], neg = [33, 102, 172];
  t = Math.max(-1, Math.min(1, t));
  const a = Math.abs(t), target = t >= 0 ? pos : neg;
  return [
    Math.round(mid[0] + a * (target[0] - mid[0])),
    Math.round(mid[1] + a * (target[1] - mid[1])),
    Math.round(mid[2] + a * (target[2] - mid[2])),
  ];
}

function renderTokens(viewId, r, clickableLabel) {
  // clickableLabel: name of the segment whose tokens are clickable to set a cutoff
  // (e.g. "response" for Panel 1). null/undefined => no clicks.
  const view = document.getElementById(viewId);
  view.innerHTML = "";
  const manualVmax = document.getElementById("vmax").value;
  const vmax = manualVmax !== "" ? Math.abs(Number(manualVmax))
    : r.scores.reduce((m, s) => Math.max(m, Math.abs(s)), 1e-9);
  const segs = r.segments || [{label: "", start: 0, end: r.tokens.length}];
  let si = 0;
  for (let i = 0; i < r.tokens.length; i++) {
    while (si < segs.length && i >= segs[si].end) si++;
    const seg = segs[si] || {label: "", start: 0};
    if (i === seg.start && segs.length > 1) {
      if (i > 0) {
        const sep = document.createElement("span");
        sep.className = "seg-sep"; view.appendChild(sep);
      }
      const lab = document.createElement("span");
      lab.className = "seg-lab"; lab.textContent = seg.label;
      view.appendChild(lab);
    }
    const span = document.createElement("span");
    const score = r.scores[i];
    const [rr, gg, bb] = diverging(score / vmax);
    const isClickable = clickableLabel && seg.label === clickableLabel;
    span.className = "tok"
      + (isClickable ? " clickable" : "")
      + (seg.label === "prompt" ? " seg-prompt" : "")
      + (seg.label === "injected" ? " seg-injected" : "");
    span.style.background = `rgb(${rr},${gg},${bb})`;
    const lum = 0.299 * rr + 0.587 * gg + 0.114 * bb;
    span.style.color = lum > 160 ? "#111" : "#fff";
    span.textContent = r.tokens[i] === "" ? " " : r.tokens[i];
    const relIdx = i - seg.start;
    span.title = `#${i}  ${seg.label}[${relIdx}]  raw="${r.raw_tokens[i]}"  ${r.metric}=${score.toFixed(4)}`;
    if (isClickable) {
      const segStart = seg.start;
      const cutoffInGen = i - segStart;
      const fullIdx = i;
      span.onclick = () => setCutoff(cutoffInGen, r.tokens[fullIdx], fullIdx);
      if (cutoffInGen === CUTOFF) span.classList.add("cutoff");
    }
    view.appendChild(span);
  }
  const plainId = viewId.replace("_view", "_plain");
  const plainEl = document.getElementById(plainId);
  if (plainEl) plainEl.textContent = r.plaintext || "";
  const minS = Math.min(...r.scores), maxS = Math.max(...r.scores);
  return `range=[${minS.toFixed(3)}, ${maxS.toFixed(3)}]  |vmax|=${vmax.toFixed(3)}  ` +
         `tokens=${r.tokens.length}`;
}

function setCutoff(cutoffInGen, tokTxt, fullIdx) {
  CUTOFF = cutoffInGen;
  document.querySelectorAll("#p1_view .tok.clickable").forEach((el) => {
    el.classList.remove("cutoff");
  });
  const allToks = document.querySelectorAll("#p1_view .tok");
  if (allToks[fullIdx]) allToks[fullIdx].classList.add("cutoff");
  document.getElementById("cutoff_meta").textContent =
    `Cutoff: response token #${cutoffInGen} ("${tokTxt}") — kept = response[0..${cutoffInGen}], ` +
    `then injected text, then continuation.`;
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
    document.getElementById("prompt_text").value = r.story;
    meta.textContent = `emotion="${r.emotion}"  version="${r.version || "(bare)"}"  idx=${r.idx}/${r.total}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
};

document.getElementById("gen_btn").onclick = async () => {
  const prompt = document.getElementById("prompt_text").value;
  if (!prompt.trim()) {
    document.getElementById("gen_meta").innerHTML =
      '<span class="err">Prompt is empty — sample or paste one first.</span>';
    return;
  }
  const body = Object.assign(probeBody(), {
    prompt: prompt,
    max_new_tokens: Number(document.getElementById("g1_max").value),
    temperature: Number(document.getElementById("g1_temp").value),
    top_p: Number(document.getElementById("g1_topp").value),
    do_sample: document.getElementById("g1_sample").checked,
    seed: Number(document.getElementById("g1_seed").value),
    apply_chat_template: document.getElementById("g1_ct").checked,
  });
  const meta = document.getElementById("gen_meta");
  meta.textContent = "Generating + probing...";
  document.getElementById("p1_view").innerHTML = "...";
  try {
    const r = await postJSON("/api/generate", body);
    P1_SESSION = r.session_id; P1_NTOK = r.n_gen_tokens; CUTOFF = null;
    document.getElementById("cutoff_meta").textContent =
      "Cutoff: (none — defaults to end of response, i.e. pure continuation)";
    const summary = renderTokens("p1_view", r, "response");
    meta.textContent =
      `probe="${body.emotion}"  version="${r.version || "(bare)"}"  metric=${r.metric}  ` +
      `layer=${r.layer}  start_at=${r.start_at_nth_token}  denoised=${r.denoised}  ${summary}  sess=${r.session_id}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
};

document.getElementById("intervene_btn").onclick = async () => {
  if (P1_SESSION == null) {
    document.getElementById("interv_meta").innerHTML =
      '<span class="err">Generate a Panel-1 response first.</span>';
    return;
  }
  const body = Object.assign(probeBody(), {
    session_id: P1_SESSION,
    cutoff: CUTOFF,
    injected_text: document.getElementById("injected_text").value,
    max_new_tokens: Number(document.getElementById("g2_max").value),
    temperature: Number(document.getElementById("g2_temp").value),
    top_p: Number(document.getElementById("g2_topp").value),
    do_sample: document.getElementById("g2_sample").checked,
    seed: Number(document.getElementById("g2_seed").value),
  });
  const meta = document.getElementById("interv_meta");
  meta.textContent = "Intervening + generating + probing...";
  document.getElementById("p2_view").innerHTML = "...";
  try {
    const r = await postJSON("/api/intervene", body);
    P2_SESSION = r.session_id;
    const summary = renderTokens("p2_view", r, null);
    meta.textContent =
      `probe="${body.emotion}"  cutoff=${r.cutoff}  injected_tokens=${r.n_injected_tokens}  ` +
      `layer=${r.layer}  metric=${r.metric}  ${summary}  sess=${r.session_id}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
};

async function recolor(which) {
  const sess = which === 1 ? P1_SESSION : P2_SESSION;
  const metaId = which === 1 ? "gen_meta" : "interv_meta";
  const viewId = which === 1 ? "p1_view" : "p2_view";
  const meta = document.getElementById(metaId);
  if (sess == null) {
    meta.innerHTML = '<span class="err">Nothing generated in this panel yet.</span>';
    return;
  }
  meta.textContent = "Re-coloring...";
  try {
    const r = await postJSON("/api/recolor", Object.assign(probeBody(), {session_id: sess}));
    const clickable = which === 1 ? "response" : null;
    const summary = renderTokens(viewId, r, clickable);
    meta.textContent =
      `probe="${r.metric === undefined ? "" : document.getElementById("probe_emotion").value}"  ` +
      `layer=${r.layer}  metric=${r.metric}  version="${r.version || "(bare)"}"  ${summary}  sess=${sess}`;
  } catch (e) {
    meta.innerHTML = '<span class="err">' + e.message + '</span>';
  }
}
document.getElementById("recolor1_btn").onclick = () => recolor(1);
document.getElementById("recolor2_btn").onclick = () => recolor(2);
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
                    200, "application/json",
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
                return self._send(200, "application/json", json.dumps({
                    "story": pool[idx], "emotion": emotion, "version": version,
                    "idx": idx, "total": len(pool),
                }))
            if path == "/api/generate":
                return self._send(200, "application/json", json.dumps(do_generate(body)))
            if path == "/api/intervene":
                return self._send(200, "application/json", json.dumps(do_intervene(body)))
            if path == "/api/recolor":
                return self._send(200, "application/json", json.dumps(do_recolor(body)))
            return self._send(404, "text/plain", "not found")
        except Exception as e:
            print(f"[http] POST {path} error: {e}")
            return self._send(400, "text/plain", str(e))


def main(
    model: str = "gemma2_9b",
    port: int = 8082,
    host: str = "127.0.0.1",
    metric: str = "dot",
    layer: int = None,
    seed: int = 0,
):
    """Launch the inline intervention viewer.

    Args:
        model: Short model name registered in data/model_names.json.
        port: HTTP port to bind. Defaults to 8082 so the probe (8080) and
            steering (8081) viewers can run alongside it.
        host: Bind interface. Use 0.0.0.0 for remote access.
        metric: Default probe metric: "dot" or "cosine". Overridable in the UI.
        layer: Default probe layer. None = ~2/3 through the model. Overridable in the UI.
        seed: RNG seed for "Sample story" randomness when no per-request seed is set.
    """
    global MODEL, TOKENIZER, MODEL_SHORT, DEFAULT_METRIC, DEFAULT_LAYER, VERSION_LIST
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
    print(f"[interv] discovered versions for {model}: {VERSION_LIST}")
    print(f"[interv] loading model={model} (this may take a moment)...")
    MODEL, TOKENIZER = load_model_and_tokenizer(model)
    DEFAULT_LAYER = layer if layer is not None else default_layer(MODEL)
    print(f"[interv] default layer={DEFAULT_LAYER}, default metric={DEFAULT_METRIC}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[interv] serving on http://{host}:{port}   (ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[interv] bye")


if __name__ == "__main__":
    fire.Fire(main)
