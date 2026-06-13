"""Build a self-contained static HTML viewer for calm-injection-v3 paired runs.

Reads ``exp2_runs.jsonl`` (one row per (prompt_idx, completion_idx, injection_id,
injection_label) cell), pairs each calm row with its matched control row, and
emits a single HTML file with the data embedded inline. The viewer lets you
choose a metric (cos/dot), window (rest_mean/next1_mean/next5_mean/next10_mean),
and emotion, then sorts pairs by the calm-specific delta::

    delta_calm = inj[emotion] - base[emotion]   (calm row)
    delta_ctrl = inj[emotion] - base[emotion]   (control row)
    calm_minus_ctrl = delta_calm - delta_ctrl

Usage:
    uv run python -m scripts.calm_injection_viewer
    uv run python -m scripts.calm_injection_viewer --top-n 200
"""

import json
import math
import os
from pathlib import Path

import fire


EMOTIONS = ["frustrated", "calm", "resentful", "peaceful", "sad", "happy"]
WINDOWS = ["rest_mean", "next1_mean", "next5_mean", "next10_mean"]
METRICS = ["cos", "dot"]


def _slim_window_dict(d):
    """Keep only the 6 relevant emotions as floats."""
    return {e: float(d[e]) for e in EMOTIONS if e in d}


def _slim_metrics(metrics):
    """Strip out deprecated aliases; keep base_/inj_ for the 4 windows."""
    out = {}
    for m in METRICS:
        sub = metrics.get(m, {})
        out[m] = {}
        for w in WINDOWS:
            bk = f"base_{w}"
            ik = f"inj_{w}"
            if bk in sub:
                out[m][bk] = _slim_window_dict(sub[bk])
            if ik in sub:
                out[m][ik] = _slim_window_dict(sub[ik])
    return out


def _load_pairs(runs_path):
    """Group rows by (prompt_idx, completion_idx, injection_id) into pairs."""
    by_key = {}
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["prompt_idx"], r["completion_idx"], r["injection_id"])
            by_key.setdefault(key, {})[r["injection_label"]] = r

    pairs = []
    for key, sides in by_key.items():
        if "calm" not in sides or "control" not in sides:
            continue
        c = sides["calm"]
        x = sides["control"]
        pair = {
            "prompt_idx": c["prompt_idx"],
            "completion_idx": c["completion_idx"],
            "injection_id": c["injection_id"],
            "prompt": c["prompt"],
            "frustrated_prefill_text": c.get("frustrated_prefill_text", ""),
            "cutoff": c.get("cutoff"),
            "n_kept": c.get("n_kept"),
            "kept_text": c.get("kept_text", ""),
            "calm": {
                "injection_text": c.get("injection_text", ""),
                "continued_text": c.get("continued_text", ""),
                "n_inj_tokens": c.get("n_inj_tokens"),
                "n_cont_tokens": c.get("n_cont_tokens"),
                "metrics": _slim_metrics(c.get("metrics", {})),
            },
            "control": {
                "injection_text": x.get("injection_text", ""),
                "continued_text": x.get("continued_text", ""),
                "n_inj_tokens": x.get("n_inj_tokens"),
                "n_cont_tokens": x.get("n_cont_tokens"),
                "metrics": _slim_metrics(x.get("metrics", {})),
            },
        }
        pairs.append(pair)
    return pairs


def _nan_to_none(obj):
    """Recursively replace float NaN with None so the payload is valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nan_to_none(v) for v in obj)
    return obj


def _default_sort_key(pair):
    """Return |calm-ctrl(cos, rest_mean, calm)| for default top-N truncation."""
    try:
        cm = pair["calm"]["metrics"]["cos"]
        xm = pair["control"]["metrics"]["cos"]
        dc = cm["inj_rest_mean"]["calm"] - cm["base_rest_mean"]["calm"]
        dx = xm["inj_rest_mean"]["calm"] - xm["base_rest_mean"]["calm"]
        return abs(dc - dx)
    except Exception:
        return 0.0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Calm-injection v3 viewer</title>
<style>
  :root {
    --bg: #fafafa;
    --panel: #ffffff;
    --border: #d9d9d9;
    --muted: #6b6b6b;
    --accent: #2c5ea8;
    --calm: #f0f7ff;
    --ctrl: #fff7f0;
    --inj-bg: #fff3b0;
    --cont-bg: #e6f3ff;
    --kept-bg: #eeeeee;
  }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1c1c1c;
  }
  header {
    position: sticky;
    top: 0;
    background: #ffffff;
    border-bottom: 1px solid var(--border);
    padding: 12px 16px;
    z-index: 10;
  }
  h1 {
    font-size: 16px;
    margin: 0 0 8px 0;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 16px;
    align-items: center;
  }
  .controls label {
    display: flex;
    flex-direction: column;
    font-size: 11px;
    color: var(--muted);
    gap: 2px;
  }
  .controls select, .controls input[type="number"] {
    font-size: 13px;
    padding: 3px 6px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: white;
  }
  .controls .prompt-filter {
    min-width: 260px;
  }
  .summary {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }
  main {
    padding: 14px 16px 60px 16px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 18px;
    padding: 12px 14px;
  }
  .card h2 {
    font-size: 14px;
    margin: 0 0 8px 0;
    font-weight: 600;
  }
  .card h2 .delta-pos { color: #1a7f37; }
  .card h2 .delta-neg { color: #b42318; }
  .card .meta {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .prompt-line {
    background: #f6f8fa;
    padding: 6px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-style: italic;
  }
  .kept-block {
    background: var(--kept-bg);
    padding: 6px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .kept-label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }
  .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
  }
  .side {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
  }
  .side.calm { background: var(--calm); }
  .side.ctrl { background: var(--ctrl); }
  .side h3 {
    font-size: 12px;
    margin: 0 0 6px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
  }
  .seg {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
    padding: 4px 6px;
    border-radius: 3px;
    margin-bottom: 4px;
  }
  .seg.inj { background: var(--inj-bg); }
  .seg.cont { background: var(--cont-bg); }
  table.emo-table {
    border-collapse: collapse;
    font-size: 11px;
    margin-top: 4px;
    width: 100%;
  }
  table.emo-table th, table.emo-table td {
    border: 1px solid var(--border);
    padding: 3px 6px;
    text-align: right;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  table.emo-table th {
    background: #f3f3f3;
    font-weight: 600;
    text-align: center;
  }
  table.emo-table th:first-child, table.emo-table td:first-child {
    text-align: left;
  }
  .pos { color: #1a7f37; }
  .neg { color: #b42318; }
  .hi { font-weight: 700; }
  .legend {
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
  }
  .pill {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 10px;
    margin-right: 4px;
  }
  .pill.inj { background: var(--inj-bg); }
  .pill.cont { background: var(--cont-bg); }
  .pill.kept { background: var(--kept-bg); }
</style>
</head>
<body>
<header>
  <h1>Calm-injection v3 viewer __SUMMARY_HEADER__</h1>
  <div class="controls">
    <label>Metric
      <select id="metric">
        <option value="cos" selected>cos</option>
        <option value="dot">dot</option>
      </select>
    </label>
    <label>Window
      <select id="window">
        <option value="rest_mean" selected>rest_mean</option>
        <option value="next1_mean">next1_mean</option>
        <option value="next5_mean">next5_mean</option>
        <option value="next10_mean">next10_mean</option>
      </select>
    </label>
    <label>Emotion
      <select id="emotion">
        <option value="frustrated">frustrated</option>
        <option value="calm" selected>calm</option>
        <option value="resentful">resentful</option>
        <option value="peaceful">peaceful</option>
        <option value="sad">sad</option>
        <option value="happy">happy</option>
      </select>
    </label>
    <label>Sort
      <select id="sort">
        <option value="abs_cmc_desc" selected>|calm-ctrl| desc</option>
        <option value="abs_cmc_asc">|calm-ctrl| asc</option>
        <option value="cmc_desc">calm-ctrl desc</option>
        <option value="cmc_asc">calm-ctrl asc</option>
        <option value="dcalm_desc">Δ_calm desc</option>
        <option value="dcalm_asc">Δ_calm asc</option>
      </select>
    </label>
    <label>Top N
      <input id="topn" type="number" min="1" max="9999" value="20" step="1" />
    </label>
    <label class="prompt-filter">Prompt filter
      <select id="promptfilter" multiple size="1" style="height: 28px;"></select>
    </label>
    <label>Injection id
      <select id="injfilter">
        <option value="all" selected>all</option>
        <option value="0">0</option><option value="1">1</option>
        <option value="2">2</option><option value="3">3</option>
        <option value="4">4</option><option value="5">5</option>
        <option value="6">6</option><option value="7">7</option>
      </select>
    </label>
  </div>
  <div class="summary" id="summary">
    <span class="pill kept">kept_text (baseline)</span>
    <span class="pill inj">injection_text</span>
    <span class="pill cont">continued_text</span>
  </div>
</header>
<main id="results"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
(function() {
  const EMOTIONS = ["frustrated","calm","resentful","peaceful","sad","happy"];
  const DATA = JSON.parse(document.getElementById("data").textContent);

  const $ = (id) => document.getElementById(id);
  const fmt = (x) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
    const a = Math.abs(x);
    if (a >= 100) return x.toFixed(1);
    if (a >= 10) return x.toFixed(2);
    if (a >= 1) return x.toFixed(3);
    return x.toFixed(4);
  };
  const cls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

  function deltas(pair, metric, window) {
    const out = {};
    const cm = pair.calm.metrics[metric] || {};
    const xm = pair.control.metrics[metric] || {};
    const cb = cm["base_" + window] || {};
    const ci = cm["inj_" + window] || {};
    const xb = xm["base_" + window] || {};
    const xi = xm["inj_" + window] || {};
    for (const e of EMOTIONS) {
      const dc = (ci[e] ?? 0) - (cb[e] ?? 0);
      const dx = (xi[e] ?? 0) - (xb[e] ?? 0);
      out[e] = { dc, dx, cmc: dc - dx, base_c: cb[e], inj_c: ci[e], base_x: xb[e], inj_x: xi[e] };
    }
    return out;
  }

  function buildPromptFilter() {
    const sel = $("promptfilter");
    const seen = new Map();
    for (const p of DATA) {
      if (!seen.has(p.prompt_idx)) seen.set(p.prompt_idx, p.prompt);
    }
    const sorted = [...seen.entries()].sort((a, b) => a[0] - b[0]);
    for (const [idx, text] of sorted) {
      const o = document.createElement("option");
      o.value = String(idx);
      const short = text.length > 60 ? text.slice(0, 60) + "…" : text;
      o.textContent = `${idx} — ${short}`;
      sel.appendChild(o);
    }
    sel.size = Math.min(8, sorted.length);
  }

  function emoTable(d, emotion) {
    let html = '<table class="emo-table"><thead><tr>'
      + '<th>emotion</th><th>base_c</th><th>inj_c</th><th>Δ_calm</th>'
      + '<th>base_x</th><th>inj_x</th><th>Δ_ctrl</th><th>calm-ctrl</th>'
      + '</tr></thead><tbody>';
    for (const e of EMOTIONS) {
      const r = d[e];
      const isHi = e === emotion ? "hi" : "";
      html += `<tr class="${isHi}">`
        + `<td>${e}</td>`
        + `<td>${fmt(r.base_c)}</td>`
        + `<td>${fmt(r.inj_c)}</td>`
        + `<td class="${cls(r.dc)}">${fmt(r.dc)}</td>`
        + `<td>${fmt(r.base_x)}</td>`
        + `<td>${fmt(r.inj_x)}</td>`
        + `<td class="${cls(r.dx)}">${fmt(r.dx)}</td>`
        + `<td class="${cls(r.cmc)}">${fmt(r.cmc)}</td>`
        + `</tr>`;
    }
    html += '</tbody></table>';
    return html;
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderCard(pair, metric, window, emotion) {
    const d = deltas(pair, metric, window);
    const r = d[emotion];
    const head = `<h2>prompt ${pair.prompt_idx} · completion ${pair.completion_idx} · injection ${pair.injection_id}`
      + `  |  calm-ctrl = <span class="${cls(r.cmc)}">${fmt(r.cmc)}</span>`
      + `  |  Δ_calm = <span class="${cls(r.dc)}">${fmt(r.dc)}</span>,`
      + `  Δ_ctrl = <span class="${cls(r.dx)}">${fmt(r.dx)}</span></h2>`;
    const meta = `<div class="meta">metric=${metric} · window=${window} · emotion=${emotion}`
      + ` · cutoff=${pair.cutoff} · n_kept=${pair.n_kept}`
      + ` · n_inj(c/x)=${pair.calm.n_inj_tokens}/${pair.control.n_inj_tokens}`
      + ` · n_cont(c/x)=${pair.calm.n_cont_tokens}/${pair.control.n_cont_tokens}</div>`;
    const prompt = `<div class="prompt-line"><b>user:</b> ${escapeHtml(pair.prompt)}</div>`;
    const kept = `<div class="kept-label">shared baseline (kept_text up to cutoff)</div>`
      + `<div class="kept-block">${escapeHtml(pair.kept_text)}</div>`;
    const calmHtml = `<div class="side calm"><h3>calm injection</h3>`
      + `<div class="seg inj">${escapeHtml(pair.calm.injection_text)}</div>`
      + `<div class="seg cont">${escapeHtml(pair.calm.continued_text)}</div>`
      + `</div>`;
    const ctrlHtml = `<div class="side ctrl"><h3>control injection</h3>`
      + `<div class="seg inj">${escapeHtml(pair.control.injection_text)}</div>`
      + `<div class="seg cont">${escapeHtml(pair.control.continued_text)}</div>`
      + `</div>`;
    const table = emoTable(d, emotion);
    return `<div class="card">${head}${meta}${prompt}${kept}`
      + `<div class="compare">${calmHtml}${ctrlHtml}</div>${table}</div>`;
  }

  function render() {
    const metric = $("metric").value;
    const window = $("window").value;
    const emotion = $("emotion").value;
    const sort = $("sort").value;
    const topn = Math.max(1, parseInt($("topn").value || "20", 10));
    const promptSel = [...$("promptfilter").selectedOptions].map(o => parseInt(o.value, 10));
    const injSel = $("injfilter").value;

    let rows = DATA.slice();
    if (promptSel.length > 0) {
      const allowed = new Set(promptSel);
      rows = rows.filter(p => allowed.has(p.prompt_idx));
    }
    if (injSel !== "all") {
      const want = parseInt(injSel, 10);
      rows = rows.filter(p => p.injection_id === want);
    }

    const scored = rows.map(p => {
      const d = deltas(p, metric, window);
      const r = d[emotion];
      return { p, dc: r.dc, dx: r.dx, cmc: r.cmc };
    });

    scored.sort((a, b) => {
      switch (sort) {
        case "abs_cmc_desc": return Math.abs(b.cmc) - Math.abs(a.cmc);
        case "abs_cmc_asc": return Math.abs(a.cmc) - Math.abs(b.cmc);
        case "cmc_desc": return b.cmc - a.cmc;
        case "cmc_asc": return a.cmc - b.cmc;
        case "dcalm_desc": return b.dc - a.dc;
        case "dcalm_asc": return a.dc - b.dc;
      }
      return 0;
    });

    const top = scored.slice(0, topn);
    const html = top.map(s => renderCard(s.p, metric, window, emotion)).join("");
    $("results").innerHTML = html || '<div class="muted">No pairs match the current filters.</div>';
    $("summary").lastChild && null;
    const sumEl = document.querySelector(".summary");
    sumEl.innerHTML =
      `<span class="pill kept">kept_text (baseline)</span>`
      + `<span class="pill inj">injection_text</span>`
      + `<span class="pill cont">continued_text</span>`
      + ` &nbsp; showing ${top.length} of ${scored.length} matching pairs (total embedded: ${DATA.length}).`;
  }

  buildPromptFilter();
  for (const id of ["metric","window","emotion","sort","topn","promptfilter","injfilter"]) {
    $(id).addEventListener("change", render);
    if (id === "topn") $(id).addEventListener("input", render);
  }
  render();
})();
</script>
</body>
</html>
"""


def build(
    runs_path: str = "results/calm_injection_v3/exp2_runs.jsonl",
    out_path: str = "results/calm_injection_v3/viewer.html",
    top_n: int | None = None,
):
    """Build the static HTML viewer.

    Args:
        runs_path: path to ``exp2_runs.jsonl`` (relative paths resolved against cwd).
        out_path: where to write the self-contained HTML file.
        top_n: if set, keep only the top-N pairs by ``|calm-ctrl(cos, rest_mean, calm)|``.
    """
    runs_path = os.path.abspath(runs_path)
    out_path = os.path.abspath(out_path)

    pairs = _load_pairs(runs_path)
    total = len(pairs)

    if top_n is not None and top_n < total:
        pairs.sort(key=_default_sort_key, reverse=True)
        pairs = pairs[:top_n]

    pairs = _nan_to_none(pairs)
    json.dumps(pairs, allow_nan=False)
    data_json = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    summary_header = f"<span style='font-size:12px;font-weight:400;color:#6b6b6b'>({len(pairs)} pairs)</span>"
    html = HTML_TEMPLATE.replace("__SUMMARY_HEADER__", summary_header).replace(
        "__DATA_JSON__", data_json
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(out_path)
    print(f"embedded {len(pairs)} pairs (of {total} total), {size_kb:.1f} KB")


def main():
    fire.Fire(build)


if __name__ == "__main__":
    main()
