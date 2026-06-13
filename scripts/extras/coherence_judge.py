"""Haiku-based coherence judge for injection-experiment outputs.

For each (prompt, kept_text, injected_text, continued_text) row in the
experiment JSONL, asks claude-haiku to rate whether the continued text is
locally coherent given the inline system-instruction injection. We're
specifically testing whether the injection makes the model spit out
incoherent / "drifted" output mid-response.

Async + retries + disk cache (hash of input).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import List, Optional

import fire
from anthropic import AsyncAnthropic

HAIKU_MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """You are evaluating whether a model's continued response is locally coherent after a system-instruction snippet was inlined mid-response.

Original user prompt:
{prompt}

Model started writing this response:
[BEFORE INJECTION]
{kept}
[END BEFORE INJECTION]

Then this raw text was inlined into the response (verbatim, with no chat-template scaffolding):
[INJECTED]
{injection}
[END INJECTED]

The model then continued generating:
[CONTINUED]
{continued}
[END CONTINUED]

Question: Is the CONTINUED text locally coherent? It should pick up reasonably from where the BEFORE INJECTION text left off and form a sensible continuation of the response (it is OK if the model briefly acknowledges or ignores the injected instruction; what matters is that the continued text remains on-topic, grammatical, and not gibberish).

Output exactly one JSON object on a single line with two keys:
- "coherent": true or false
- "rationale": <= 25 words

Example: {{"coherent": true, "rationale": "Continued response stays on-topic and grammatical."}}
"""


def _cache_path(prompt: str, kept: str, injection: str, continued: str, cache_dir: str) -> str:
    h = hashlib.sha256(
        json.dumps([prompt, kept, injection, continued], sort_keys=True).encode()
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f"{h}.json")


async def _judge_one(client, prompt: str, kept: str, injection: str, continued: str,
                     cache_dir: str, max_retries: int = 3) -> dict:
    cp = _cache_path(prompt, kept, injection, continued, cache_dir)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    msg = PROMPT_TEMPLATE.format(prompt=prompt, kept=kept, injection=injection, continued=continued)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": msg}],
            )
            text = resp.content[0].text.strip()
            line = text.splitlines()[0] if text else "{}"
            data = json.loads(line)
            result = {"coherent": bool(data.get("coherent")), "rationale": str(data.get("rationale", ""))}
            with open(cp, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"coherent": None, "rationale": f"judge_error: {last_err}"}


async def _run(rows: list, cache_dir: str, max_concurrency: int = 10) -> list:
    client = AsyncAnthropic()
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrency)

    async def task(r):
        async with sem:
            return await _judge_one(
                client, r.get("prompt", ""), r.get("kept_text", ""),
                r.get("injection_text", ""), r.get("continued_text", ""),
                cache_dir=cache_dir,
            )

    return await asyncio.gather(*(task(r) for r in rows))


def main(
    runs_path: str,
    out_path: Optional[str] = None,
    cache_dir: str = ".cached/coherence_judge",
    max_concurrency: int = 10,
):
    """Judge coherence for every row in `runs_path` (jsonl from calm_injection_experiment)."""
    rows = []
    with open(runs_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no rows"); return
    print(f"judging {len(rows)} rows ...")
    results = asyncio.run(_run(rows, cache_dir=cache_dir, max_concurrency=max_concurrency))
    if out_path is None:
        out_path = runs_path.replace(".jsonl", ".coherence.jsonl")
    n_ok = sum(1 for r in results if r.get("coherent") is True)
    n_bad = sum(1 for r in results if r.get("coherent") is False)
    n_err = sum(1 for r in results if r.get("coherent") is None)
    print(f"coherent: {n_ok}/{len(results)}  incoherent: {n_bad}  errors: {n_err}")
    with open(out_path, "w") as f:
        for row, judg in zip(rows, results):
            f.write(json.dumps({"prompt_idx": row.get("prompt_idx"),
                                 "injection_id": row.get("injection_id"),
                                 **judg}) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
