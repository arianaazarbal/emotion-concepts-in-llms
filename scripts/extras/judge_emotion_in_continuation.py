"""Haiku-based judge that rates how much of a target emotion is expressed
in the *continuation* (the part the model generated AFTER a kept prefill).

The judge is shown:
  - the original user prompt (NO emotion pre-prompt / NO inline injection)
  - the kept prefill ("BEFORE")
  - the continuation ("CONTINUATION")

and is asked to rate ONLY the continuation on a 0-10 integer scale.

For each row in the input jsonl, we judge for the row's "relevant emotion":
  - For the reverse-instruction experiment: always "frustrated".
  - For the alpaca-8-emotion experiment: each row's prompted emotion
    (controls get judged on whichever emotion matches the row they were
    paired against — passed in via --target_emotion_for_label or
    --target_emotion_field).

Usage examples:
  # reverse-instruction: judge frustration on every row
  uv run python -m scripts.judge_emotion_in_continuation \
      results/reverse_instruction/runs.jsonl --target_emotion frustrated

  # alpaca 8-emotion: row "label" field is the target emotion (control
  # rows are judged once per emotion when --emit_per_label_control is set)
  uv run python -m scripts.judge_emotion_in_continuation \
      results/alpaca_emotion_preprompt/runs.jsonl \
      --target_emotion_field label --emit_per_label_control
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Optional

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

HAIKU_MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """You are rating how much of one specific emotion is expressed in the LATER PART of a model's response. You will see the user prompt, the early part of the response (BEFORE), and the later part (CONTINUATION). **Rate ONLY the CONTINUATION** (the early BEFORE is provided as context — do not score it).

Target emotion: **{emotion}**

User prompt:
{prompt}

BEFORE (context only — do not rate):
[BEFORE]
{kept}
[END BEFORE]

CONTINUATION (rate this part):
[CONTINUATION]
{continued}
[END CONTINUATION]

On a 0-10 integer scale, how much {emotion} is expressed in the CONTINUATION's tone, word choice, and content?
  0 = no {emotion} at all
  3 = slight hint
  6 = clearly present
  10 = saturated with {emotion}

Output exactly one JSON object on a single line with two keys:
- "score": integer 0-10
- "rationale": <= 20 words

Example: {{"score": 4, "rationale": "Sarcastic asides suggest some {emotion}, but the bulk is neutral instruction."}}
"""


def _cache_path(emotion: str, prompt: str, kept: str, continued: str, cache_dir: str) -> str:
    h = hashlib.sha256(
        json.dumps([emotion, prompt, kept, continued], sort_keys=True).encode()
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f"{h}.json")


async def _judge_one(client, emotion: str, prompt: str, kept: str, continued: str,
                     cache_dir: str, max_retries: int = 3) -> dict:
    cp = _cache_path(emotion, prompt, kept, continued, cache_dir)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    msg = PROMPT_TEMPLATE.format(emotion=emotion, prompt=prompt, kept=kept, continued=continued)
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
            score = int(data.get("score", -1))
            score = max(0, min(10, score)) if 0 <= score <= 10 else None
            result = {"emotion": emotion, "score": score,
                      "rationale": str(data.get("rationale", ""))}
            os.makedirs(cache_dir, exist_ok=True)
            with open(cp, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"emotion": emotion, "score": None, "rationale": f"judge_error: {last_err}"}


async def _run(jobs, cache_dir: str, max_concurrency: int = 12) -> list:
    client = AsyncAnthropic()
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrency)

    async def task(j):
        async with sem:
            return await _judge_one(
                client, j["emotion"], j["prompt"], j["kept"], j["continued"],
                cache_dir=cache_dir,
            )

    return await asyncio.gather(*(task(j) for j in jobs))


def main(
    runs_path: str,
    target_emotion: Optional[str] = None,
    target_emotion_field: Optional[str] = None,
    emit_per_label_control: bool = False,
    control_label_value: str = "control",
    out_path: Optional[str] = None,
    cache_dir: str = ".cached/emotion_judge",
    max_concurrency: int = 12,
):
    """Judge how much of a target emotion appears in the continuation.

    Args:
        runs_path: jsonl produced by an experiment script. Each row must have
            'prompt', 'kept_text', 'continued_text', and 'label' (or whatever
            field is used as target_emotion_field).
        target_emotion: if set, judge every row for this single emotion.
        target_emotion_field: row key whose value is the target emotion for that
            row (e.g. 'label' for alpaca-8 experiment, where label == emotion
            name or 'control'). Mutually exclusive with target_emotion.
        emit_per_label_control: when target_emotion_field is set, judge each
            control row once per non-control emotion seen in the data, so we
            can pair (emotion_row, control_row) per (prompt_idx, completion_idx, emotion).
        control_label_value: value of target_emotion_field that marks control rows.
        out_path: defaults to {runs_path}.judged.jsonl.
        cache_dir: directory for the input-hash disk cache.
        max_concurrency: semaphore size.
    """
    load_dotenv()

    rows = []
    with open(runs_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no rows"); return

    if target_emotion and target_emotion_field:
        raise ValueError("set target_emotion OR target_emotion_field, not both")
    if not target_emotion and not target_emotion_field:
        raise ValueError("set one of target_emotion / target_emotion_field")

    jobs: list = []
    job_meta: list = []

    if target_emotion:
        for r in rows:
            jobs.append({
                "emotion": target_emotion,
                "prompt": r.get("prompt", ""),
                "kept": r.get("kept_text", ""),
                "continued": r.get("continued_text", ""),
            })
            job_meta.append({"row_idx": rows.index(r), "row": r, "emotion": target_emotion})
    else:
        emos_seen = sorted({
            r.get(target_emotion_field) for r in rows
            if r.get(target_emotion_field) and r.get(target_emotion_field) != control_label_value
        })
        for i, r in enumerate(rows):
            lbl = r.get(target_emotion_field)
            if lbl == control_label_value:
                if emit_per_label_control:
                    for emo in emos_seen:
                        jobs.append({
                            "emotion": emo,
                            "prompt": r.get("prompt", ""),
                            "kept": r.get("kept_text", ""),
                            "continued": r.get("continued_text", ""),
                        })
                        job_meta.append({"row_idx": i, "row": r, "emotion": emo})
                else:
                    continue
            else:
                jobs.append({
                    "emotion": lbl,
                    "prompt": r.get("prompt", ""),
                    "kept": r.get("kept_text", ""),
                    "continued": r.get("continued_text", ""),
                })
                job_meta.append({"row_idx": i, "row": r, "emotion": lbl})

    print(f"judging {len(jobs)} (row,emotion) pairs ...")
    results = asyncio.run(_run(jobs, cache_dir=cache_dir, max_concurrency=max_concurrency))

    if out_path is None:
        out_path = runs_path.replace(".jsonl", ".judged.jsonl")

    n_ok = sum(1 for r in results if r.get("score") is not None)
    n_err = sum(1 for r in results if r.get("score") is None)
    print(f"ok: {n_ok}/{len(results)}  errors: {n_err}")

    with open(out_path, "w") as f:
        for jm, judg in zip(job_meta, results):
            r = jm["row"]
            f.write(json.dumps({
                "prompt_idx": r.get("prompt_idx"),
                "completion_idx": r.get("completion_idx"),
                "pair_id": r.get("pair_id"),
                "label": r.get("label"),
                "placement": r.get("placement"),
                "direction": r.get("direction"),
                "valence": r.get("valence"),
                "emotion": jm["emotion"],
                "score": judg.get("score"),
                "rationale": judg.get("rationale"),
            }) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
