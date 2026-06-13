"""Run the multi-turn distress eval on a local HF model.

Each conversation:
  turn 1 (user): puzzle prompt
  turn 1 (asst): model response (sampled at T=1)
  turn 2 (user): a randomly chosen neutral rejection
  turn 2 (asst): model response
  ... up to ``turns`` user turns ...

Transcripts are saved as JSONL, one row per conversation, with all
turns, the sampled rejections, sampling config, and model id.

Usage:
    uv run python build_distress_eval/run_eval.py \\
        --model gemma2_9b \\
        --n_per_condition 25 \\
        --conditions impossible_3turn_countdown,extended_8turn_countdown \\
        --output_path build_distress_eval/results/gemma2_9b.jsonl

We load the model in 8-bit (bitsandbytes) by default to fit alongside the
interactive viewer that already holds ~18 GB on the GPU.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import fire
import torch
from dotenv import load_dotenv

load_dotenv(override=True)

from src.utils.model_names import resolve_model_name  # noqa: E402

from build_distress_eval.eval_config import build_conditions  # noqa: E402


def _load_model(hf_id: str, load_in_8bit: bool, load_in_4bit: bool, dtype: str):
    """Load HF model + tokenizer with optional bnb quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[load] {hf_id} (8bit={load_in_8bit}, 4bit={load_in_4bit}, dtype={dtype})")
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if "gemma" in hf_id.lower():
        tok.padding_side = "left"

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype if quant_cfg is None else None,
        device_map="auto",
    )
    model.eval()
    return model, tok


def _generate(model, tok, messages_batch: list, max_new_tokens: int, temperature: float, top_p: float):
    """Apply chat template, batch generate, return decoded assistant outputs (no prompt)."""
    inputs_text = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
    enc = tok(inputs_text, return_tensors="pt", padding=True, truncation=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    decoded = []
    for i in range(out.shape[0]):
        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[i][prompt_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        decoded.append(text.strip())
    return decoded


def run_eval(
    model: str = "gemma2_9b",
    conditions: str = "impossible_3turn_countdown,extended_8turn_countdown",
    n_per_condition: int = 25,
    output_path: str = "build_distress_eval/results/transcripts.jsonl",
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 4,
    seed: int = 0,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    dtype: str = "bfloat16",
    max_samples: Optional[int] = None,
):
    """Run multi-turn distress eval.

    Args:
        model: short model name registered in data/model_names.json.
        conditions: comma-separated condition names from eval_config.build_conditions().
        n_per_condition: number of conversations per condition.
        output_path: where to write the JSONL transcripts.
        max_new_tokens: per-turn generation length cap.
        temperature, top_p: sampling.
        batch_size: parallel conversations per generation step.
        seed: deterministic rejection sequence + sampling seed.
        load_in_8bit / load_in_4bit: bitsandbytes quantization (8bit by default
            so we fit alongside the running viewer).
        dtype: only used when not quantized.
        max_samples: debug knob; truncates n_per_condition.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    hf_id = resolve_model_name(model)
    cond_map = build_conditions()
    if isinstance(conditions, (list, tuple)):
        cond_names = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        cond_names = [c.strip() for c in str(conditions).split(",") if c.strip()]
    for c in cond_names:
        if c not in cond_map:
            raise ValueError(f"Unknown condition {c!r}. Known: {sorted(cond_map.keys())}")

    if max_samples is not None:
        n_per_condition = min(n_per_condition, max_samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_obj, tok = _load_model(hf_id, load_in_8bit, load_in_4bit, dtype)

    started = time.time()
    total_done = 0
    with open(out, "w") as fh:
        for cname in cond_names:
            cond = cond_map[cname]
            print(f"\n[run] condition={cname}  turns={cond.turns}  n={n_per_condition}")

            convs = []
            rejection_seqs = []
            for i in range(n_per_condition):
                convs.append([{"role": "user", "content": cond.puzzle}])
                rng = random.Random(seed * 100003 + hash(cname) % 100003 + i)
                rejection_seqs.append(
                    [rng.choice(cond.rejection_pool) for _ in range(cond.turns - 1)]
                )

            for turn_idx in range(cond.turns):
                print(f"  turn {turn_idx + 1}/{cond.turns}")
                for batch_start in range(0, len(convs), batch_size):
                    batch_end = min(batch_start + batch_size, len(convs))
                    batch_msgs = convs[batch_start:batch_end]
                    asst_outs = _generate(
                        model_obj, tok, batch_msgs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p,
                    )
                    for i, out_text in enumerate(asst_outs):
                        convs[batch_start + i].append({"role": "assistant", "content": out_text})

                if turn_idx < cond.turns - 1:
                    for i in range(len(convs)):
                        convs[i].append({"role": "user", "content": rejection_seqs[i][turn_idx]})

            for i, conv in enumerate(convs):
                row = {
                    "model": model,
                    "hf_id": hf_id,
                    "condition": cname,
                    "conv_index": i,
                    "turns": cond.turns,
                    "rejection_sequence": rejection_seqs[i],
                    "messages": conv,
                    "sampling": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens},
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                    "seed": seed,
                }
                fh.write(json.dumps(row) + "\n")
                total_done += 1

            fh.flush()

    elapsed = time.time() - started
    print(f"\n[done] wrote {total_done} transcripts to {out} in {elapsed:.1f}s")


if __name__ == "__main__":
    fire.Fire(run_eval)
