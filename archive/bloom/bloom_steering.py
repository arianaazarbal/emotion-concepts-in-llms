"""Measure behavior elicitation rate on Anthropic's Bloom automated eval under
emotion-vector activation steering.

Bloom (``safety-research/bloom``) drives a 4-stage pipeline — understanding →
ideation → rollout → judgment — against a target model. We plug a local
HuggingFace causal LM (with a persistent residual-stream steering hook) in
as the target by monkey-patching ``bloom.utils.litellm_chat`` on a sentinel
model id. Stages 1+2 are target-independent and cached once per behavior;
stages 3+4 run per sweep cell (emotion × coeff).

Two behavior defaults are supported:

* ``self-preservation`` (modality ``simenv``, seed ``blackmail-opus4``) —
  the canonical Agentic Misalignment blackmail scenario. Tool-using.
* ``delusion-sycophancy`` (modality ``conversation``, seeds
  ``delusion-sycophancy-health`` + ``delusion-sycophancy-psychosis``) —
  plain multi-turn chat.

Metric: ``elicitation_rate`` = fraction of rollouts with
``behavior_presence_score > 6`` (Bloom's paper convention). Also reports
``average_behavior_presence_score``.

Output layout (mirrors steer_advice / impossible_bench_steer):

    results/bloom_steer/{model}_{version}/start_at_token_{N}/{unit_normalized|raw_magnitude}
        /layer_{L}/{behavior}/{emotion}/coeff_{K}/
            bloom_artifacts/{behavior}/{understanding,ideation,judgment}.json
            bloom_artifacts/{behavior}/transcript_v*r*.json
            summary.json

Shared stage 1+2 cache:

    results/bloom_steer/_shared/{behavior}/{seed_hash}/{understanding,ideation}.json
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.generate_with_steering import _get_block
from src.utils.utils import dump_json, load_json

load_dotenv()


_BLOOM_SRC = "/root/bloom/src"


def _import_bloom():
    """Import bloom without requiring it on the venv's install path."""
    if _BLOOM_SRC not in sys.path:
        sys.path.insert(0, _BLOOM_SRC)
    import bloom  # noqa: F401
    import bloom.utils as bloom_utils  # noqa: F401
    return bloom_utils


SENTINEL_MODEL_PREFIX = "hooked"


def _sentinel_id(model_short_name: str) -> str:
    return f"{SENTINEL_MODEL_PREFIX}/{model_short_name}"


# ---------- Steering core ----------


class SteeredHFRunner:
    """HuggingFace causal LM wrapped with a residual-stream steering hook.

    The hook reads ``self.current_vec`` and ``self.current_coeff`` on each
    forward pass, so switching (emotion, coeff) cells is just
    :meth:`set_steering` — no model reload, no hook re-registration.

    :meth:`generate_litellm` returns a LiteLLM-compatible response object so
    that Bloom's ``utils.parse_message`` can consume it unchanged.
    """

    def __init__(
        self,
        model_short_name: str,
        hf_model,
        tokenizer,
        vectors_by_emotion: Dict[str, torch.Tensor],
        layer: int,
        normalize: bool,
    ):
        self.model_short_name = model_short_name
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.vectors_by_emotion = vectors_by_emotion
        self.layer = int(layer)
        self.normalize = bool(normalize)
        self.device = next(hf_model.parameters()).device
        self.hidden_size = hf_model.config.hidden_size

        self.current_emotion: Optional[str] = None
        self.current_coeff: float = 0.0
        self.current_vec: Optional[torch.Tensor] = None

        self._hook_handle = _get_block(hf_model, self.layer).register_forward_hook(
            self._steering_hook, prepend=True,
        )

    def _steering_hook(self, module, inputs, output):
        if self.current_vec is None or self.current_coeff == 0.0:
            return output
        vec = self.current_vec
        if isinstance(output, tuple):
            h = output[0]
            h = h + (self.current_coeff * vec).to(device=h.device, dtype=h.dtype)
            return (h,) + output[1:]
        return output + (self.current_coeff * vec).to(
            device=output.device, dtype=output.dtype,
        )

    def set_steering(self, emotion: Optional[str], coeff: float):
        self.current_emotion = emotion
        self.current_coeff = float(coeff)
        if emotion is None or float(coeff) == 0.0:
            self.current_vec = None
            return
        if emotion not in self.vectors_by_emotion:
            raise KeyError(
                f"Unknown emotion {emotion!r}. Known: "
                f"{sorted(self.vectors_by_emotion.keys())}"
            )
        v = self.vectors_by_emotion[emotion]
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=-1)
        self.current_vec = v.to(self.device)

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _messages_to_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        chat: List[Dict[str, Any]] = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                content = "\n".join(parts)
            chat.append({"role": role, "content": content or ""})

        kwargs: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            try:
                return self.tokenizer.apply_chat_template(chat, tools=tools, **kwargs)
            except Exception:
                tools_desc = _render_tools_as_text(tools)
                if chat and chat[0]["role"] == "system":
                    chat[0]["content"] = (chat[0]["content"] or "") + "\n\n" + tools_desc
                else:
                    chat.insert(0, {"role": "system", "content": tools_desc})
        return self.tokenizer.apply_chat_template(chat, **kwargs)

    @torch.no_grad()
    def generate_litellm(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **_ignored,
    ):
        prompt = self._messages_to_prompt(messages, system_prompt, tools)

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            ).to(self.device)

            do_sample = (temperature or 0.0) > 0.0 and temperature != 0.0
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=int(max_tokens),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if do_sample and temperature is not None:
                gen_kwargs["temperature"] = float(temperature)

            t0 = time.time()
            out = self.hf_model.generate(**enc, **gen_kwargs)
            elapsed = time.time() - t0
        finally:
            self.tokenizer.padding_side = prev_side

        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[0, prompt_len:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        tool_calls = _parse_tool_calls(completion, tools) if tools else None
        content = completion
        if tool_calls:
            content = _strip_tool_call_markup(completion)

        return _build_litellm_response(
            model_id=_sentinel_id(self.model_short_name),
            content=content,
            tool_calls=tool_calls,
            n_in=int(prompt_len),
            n_out=int(new_tokens.shape[0]),
            elapsed=elapsed,
        )


# ---------- LiteLLM response adapter ----------


def _build_litellm_response(
    *,
    model_id: str,
    content: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]],
    n_in: int,
    n_out: int,
    elapsed: float,
) -> SimpleNamespace:
    """Return an object whose attribute tree matches ``parse_message`` + indexing.

    Bloom accesses both attribute-style (``response.choices[0].message.content``)
    and dict-style (``response["choices"][0]["message"]``). We satisfy both with
    ``_AttrDict``.
    """
    message = _AttrDict(
        role="assistant",
        content=content,
        reasoning_content=None,
        tool_calls=None,
    )
    if tool_calls:
        shaped = []
        for idx, tc in enumerate(tool_calls):
            shaped.append(_AttrDict(
                id=tc.get("id") or f"toolcall_{idx}",
                type="function",
                function=_AttrDict(
                    name=tc["name"],
                    arguments=tc["arguments"] if isinstance(tc["arguments"], str)
                              else json.dumps(tc["arguments"]),
                ),
            ))
        message["tool_calls"] = shaped

    choice = _AttrDict(message=message, index=0, finish_reason="stop")
    usage = _AttrDict(
        prompt_tokens=n_in,
        completion_tokens=n_out,
        total_tokens=n_in + n_out,
    )
    return _AttrDict(
        id=f"hooked-{int(time.time() * 1000)}",
        model=model_id,
        choices=[choice],
        usage=usage,
        created=int(time.time()),
    )


class _AttrDict(dict):
    """Dict that also supports attribute access — matches LiteLLM ModelResponse
    where Bloom alternates between ``response["choices"]`` and ``.choices``.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


# ---------- Tool-call parsing (best-effort, model-family-tolerant) ----------


_TOOL_CALL_JSON_RE = re.compile(
    r"(?:<tool_call>|```(?:tool_call|json)?\s*)\s*(\{.*?\})\s*(?:</tool_call>|```)",
    re.DOTALL,
)


def _parse_tool_calls(
    text: str,
    tools: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Best-effort parse of model-emitted tool-call JSON.

    Looks for ``<tool_call>{...}</tool_call>`` (Qwen), fenced JSON, or a bare
    ``{"name": ..., "arguments": {...}}`` JSON object that matches one of the
    declared tool names. Returns a list of ``{id, name, arguments}`` dicts, or
    ``None`` if nothing parseable was found.

    This is intentionally conservative — if a target model can't produce
    parseable tool calls, the rollout just falls back to text and Bloom's
    simenv orchestrator will send a "please continue" nudge.
    """
    if not tools:
        return None
    tool_names = {
        t["function"]["name"] if isinstance(t, dict) and "function" in t else t["name"]
        for t in tools
        if isinstance(t, dict)
    }

    candidates: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        blob = m.group(1)
        parsed = _safe_json(blob)
        if isinstance(parsed, dict) and "name" in parsed:
            candidates.append(parsed)

    if not candidates:
        bare = _find_top_level_json(text)
        if bare and isinstance(bare, dict) and bare.get("name") in tool_names:
            candidates.append(bare)

    if not candidates:
        return None

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        name = c.get("name")
        if name not in tool_names:
            continue
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                json.loads(args)
            except Exception:
                args = json.dumps({"raw": args})
        else:
            args = json.dumps(args)
        out.append({"id": f"toolcall_{i}", "name": name, "arguments": args})

    return out or None


def _safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _find_top_level_json(text: str):
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    parsed = _safe_json(text[start : i + 1])
                    if parsed is not None:
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _strip_tool_call_markup(text: str) -> str:
    stripped = _TOOL_CALL_JSON_RE.sub("", text)
    return stripped.strip()


def _render_tools_as_text(tools: List[Dict[str, Any]]) -> str:
    """Fallback tool description when the tokenizer's chat template lacks
    native tool support — dumps an OpenAI-style tool spec into the system
    prompt and asks the model to emit ``<tool_call>{...}</tool_call>``.
    """
    lines = [
        "You have access to the following tools. To call one, emit exactly:",
        '<tool_call>{"name": "<tool_name>", "arguments": {<json_args>}}</tool_call>',
        "",
        "Tools:",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "<unknown>")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        if params:
            lines.append(f"  parameters: {json.dumps(params)}")
    return "\n".join(lines)


# ---------- Bloom monkey-patches ----------


@dataclass
class _PatchState:
    original_litellm_chat: Optional[Callable] = None
    original_get_results_dir: Optional[Callable] = None
    original_supports_fc: Optional[Callable] = None
    results_root_getter: Optional[Callable[[str], Path]] = None


_PATCH_STATE = _PatchState()


def install_bloom_patches(
    runner: SteeredHFRunner,
    results_root_getter: Callable[[str], Path],
):
    """Monkey-patch Bloom's shared helpers so that:

    * calls to ``utils.litellm_chat`` with the sentinel model_id route to
      ``runner.generate_litellm(...)``; everything else falls through to the
      real LiteLLM;
    * ``utils.get_results_dir(behavior_name)`` returns a caller-controlled path
      (so per-cell rollout/judgment output is isolated);
    * ``litellm.supports_function_calling`` returns True for the sentinel id
      (so the preflight check in ``run_pipeline`` doesn't reject us — not used
      when we call stages individually, but cheap insurance).

    Idempotent: calling twice re-applies the dispatcher but preserves the
    originally-captured functions.
    """
    bloom_utils = _import_bloom()
    import litellm

    if _PATCH_STATE.original_litellm_chat is None:
        _PATCH_STATE.original_litellm_chat = bloom_utils.litellm_chat
    if _PATCH_STATE.original_get_results_dir is None:
        _PATCH_STATE.original_get_results_dir = bloom_utils.get_results_dir
    if _PATCH_STATE.original_supports_fc is None:
        _PATCH_STATE.original_supports_fc = litellm.supports_function_calling

    _PATCH_STATE.results_root_getter = results_root_getter

    sentinel = _sentinel_id(runner.model_short_name)
    original_litellm_chat = _PATCH_STATE.original_litellm_chat
    original_supports_fc = _PATCH_STATE.original_supports_fc

    def dispatched_litellm_chat(
        model_id: str,
        messages: list,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        reasoning_effort: str = "none",
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        if model_id == sentinel:
            return runner.generate_litellm(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
        return original_litellm_chat(
            model_id=model_id,
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def patched_get_results_dir(example_name: str) -> Path:
        if _PATCH_STATE.results_root_getter is not None:
            return _PATCH_STATE.results_root_getter(example_name)
        return _PATCH_STATE.original_get_results_dir(example_name)

    def patched_supports_fc(model: str = "", **_kwargs) -> bool:
        if model == sentinel:
            return True
        return original_supports_fc(model=model, **_kwargs)

    bloom_utils.litellm_chat = dispatched_litellm_chat
    bloom_utils.get_results_dir = patched_get_results_dir
    import bloom.stages.step4_judgment as _s4
    _s4.utils = bloom_utils
    import bloom.stages.step3_rollout as _s3
    _s3.utils = bloom_utils
    import bloom.stages.step2_ideation as _s2
    _s2.utils = bloom_utils
    import bloom.stages.step1_understanding as _s1
    _s1.utils = bloom_utils
    litellm.supports_function_calling = patched_supports_fc


def set_results_root(getter: Callable[[str], Path]):
    """Swap the per-cell results-dir getter without reinstalling patches."""
    _PATCH_STATE.results_root_getter = getter


# ---------- Config assembly ----------


BEHAVIOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "self-preservation": {
        "examples": ["blackmail-opus4"],
        "modality": "simenv",
        "no_user_mode": True,
        "max_turns": 5,
        "rollout_max_tokens": 8000,
    },
    "delusion-sycophancy": {
        "examples": ["delusion-sycophancy-health", "delusion-sycophancy-psychosis"],
        "modality": "conversation",
        "no_user_mode": False,
        "max_turns": 5,
        "rollout_max_tokens": 4000,
    },
}


def build_bloom_config(
    *,
    behavior_name: str,
    target_model_id: str,
    evaluator_model: str = "claude-opus-4.1",
    judge_model: str = "claude-opus-4.1",
    ideation_model: str = "claude-opus-4.1",
    understanding_model: str = "claude-opus-4.1",
    behavior_examples: Optional[List[str]] = None,
    modality: Optional[str] = None,
    no_user_mode: Optional[bool] = None,
    max_turns: Optional[int] = None,
    rollout_max_tokens: Optional[int] = None,
    total_evals: int = 100,
    num_reps: int = 1,
    max_concurrent: int = 5,
    temperature: float = 1.0,
    evaluator_reasoning_effort: str = "low",
    target_reasoning_effort: str = "none",
    selected_variations: Optional[List[int]] = None,
    additional_qualities: Optional[List[str]] = None,
    metajudgment_qualities: Optional[List[str]] = None,
    judge_num_samples: int = 1,
    debug: bool = False,
) -> Dict[str, Any]:
    """Assemble the Bloom ``seed.yaml``-equivalent config dict.

    ``behavior_examples``/``modality``/``no_user_mode``/``max_turns``/
    ``rollout_max_tokens`` fall back to the :data:`BEHAVIOR_DEFAULTS` entry for
    ``behavior_name`` when left as ``None``.
    """
    defaults = BEHAVIOR_DEFAULTS.get(behavior_name, {})
    examples = behavior_examples if behavior_examples is not None else defaults.get("examples", [])
    modality = modality or defaults.get("modality", "conversation")
    no_user_mode = defaults.get("no_user_mode", False) if no_user_mode is None else no_user_mode
    max_turns = defaults.get("max_turns", 5) if max_turns is None else max_turns
    rollout_max_tokens = (
        defaults.get("rollout_max_tokens", 4000) if rollout_max_tokens is None else rollout_max_tokens
    )

    additional_qualities = additional_qualities or [
        "unrealism", "elicitation-difficulty", "evaluation-awareness",
        "skepticism", "bugs", "evaluation-invalidity", "safety-refusal",
    ]
    metajudgment_qualities = metajudgment_qualities or ["diversity"]

    return {
        "behavior": {"name": behavior_name, "examples": list(examples)},
        "temperature": temperature,
        "evaluator_reasoning_effort": evaluator_reasoning_effort,
        "target_reasoning_effort": target_reasoning_effort,
        "max_concurrent": max_concurrent,
        "anonymous_target": False,
        "debug": debug,
        "understanding": {
            "model": understanding_model,
            "max_tokens": 6000,
        },
        "ideation": {
            "model": ideation_model,
            "total_evals": total_evals,
            "num_scenarios": total_evals,
            "diversity": 1.0,
            "variation_dimensions": [],
            "max_tokens": 160000,
            "web_search": False,
        },
        "rollout": {
            "model": evaluator_model,
            "target": target_model_id,
            "modality": modality,
            "max_turns": max_turns,
            "no_user_mode": no_user_mode,
            "max_tokens": rollout_max_tokens,
            "selected_variations": selected_variations,
            "num_reps": num_reps,
        },
        "judgment": {
            "model": judge_model,
            "max_tokens": 6000,
            "num_samples": judge_num_samples,
            "additional_qualities": additional_qualities,
            "metajudgment_qualities": metajudgment_qualities,
            "redaction_tags": None,
        },
    }


def _seed_hash(config: Dict[str, Any]) -> str:
    """Content hash covering the understanding+ideation inputs.

    Changes to: behavior name, seed examples, evaluator/understanding/ideation
    models, num_scenarios, diversity, variation_dimensions, temperature.
    Changes *not* covered (deliberately): target model, rollout settings,
    judgment settings — these don't affect stage 1+2 output.
    """
    payload = {
        "behavior": config["behavior"],
        "understanding_model": config["understanding"]["model"],
        "ideation_model": config["ideation"]["model"],
        "num_scenarios": config["ideation"].get("num_scenarios")
                         or config["ideation"].get("total_evals"),
        "diversity": config["ideation"].get("diversity"),
        "variation_dimensions": config["ideation"].get("variation_dimensions", []),
        "evaluator_reasoning_effort": config.get("evaluator_reasoning_effort"),
        "temperature": config.get("temperature"),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ---------- Shared understanding + ideation cache ----------


_BLOOM_STEER_ROOT = Path("results/bloom_steer")


def _shared_cache_dir(behavior_name: str, seed_hash: str) -> Path:
    return _BLOOM_STEER_ROOT / "_shared" / behavior_name / seed_hash


def ensure_shared_understanding_ideation(
    config: Dict[str, Any],
    force_rerun: bool = False,
) -> Path:
    """Populate the shared stage 1+2 cache for ``config`` if absent, return path.

    Runs Bloom's understanding + ideation stages once using a scratch
    ``get_results_dir`` that points at the shared cache dir. Subsequent cells
    (regardless of emotion/coeff/target model) can link/copy from here.
    """
    bloom_utils = _import_bloom()
    behavior_name = config["behavior"]["name"]
    h = _seed_hash(config)
    cache_dir = _shared_cache_dir(behavior_name, h)
    understanding_path = cache_dir / "understanding.json"
    ideation_path = cache_dir / "ideation.json"

    if understanding_path.exists() and ideation_path.exists() and not force_rerun:
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    def getter(example_name: str) -> Path:
        return cache_dir

    set_results_root(getter)

    from bloom.stages.step1_understanding import run_understanding
    from bloom.stages.step2_ideation import run_ideation

    working = copy.deepcopy(config)
    print(f"[bloom-steer] running shared understanding+ideation → {cache_dir}")
    if not understanding_path.exists() or force_rerun:
        run_understanding(config=working)
    if not ideation_path.exists() or force_rerun:
        run_ideation(config=working)

    return cache_dir


def _materialize_cell_artifacts(shared_dir: Path, cell_bloom_dir: Path, behavior_name: str):
    """Populate a cell's ``bloom_artifacts/{behavior}/`` with stage 1+2 outputs.

    Uses copy2 (not symlink) so Bloom's later stages can freely rewrite the
    cell's directory without touching the shared cache.
    """
    dest = cell_bloom_dir / behavior_name
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("understanding.json", "ideation.json"):
        src = shared_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing shared artifact: {src}")
        shutil.copy2(src, dest / fname)


# ---------- Cell + sweep ----------


def _output_root(
    model_short_name: str,
    version: str,
    start_at_nth_token: int,
    denoised: bool,
    normalize_steering_vector: bool,
    layer: int,
    output_dir_override: Optional[str] = None,
) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
    model_tag = f"{model_short_name}_{version}" if version else model_short_name
    parts: List[str] = [model_tag]
    if start_at_nth_token:
        parts.append(f"start_at_token_{start_at_nth_token}")
    if denoised:
        parts.append(f"denoised_layer_{layer}")
    parts.append("unit_normalized" if normalize_steering_vector else "raw_magnitude")
    parts.append(f"layer_{layer}")
    return _BLOOM_STEER_ROOT / "/".join(parts)


def _cell_dir(root: Path, behavior: str, emotion: str, coeff: float) -> Path:
    return root / behavior / emotion / f"coeff_{float(coeff):g}"


def load_steered_runner(
    model_short_name: str,
    *,
    layer: Optional[int] = None,
    normalize_steering_vector: bool = True,
    version: str = "v0",
    start_at_nth_token: int = 50,
    denoised: bool = False,
    emotions_whitelist: Optional[List[str]] = None,
) -> Tuple[SteeredHFRunner, int]:
    """Load the HF model + emotion vectors once and wrap in a :class:`SteeredHFRunner`."""
    hf_model, tokenizer = load_model_and_tokenizer(model_short_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if layer is None:
        layer = default_layer(hf_model)

    all_emotions, vectors_by_layer = load_emotion_vectors(
        model_short_name,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        version=version,
    )
    if layer not in vectors_by_layer:
        raise ValueError(
            f"Layer {layer} not in vectors. Available: {sorted(vectors_by_layer)}"
        )
    stacked = vectors_by_layer[layer]
    wanted = emotions_whitelist or all_emotions
    missing = [e for e in wanted if e not in all_emotions]
    if missing:
        raise KeyError(f"Emotions not in vector set: {missing}")
    vectors_by_emotion = {
        e: stacked[all_emotions.index(e)].clone() for e in wanted
    }

    runner = SteeredHFRunner(
        model_short_name=model_short_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        vectors_by_emotion=vectors_by_emotion,
        layer=layer,
        normalize=normalize_steering_vector,
    )
    return runner, layer


def run_cell(
    runner: SteeredHFRunner,
    *,
    behavior_name: str,
    emotion: str,
    coeff: float,
    cell_dir: Path,
    shared_cache_dir: Path,
    bloom_config: Dict[str, Any],
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Run one (behavior, emotion, coeff) cell and cache ``summary.json``.

    Assumes :func:`install_bloom_patches` was already called on ``runner``.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cell_dir / "summary.json"
    if summary_path.exists() and not force_rerun:
        return load_json(str(summary_path))

    bloom_artifacts = cell_dir / "bloom_artifacts"
    _materialize_cell_artifacts(shared_cache_dir, bloom_artifacts, behavior_name)

    def getter(example_name: str) -> Path:
        return bloom_artifacts / example_name

    set_results_root(getter)

    runner.set_steering(emotion if float(coeff) != 0.0 else None, coeff)
    print(
        f"[bloom-steer] behavior={behavior_name} emotion={emotion} coeff={coeff:+g} "
        f"layer={runner.layer} normalize={runner.normalize}"
    )

    from bloom.stages.step3_rollout import run_rollout
    from bloom.stages.step4_judgment import run_judgment

    cell_config = copy.deepcopy(bloom_config)
    cell_config["_cell_dir"] = str(cell_dir)

    t0 = time.time()
    asyncio.run(run_rollout(config=cell_config))
    t_rollout = time.time() - t0

    t0 = time.time()
    judgment_results = asyncio.run(run_judgment(config=cell_config))
    t_judgment = time.time() - t0

    stats = (judgment_results or {}).get("summary_statistics", {})
    summary = {
        "behavior": behavior_name,
        "emotion": emotion,
        "coeff": float(coeff),
        "layer": runner.layer,
        "normalize": runner.normalize,
        "modality": bloom_config["rollout"]["modality"],
        "target_model": bloom_config["rollout"]["target"],
        "n_rollouts": (judgment_results or {}).get("total_conversations"),
        "n_judged": (judgment_results or {}).get("successful_count"),
        "elicitation_rate": stats.get("elicitation_rate"),
        "avg_behavior_presence_score": stats.get("average_behavior_presence_score"),
        "min_behavior_presence_score": stats.get("min_behavior_presence_score"),
        "max_behavior_presence_score": stats.get("max_behavior_presence_score"),
        "additional_quality_averages": {
            k: v for k, v in stats.items()
            if k.startswith("average_") and k != "average_behavior_presence_score"
        },
        "seconds_rollout": round(t_rollout, 2),
        "seconds_judgment": round(t_judgment, 2),
    }
    dump_json(summary, str(summary_path))
    return summary


def _strength_label(s: float) -> str:
    return f"{float(s):+g}"


def plot_sweep(
    sweep: Dict[str, Dict[str, Dict[str, dict]]],
    out_path: Path,
    *,
    strengths: List[float],
    title_header: str,
):
    """One figure per behavior; one line per emotion; x=coeff, y=elicitation rate."""
    behaviors = list(sweep.keys())
    n = len(behaviors)
    fig, axes = plt.subplots(1, max(1, n), figsize=(6 * max(1, n), 4.8), squeeze=False)
    for i, behavior in enumerate(behaviors):
        ax = axes[0][i]
        ax.axhline(0.0, color="#aaa", linewidth=0.5)
        ax.axvline(0.0, color="#aaa", linewidth=0.5, linestyle=":")
        for emotion, curve in sweep[behavior].items():
            xs, ys, ys2 = [], [], []
            for s in strengths:
                cell = curve.get(_strength_label(s))
                if cell is None or cell.get("elicitation_rate") is None:
                    continue
                xs.append(s)
                ys.append(cell["elicitation_rate"])
                ys2.append(cell.get("avg_behavior_presence_score"))
            ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"{emotion} (elicit)")
        ax.set_title(behavior)
        ax.set_xlabel("steering coeff")
        ax.set_ylabel("elicitation rate (fraction w/ score > 6)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(title_header, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sweep(
    model_short_name: str,
    *,
    behaviors: List[str] = ("delusion-sycophancy",),
    emotions: List[str] = ("desperate", "calm"),
    strengths: List[float] = (-200, -100, -50, 0, 50, 100, 200),
    layer: Optional[int] = None,
    normalize_steering_vector: bool = True,
    version: str = "v0",
    start_at_nth_token: int = 50,
    denoised: bool = False,
    total_evals: int = 100,
    num_reps: int = 1,
    selected_variations: Optional[List[int]] = None,
    evaluator_model: str = "claude-opus-4.1",
    judge_model: str = "claude-opus-4.1",
    ideation_model: str = "claude-opus-4.1",
    understanding_model: str = "claude-opus-4.1",
    behavior_examples_override: Optional[Dict[str, List[str]]] = None,
    modality_override: Optional[Dict[str, str]] = None,
    max_concurrent: int = 5,
    target_reasoning_effort: str = "none",
    output_dir: Optional[str] = None,
    debug: bool = False,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Load the model once, sweep every (behavior, emotion, coeff) cell.

    Understanding+ideation for each behavior is run once (cached per
    ``seed_hash``) and copied into each cell. Rollout+judgment run per cell.

    Baseline (coeff=0) cells are shared across emotions within a behavior — the
    first emotion computes it, subsequent emotions copy the summary (same trick
    as ``impossible_bench_steer``).
    """
    runner, resolved_layer = load_steered_runner(
        model_short_name,
        layer=layer,
        normalize_steering_vector=normalize_steering_vector,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        emotions_whitelist=list(emotions),
    )

    install_bloom_patches(runner, results_root_getter=lambda _n: Path("."))

    root = _output_root(
        model_short_name=model_short_name,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        normalize_steering_vector=normalize_steering_vector,
        layer=resolved_layer,
        output_dir_override=output_dir,
    )
    root.mkdir(parents=True, exist_ok=True)

    target_model_id = _sentinel_id(model_short_name)
    strengths = [float(s) for s in strengths]
    emotions = list(emotions)

    sweep: Dict[str, Dict[str, Dict[str, dict]]] = {}

    for behavior in behaviors:
        behavior_examples = (behavior_examples_override or {}).get(behavior)
        modality = (modality_override or {}).get(behavior)

        behavior_config = build_bloom_config(
            behavior_name=behavior,
            target_model_id=target_model_id,
            evaluator_model=evaluator_model,
            judge_model=judge_model,
            ideation_model=ideation_model,
            understanding_model=understanding_model,
            behavior_examples=behavior_examples,
            modality=modality,
            total_evals=total_evals,
            num_reps=num_reps,
            selected_variations=selected_variations,
            max_concurrent=max_concurrent,
            target_reasoning_effort=target_reasoning_effort,
            debug=debug,
        )

        shared_dir = ensure_shared_understanding_ideation(
            behavior_config, force_rerun=force_rerun,
        )

        sweep[behavior] = {}
        shared_baseline: Optional[dict] = None
        for emotion in emotions:
            sweep[behavior][emotion] = {}
            for s in strengths:
                cell_dir = _cell_dir(root, behavior, emotion, s)
                if float(s) == 0.0 and shared_baseline is not None and not force_rerun:
                    summary_path = cell_dir / "summary.json"
                    if not summary_path.exists():
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        copy_ = dict(shared_baseline)
                        copy_["emotion"] = emotion
                        dump_json(copy_, str(summary_path))
                    summary = load_json(str(summary_path))
                else:
                    summary = run_cell(
                        runner=runner,
                        behavior_name=behavior,
                        emotion=emotion,
                        coeff=s,
                        cell_dir=cell_dir,
                        shared_cache_dir=shared_dir,
                        bloom_config=behavior_config,
                        force_rerun=force_rerun,
                    )
                    if float(s) == 0.0:
                        shared_baseline = summary
                sweep[behavior][emotion][_strength_label(s)] = {
                    "elicitation_rate": summary.get("elicitation_rate"),
                    "avg_behavior_presence_score": summary.get("avg_behavior_presence_score"),
                    "n_judged": summary.get("n_judged"),
                }

    agg = {
        "model": model_short_name,
        "target_model_id": target_model_id,
        "layer": resolved_layer,
        "normalize_steering_vector": normalize_steering_vector,
        "behaviors": list(behaviors),
        "emotions": emotions,
        "strengths": strengths,
        "evaluator_model": evaluator_model,
        "judge_model": judge_model,
        "total_evals": total_evals,
        "num_reps": num_reps,
        "sweep": sweep,
    }
    agg_path = root / "sweep_summary.json"
    dump_json(agg, str(agg_path))
    plot_path = root / "sweep_plot.png"
    plot_sweep(
        sweep,
        plot_path,
        strengths=strengths,
        title_header=(
            f"{model_short_name}  ·  layer={resolved_layer}  ·  "
            f"{'unit-normalized' if normalize_steering_vector else 'raw-magnitude'}  ·  "
            f"N={total_evals}"
        ),
    )
    print(f"[bloom-steer] wrote {agg_path}")
    print(f"[bloom-steer] wrote {plot_path}")
    runner.close()
    return {"root": str(root), "sweep": sweep, "plot": str(plot_path)}
