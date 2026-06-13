"""Local batched generation via vLLM.

Mirrors the interface of generate_openrouter.generate_sync so it can be used
as a drop-in fallback when OpenRouter has no endpoints for a model.

The vLLM engine is loaded and torn down inside each call, which makes this
suitable for one-off bulk generation jobs but wasteful for interactive per-
prompt use. Callers should batch their prompts and cache outputs themselves.
"""

import gc
from typing import List, Optional

import torch


def _patch_transformers_5x_for_vllm_07x() -> None:
    """Shim removed attrs so vllm 0.7.x can import under transformers 5.x.

    transformers 5.x moved several top-level config fields into nested dicts
    (e.g. ``config.rope_theta`` → ``config.rope_parameters["rope_theta"]``)
    and dropped ``tokenizer.all_special_tokens_extended``. vllm 0.7.3 still
    reads both, so we install a lightweight proxy on ``PretrainedConfig``
    that falls back to the nested dicts, plus a tokenizer-level alias.
    """
    from transformers import PretrainedConfig, PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    if not getattr(PretrainedConfig, "_vllm07x_rope_compat_installed", False):
        _ROPE_ALIASES = {
            "rope_theta": ("rope_parameters", "rope_theta"),
            "rope_scaling": ("rope_parameters", "rope_scaling"),
            "rope_type": ("rope_parameters", "rope_type"),
        }

        def _patched_getattr(self, key):
            alias = _ROPE_ALIASES.get(key)
            if alias is not None:
                nested = self.__dict__.get(alias[0])
                if isinstance(nested, dict) and alias[1] in nested:
                    return nested[alias[1]]
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {key!r}"
            )

        PretrainedConfig.__getattr__ = _patched_getattr
        PretrainedConfig._vllm07x_rope_compat_installed = True


def generate_sync(
    model: str,
    prompts: List[str],
    N: int = 1,
    max_samples: Optional[int] = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
    **_ignored,
) -> List[List[Optional[str]]]:
    """Generate N completions per prompt using a local vLLM engine.

    Args:
        model: HuggingFace model identifier (e.g. "google/gemma-2-9b-it").
        prompts: User-turn prompts; chat templates are applied per model.
        N: Completions per prompt (vLLM's native `n` sampling).
        max_samples: Optional truncation of `prompts` for testing.
        max_tokens: Completion length cap.
        temperature, top_p, seed: Standard sampling knobs.
        gpu_memory_utilization: Fraction of GPU memory vLLM may claim.
        dtype: Weight dtype ("bfloat16" / "float16" / "auto").
        max_model_len: Optional cap on context window (saves KV-cache memory).
        **_ignored: swallow args like `max_concurrency` so callers can pass
            the same kwargs they use for generate_openrouter.generate_sync.

    Returns:
        List (per prompt) of lists (length N) of response strings. None
        entries fill gaps if vLLM returns fewer than N completions (rare).
    """
    _patch_transformers_5x_for_vllm_07x()
    from vllm import LLM, SamplingParams

    if max_samples is not None:
        prompts = prompts[:max_samples]

    print(f"[vllm] loading {model} (dtype={dtype}, gpu_util={gpu_memory_utilization})")
    llm = LLM(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    sampling = SamplingParams(
        n=N,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    print(f"[vllm] generating {len(prompts)} prompts x {N} completions")
    outputs = llm.generate(formatted, sampling_params=sampling)

    results = []
    for out in outputs:
        responses: List[Optional[str]] = [c.text for c in out.outputs]
        while len(responses) < N:
            responses.append(None)
        results.append(responses)

    print("[vllm] releasing engine")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"[vllm] finished {len(prompts)} prompts")
    return results
