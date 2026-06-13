"""On-disk cache for OpenRouter generations.

Wraps :func:`src.utils.generate_openrouter.generate_sync` with a JSON file
keyed by the (model, sampling-params, prompt-list) tuple. If a request with
the same key was already made, the cached result is reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, List, Optional

from src.utils.generate_openrouter import generate_sync


def _cache_key(model: str, prompts: List[str], n: int, sampling_params: dict) -> str:
    payload = json.dumps(
        {"model": model, "prompts": prompts, "n": n, "sampling_params": sampling_params},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cached_generate(
    *,
    model: str,
    prompts: List[str],
    n: int,
    cache_dir: str,
    sampling_params: Optional[dict] = None,
    max_concurrency: int = 200,
    force_regenerate: bool = False,
) -> List[List[Optional[str]]]:
    """Generate ``n`` completions per prompt via OpenRouter, with disk cache.

    Args:
        model: OpenRouter model id.
        prompts: List of user prompts.
        n: Completions per prompt.
        cache_dir: Directory to write/read the cache JSON.
        sampling_params: Extra kwargs forwarded to OpenRouter (temperature,
            top_p, max_tokens, etc.). Default ``{}``.
        max_concurrency: Concurrent in-flight requests.
        force_regenerate: If True, ignore any existing cache entry.

    Returns:
        List (per prompt) of lists (length ``n``) of completion strings (or
        ``None`` when the request failed).
    """
    sampling_params = dict(sampling_params or {})
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(model, prompts, n, sampling_params)
    cache_path = os.path.join(cache_dir, f"openrouter_{key}.json")

    if not force_regenerate and os.path.exists(cache_path):
        with open(cache_path) as f:
            entry = json.load(f)
        print(f"[or_cache] hit: {cache_path} ({len(entry['responses'])} prompts)")
        return entry["responses"]

    responses = generate_sync(
        model=model,
        prompts=prompts,
        N=n,
        max_concurrency=max_concurrency,
        **sampling_params,
    )
    payload: dict[str, Any] = {
        "model": model,
        "n": n,
        "sampling_params": sampling_params,
        "prompts": prompts,
        "responses": responses,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[or_cache] wrote: {cache_path}")
    return responses
