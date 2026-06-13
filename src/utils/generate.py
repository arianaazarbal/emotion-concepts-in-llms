"""Generation dispatcher: probe OpenRouter once, fall back to local vLLM.

OpenRouter provider availability is flaky for some open-weight models and
its slug format differs from HuggingFace (case, hyphenation, vendor prefix).
Callers pass the HF identifier for local vLLM plus (optionally) the OR slug;
we probe OR once, and on any failure route the whole batch to vLLM using
the HF id.
"""

from typing import Callable, List, Optional, Tuple

from src.utils.generate_openrouter import generate_sync as openrouter_sync


def _probe_openrouter(model: str, test_prompt: str) -> bool:
    resp = openrouter_sync(model, [test_prompt], N=1, max_concurrency=1)
    return bool(resp) and bool(resp[0]) and resp[0][0] is not None


def pick_backend(
    model: str,
    test_prompt: str,
    local_only: bool = False,
    openrouter_model: Optional[str] = None,
) -> Tuple[str, Callable, str]:
    """Pick a generation backend with a single OpenRouter probe.

    ``model`` is the HuggingFace id used by the local vLLM backend.
    ``openrouter_model`` is the OR slug; when None, we skip the OR probe
    (the model is not on OR).

    Returns ``(source, generate_sync, effective_model)``. Call
    ``generate_sync(effective_model, prompts, ...)`` so OR gets its slug
    and vLLM gets the HF id.
    """
    if not local_only and openrouter_model is not None:
        print(f"[generate] probing openrouter:{openrouter_model}")
        if _probe_openrouter(openrouter_model, test_prompt):
            print(f"[generate] using openrouter:{openrouter_model}")
            return f"openrouter:{openrouter_model}", openrouter_sync, openrouter_model
        print(f"[generate] openrouter:{openrouter_model} unavailable, falling back to vllm")
    elif not local_only and openrouter_model is None:
        print(f"[generate] no openrouter id registered for {model}, using vllm")
    else:
        print(f"[generate] local_only=True, using vllm:{model}")
    from src.utils.generate_vllm import generate_sync as vllm_sync
    return f"vllm:{model}", vllm_sync, model


def generate_with_fallback(
    model: str,
    prompts: List[str],
    N: int = 1,
    local_only: bool = False,
    openrouter_model: Optional[str] = None,
    **kwargs,
) -> Tuple[str, List[List[Optional[str]]]]:
    """Pick a backend and run the batch through it once."""
    if not prompts:
        return "openrouter:noop", []
    source, generate_sync, effective_model = pick_backend(
        model, prompts[0], local_only=local_only, openrouter_model=openrouter_model
    )
    return source, generate_sync(effective_model, prompts, N=N, **kwargs)
