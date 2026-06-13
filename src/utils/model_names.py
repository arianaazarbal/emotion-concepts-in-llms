"""Helpers for resolving short model names to their full HuggingFace identifiers."""

from typing import Optional

from src.utils.utils import load_json

MODEL_NAMES_PATH = "data/model_names.json"
OPENROUTER_IDS_PATH = "data/openrouter_ids.json"


def resolve_model_name(short_name: str) -> str:
    """Resolve a short model name to its full HuggingFace identifier.

    Raises a ValueError if the short name is not registered in data/model_names.json.
    """
    mapping = load_json(MODEL_NAMES_PATH)
    if short_name not in mapping:
        raise ValueError(
            f"Unknown short model name: {short_name!r}. "
            f"Known names: {sorted(mapping.keys())}. "
            f"Add new entries to {MODEL_NAMES_PATH}."
        )
    return mapping[short_name]


def resolve_openrouter_id(short_name: str) -> Optional[str]:
    """Return the OpenRouter slug for ``short_name``, or None if unavailable.

    The slug format differs from the HuggingFace id (case, separator, vendor
    prefix), so we keep an explicit mapping in data/openrouter_ids.json.
    A None value means the model is not hosted on OpenRouter; callers should
    skip the probe and route directly to a local backend (vLLM).
    """
    mapping = load_json(OPENROUTER_IDS_PATH)
    if short_name not in mapping:
        raise ValueError(
            f"Unknown short model name: {short_name!r}. "
            f"Known names: {sorted(k for k in mapping if not k.startswith('_'))}. "
            f"Add new entries to {OPENROUTER_IDS_PATH}."
        )
    return mapping[short_name]
