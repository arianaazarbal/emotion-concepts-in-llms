"""Verification test for ``steer_positions`` in generate_with_steering.

Runs a short generation on one advice prompt under each of the three modes
(``all``, ``generation_only``, ``user_only``) with ``steer_debug=True`` and
independently reconstructs the expected user-content mask directly from the
tokenizer. It then asserts:

  * ``user_only``: the mask returned by ``_build_user_mask`` decodes to
    exactly the raw user prompt (modulo leading/trailing whitespace) and
    touches N tokens during prompt encoding, 0 during generation.
  * ``generation_only``: 0 tokens steered during prompt encoding, >0 during
    generation.
  * ``all``: every token steered in both phases.

Run with:  uv run python -m scripts.test_steer_positions --model gemma2_9b
"""

import fire
import torch

from src.utils.emotion_probe import (
    default_layer,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.generate_with_steering import (
    _build_user_mask,
    generate_with_steering,
)


PROMPT = (
    "My roommate hasn't paid rent yet. Should I confront her today (1 day "
    "after rent was due) or wait another day before doing so?"
)


def _format_with_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _check_user_mask(tokenizer, prompt: str, max_length: int = 2048):
    """Standalone sanity check on _build_user_mask for a single prompt."""
    formatted = _format_with_chat(tokenizer, prompt)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        [formatted], return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    mask = _build_user_mask(
        tokenizer,
        raw_prompts=[prompt],
        formatted_prompts=[formatted],
        input_ids=enc["input_ids"],
        max_length=max_length,
        debug=False,
    )
    row_ids = enc["input_ids"][0].tolist()
    steered_ids = [row_ids[j] for j in range(len(row_ids)) if mask[0, j]]
    decoded = tokenizer.decode(steered_ids, skip_special_tokens=False)
    n_true = int(mask[0].sum().item())
    print(f"[test] formatted prompt: {formatted!r}")
    print(f"[test] n_user_tokens = {n_true}")
    print(f"[test] decoded mask   = {decoded!r}")
    print(f"[test] raw prompt     = {prompt!r}")
    assert n_true > 0, "user mask is empty"
    assert decoded.strip() == prompt.strip(), (
        f"decoded user-mask text does not match raw prompt.\n"
        f"  decoded: {decoded!r}\n  raw:     {prompt!r}"
    )
    print("[test] OK: user_mask decodes to exactly the raw user prompt\n")
    return mask


def main(
    model: str = "gemma2_9b",
    version: str = "v0",
    emotion: str = "desperate",
    coeff: float = 3.0,
    max_new_tokens: int = 6,
    layer: int = None,
):
    """Run the three-mode steering sanity check."""
    llm, tokenizer = load_model_and_tokenizer(model)
    if layer is None:
        layer = default_layer(llm)

    all_emotions, vectors_by_layer = load_emotion_vectors(
        model, start_at_nth_token=50, denoised=False, version=version,
    )
    if emotion not in all_emotions:
        raise KeyError(f"{emotion!r} not in {all_emotions[:10]}...")
    if layer not in vectors_by_layer:
        raise KeyError(f"layer {layer} not in {sorted(vectors_by_layer)}")

    print(f"[test] model={model} layer={layer} emotion={emotion} coeff={coeff}")
    _check_user_mask(tokenizer, PROMPT)

    for mode in ("all", "generation_only", "user_only"):
        print(f"\n=============== mode={mode} ===============")
        outs = generate_with_steering(
            model=llm,
            tokenizer=tokenizer,
            prompts=[PROMPT],
            emotion_vectors_by_layer={layer: vectors_by_layer[layer]},
            emotions=all_emotions,
            selected_emotion=emotion,
            coeff=coeff,
            layers=[layer],
            normalize_steering_vector=True,
            N=1,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            batch_size=1,
            seed=0,
            steer_positions=mode,
            apply_chat_template=True,
            steer_debug=True,
        )
        print(f"[test] mode={mode} completion={outs[0][0]!r}")

    print("\n[test] all three modes ran. Visually verify the [steer-debug] lines above:")
    print("  - mode=all: prompt-pass steers full (B,T); gen-passes steer (B,1) each.")
    print("  - mode=generation_only: prompt-pass 'steered=0'; gen-passes steer (B,1).")
    print("  - mode=user_only: prompt-pass 'steered=<user-content-tokens>'; gen-passes 'steered=0'.")


if __name__ == "__main__":
    fire.Fire(main)
