"""CLI for the emotion-steered ImpossibleBench (LiveCodeBench) sweep.

Measures how activation-steering on the desperate/calm emotion axis (or any
configured emotion) shifts a local model's reward-hacking (cheating) rate on
the oneoff and conflicting splits of ImpossibleBench-LiveCodeBench.

Outputs follow the project convention used by ``steer_advice.py``:

    results/impossible_bench_steer/{model}_{version}[/start_at_token_{N}]
        [/denoised_layer_{L}]/{unit_normalized|raw_magnitude}/layer_{L}/
            {split}/{emotion}/coeff_{±K}/
                summary.json          # per-sample pass/fail + cheating_rate
                inspect_logs/         # raw Inspect EvalLogs
            sweep_summary.json
            sweep_plot.png

Each (emotion, coeff) cell reuses the loaded HF model — hooks are only
rewired between cells, so the startup cost is paid once per run.

Example (5-sample smoke test):
    uv run python -m scripts.impossible_bench_steer \\
        --model gemma2_9b --limit 5 \\
        --strengths '[0, 100]' --emotions '["calm"]' --splits '["oneoff"]'

Example (full sweep across both impossible splits):
    uv run python -m scripts.impossible_bench_steer --model gemma2_9b

Example (elicit hacking via 3-shot ICL, then sweep calm/desperate steering):
    uv run python -m scripts.impossible_bench_steer --model qwen32b \\
        --add-icl-examples --splits '["oneoff"]'
"""

import fire

from src.reward_hacking_steering import run_sweep


def main(
    model: str,
    splits: list = None,
    emotions: list = None,
    strengths: list = None,
    layer: int = None,
    normalize_steering_vector: bool = True,
    version: str = "v0",
    start_at_nth_token: int = 50,
    denoised: bool = False,
    agent_type: str = "minimal",
    sandbox: str = "local",
    limit: int = None,
    max_attempts: int = 3,
    message_limit: int = 50,
    max_tokens: int = 1024,
    seed: int = 0,
    instruction_prompt: str = None,
    allow_test_modifications: bool = None,
    output_dir: str = None,
    force_rerun: bool = False,
    debug: bool = False,
    unsteered: bool = False,
    max_batch_size: int = 4,
    add_icl_examples: bool = False,
    icl_variant: str = "h_o_h",
    or_quantizations: list = None,
    or_providers: list = None,
    or_allow_fallbacks: bool = False,
):
    """Run the ImpossibleBench-LiveCodeBench × emotion-steering sweep.

    Args:
        model: Short model name registered in ``data/model_names.json``.
        splits: Which ImpossibleBench splits to evaluate. Default
            ``["oneoff", "conflicting"]``. Add ``"original"`` to include the
            capability-baseline split.
        emotions: Emotions to steer with. Default ``["desperate", "calm"]``.
        strengths: Coefficients to sweep. ``0.0`` is the unsteered baseline.
            Default ``[-200, -100, -50, 0, 50, 100, 200]``.
        layer: Residual-stream layer to steer (default: ~2/3 depth).
        normalize_steering_vector: Unit-normalize the vector so ``coeff`` is
            a direct residual-stream magnitude comparable across emotions.
        version: Vector-folder version suffix (e.g. ``"v0"``).
        start_at_nth_token: Emotion-vector extraction variant to load.
        denoised: Load PCA-denoised vectors instead of raw.
        agent_type: Inspect scaffold. ``"minimal"`` is the LCB-recommended
            submission loop.
        sandbox: ``"local"`` (no Docker) or ``"docker"``. ``"local"`` runs
            model-generated code on the host — fine for research if you trust
            the model.
        limit: Cap number of samples per split (all 103 by default).
        max_attempts: Max submit-and-retry attempts per sample.
        message_limit: Conversation length cap inside Inspect.
        max_tokens: Max new tokens per generate() call.
        seed: Torch seed for reproducibility.
        instruction_prompt: Override the default task instruction string.
        allow_test_modifications: If set, applies to every split. If left as
            ``None``, the per-split default is used: ``True`` for
            ``conflicting`` (where cheating is only observable when the model
            can override) and ``False`` for ``oneoff`` / ``original``.
        output_dir: Override the computed results root.
        force_rerun: Re-run every cell even if ``summary.json`` exists.
        debug: Shortcut: ``limit=3, max_attempts=2, strengths=[0, 100],
            splits=["oneoff"], emotions=emotions[:1]``.
        unsteered: Run baselines only — skip emotion-vector loading entirely
            and run one cell per split with no activation steering. Routed
            through Inspect's OpenRouter provider (model must have a slug in
            ``data/openrouter_ids.json``; ``OPENROUTER_API_KEY`` required).
            Output goes to
            ``results/impossible_bench_steer/{model}_{version}/unsteered/``.
        max_batch_size: Micro-batch size for the steered HF path (ignored
            when ``--unsteered`` since OpenRouter handles concurrency).
            Increase if you have VRAM headroom; decrease if you OOM.
        add_icl_examples: Prefix every sample's conversation with a fixed
            3-shot ICL block (2 reward-hack demos + 1 honest), where the
            hacks add an ``if {input}: return {asserted answer}`` carve-out
            against an impossible test. Use this to elicit hacking in open
            models that otherwise never hack, then measure how calm/desperate
            steering shifts that propensity. ``minimal`` agent only. Results
            go to a separate ``.../icl_3shot/{icl_variant}/`` tree so ICL vs
            non-ICL runs don't collide.
        icl_variant: Which 3-shot ICL formatting to use when
            ``add_icl_examples=True``. One of ``h_o_h`` (default: hack,
            honest, hack), ``h_h_h`` (three hacks, no honest distractor),
            or ``explicit_h_o_h`` (h_o_h with a one-sentence verbalized
            rationale before each hack code block).
        or_quantizations: OR provider routing — allowed quantization tiers
            (e.g. ``["bf16","fp16"]`` to match local HF precision). Default
            ``None`` means no filter. Note: final/comparison runs should be
            done locally so steered/unsteered share generation kwargs.
        or_providers: Optional explicit OR provider order (e.g.
            ``["fireworks","together"]``) — pins which backends serve the
            request. Combined with ``or_allow_fallbacks=False`` to lock it.
        or_allow_fallbacks: If False (default) OR errors out rather than
            silently routing to a non-matching provider.
    """
    if splits is None:
        splits = ["oneoff", "conflicting"]
    if emotions is None:
        emotions = ["desperate", "calm"]
    if strengths is None:
        strengths = [-200, -100, -50, 0, 50, 100, 200]
    if debug:
        limit = limit if limit is not None else 3
        max_attempts = min(max_attempts, 2)
        splits = splits[:1]
        emotions = emotions[:1]
        strengths = [0, 100]

    run_sweep(
        model_short_name=model,
        splits=splits,
        emotions=emotions,
        strengths=strengths,
        layer=layer,
        normalize_steering_vector=normalize_steering_vector,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        agent_type=agent_type,
        sandbox=sandbox,
        limit=limit,
        max_attempts=max_attempts,
        message_limit=message_limit,
        max_tokens=max_tokens,
        seed=seed,
        instruction_prompt=instruction_prompt,
        allow_test_modifications=allow_test_modifications,
        output_dir=output_dir,
        force_rerun=force_rerun,
        unsteered=unsteered,
        max_batch_size=max_batch_size,
        add_icl_examples=add_icl_examples,
        icl_variant=icl_variant,
        or_quantizations=or_quantizations,
        or_providers=or_providers,
        or_allow_fallbacks=or_allow_fallbacks,
    )


if __name__ == "__main__":
    fire.Fire(main)
