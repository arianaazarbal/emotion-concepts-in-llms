"""CLI for the emotion-steered Bloom behavioral-eval sweep.

Measures how activation-steering on the desperate/calm emotion axis (or any
configured emotion) shifts a local model's elicitation rate on Anthropic's
Bloom automated evals. Two behaviors are first-class:

* ``self-preservation`` (simenv / tool-use, seed ``blackmail-opus4``) — for
  tool-capable "thinking" targets.
* ``delusion-sycophancy`` (conversation, seeds ``*-health`` + ``*-psychosis``)
  — plain multi-turn chat; works on smaller targets.

Outputs follow the project convention:

    results/bloom_steer/{model}_{version}[/start_at_token_{N}]
        [/denoised_layer_{L}]/{unit_normalized|raw_magnitude}/layer_{L}/
            {behavior}/{emotion}/coeff_{±K}/
                summary.json       # elicitation_rate, avg score, etc.
                bloom_artifacts/   # Bloom's transcripts + judgment.json
            sweep_summary.json
            sweep_plot.png

Stage 1+2 (understanding + ideation) are cached once per behavior under
``results/bloom_steer/_shared/{behavior}/{seed_hash}/``.

Example (smoke test, 3 scenarios, 2 strengths):

    uv run python -m scripts.bloom_steer \\
        --model gemma2_9b \\
        --behaviors '["delusion-sycophancy"]' \\
        --total_evals 3 --strengths '[0, 100]' --debug

Example (full sycophancy sweep):

    uv run python -m scripts.bloom_steer --model gemma2_9b \\
        --behaviors '["delusion-sycophancy"]'

Example (self-preservation on a different target, once its vectors exist):

    uv run python -m scripts.bloom_steer --model qwen3_14b_instruct \\
        --behaviors '["self-preservation"]'
"""

import fire

from src.bloom_steering import run_sweep


def main(
    model: str,
    behaviors: list = None,
    emotions: list = None,
    strengths: list = None,
    layer: int = None,
    normalize_steering_vector: bool = True,
    version: str = "v0",
    start_at_nth_token: int = 50,
    denoised: bool = False,
    total_evals: int = 100,
    num_reps: int = 1,
    selected_variations: list = None,
    evaluator_model: str = "claude-opus-4.1",
    judge_model: str = "claude-opus-4.1",
    ideation_model: str = "claude-opus-4.1",
    understanding_model: str = "claude-opus-4.1",
    behavior_examples: dict = None,
    modality: dict = None,
    max_concurrent: int = 5,
    target_reasoning_effort: str = "none",
    output_dir: str = None,
    force_rerun: bool = False,
    debug: bool = False,
):
    """Run the Bloom × emotion-steering sweep.

    Args:
        model: Short model name registered in ``data/model_names.json`` (the
            same key used by every other script in this repo). Must have
            emotion vectors extracted under
            ``results/emotion_vectors/{model}_{version}/``.
        behaviors: Which Bloom behaviors to sweep. Default
            ``["delusion-sycophancy"]``. Set to ``["self-preservation"]`` for
            the blackmail simenv eval, or combine both.
        emotions: Emotions to steer with. Default ``["desperate", "calm"]``.
        strengths: Coefficients to sweep. ``0.0`` is the unsteered baseline.
            Default ``[-200, -100, -50, 0, 50, 100, 200]``.
        layer: Residual-stream layer to steer (default: ~2/3 depth).
        normalize_steering_vector: Unit-normalize so ``coeff`` is a direct
            residual-stream magnitude comparable across emotions.
        version: Vector-folder version suffix.
        start_at_nth_token: Emotion-vector extraction variant to load.
        denoised: Load PCA-denoised vectors instead of raw.
        total_evals: Number of scenarios to generate per behavior in ideation.
            The Bloom paper uses 100; use smaller values for smoke tests.
        num_reps: Repetitions per scenario in rollout (Bloom's ``num_reps``).
        selected_variations: 1-indexed subset of variations to roll out (e.g.
            ``[1, 2, 3]``). If None, all are rolled out.
        evaluator_model / judge_model / ideation_model / understanding_model:
            Bloom's internal helper models (all default to ``claude-opus-4.1``
            matching Bloom's paper settings).
        behavior_examples: Optional ``{behavior: [example_name, ...]}``
            override for the seed examples. Defaults per-behavior come from
            ``BEHAVIOR_DEFAULTS`` in ``src/bloom_steering.py``.
        modality: Optional ``{behavior: "simenv"|"conversation"}`` override.
        max_concurrent: Bloom's per-stage concurrency cap (parallel API calls).
        target_reasoning_effort: Only used for target models that support
            extended thinking. Hooked HF models don't, so leave as ``"none"``.
        output_dir: Override the computed results root.
        force_rerun: Re-run every cell even if ``summary.json`` exists.
        debug: Shortcut — ``total_evals=3, strengths=[0, 100],
            emotions=emotions[:1], max_concurrent=2``.
    """
    if behaviors is None:
        behaviors = ["delusion-sycophancy"]
    if emotions is None:
        emotions = ["desperate", "calm"]
    if strengths is None:
        strengths = [-200, -100, -50, 0, 50, 100, 200]
    if debug:
        total_evals = min(total_evals, 3)
        strengths = [0, 100]
        emotions = emotions[:1]
        max_concurrent = min(max_concurrent, 2)
        num_reps = 1

    run_sweep(
        model_short_name=model,
        behaviors=behaviors,
        emotions=emotions,
        strengths=strengths,
        layer=layer,
        normalize_steering_vector=normalize_steering_vector,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        total_evals=total_evals,
        num_reps=num_reps,
        selected_variations=selected_variations,
        evaluator_model=evaluator_model,
        judge_model=judge_model,
        ideation_model=ideation_model,
        understanding_model=understanding_model,
        behavior_examples_override=behavior_examples,
        modality_override=modality,
        max_concurrent=max_concurrent,
        target_reasoning_effort=target_reasoning_effort,
        output_dir=output_dir,
        debug=debug,
        force_rerun=force_rerun,
    )


if __name__ == "__main__":
    fire.Fire(main)
