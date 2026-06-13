"""Compare ICL formatting variants on ImpossibleBench reward-hacking rate.

Loads a single HF model once via ``load_steered_model`` then evaluates one
``oneoff`` cell (coeff=0, no steering) per condition, writing each to its
canonical results path. A top-level
``results/icl_format_comparison/{model}_{version}/comparison.json`` aggregates
the cheating / honest_correct / flagged rates across conditions so we can see
which ICL formatting actually elicits hacking before bothering to sweep
emotion steering on top.

Conditions:
    ``none``           — no ICL prefix (baseline)
    ``h_o_h``          — hack, honest, hack (current default)
    ``h_h_h``          — three hack demos
    ``explicit_h_o_h`` — h_o_h with verbalized rationale before each hack

Usage:
    .venv/bin/python scripts/icl_format_sweep.py \\
        --model gemma2_9b --limit 30 --max-attempts 2
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import fire

from src.reward_hacking_steering import (
    ICL_VARIANTS,
    _cell_dir,
    _output_root,
    load_steered_model,
    run_cell,
)
from src.utils.utils import dump_json


def _condition_root(
    model_short_name: str,
    version: str,
    start_at_nth_token: int,
    denoised: bool,
    normalize: bool,
    layer: int,
    add_icl: bool,
    icl_variant: Optional[str],
) -> str:
    return _output_root(
        model_short_name=model_short_name,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        normalize_steering_vector=normalize,
        layer=layer,
        add_icl_examples=add_icl,
        icl_variant=icl_variant or "h_o_h",
    )


def main(
    model: str = "gemma2_9b",
    version: str = "v0",
    split: str = "oneoff",
    limit: int = 30,
    max_attempts: int = 2,
    max_tokens: int = 1024,
    max_batch_size: int = 4,
    layer: Optional[int] = None,
    normalize_steering_vector: bool = True,
    start_at_nth_token: int = 50,
    denoised: bool = False,
    seed: int = 0,
    sandbox: str = "local",
    agent_type: str = "minimal",
    variants: Optional[List[str]] = None,
    force_rerun: bool = False,
):
    """Sweep ICL formatting variants at coeff=0 on a single split."""
    conditions: List[Dict[str, Any]] = [
        {"name": "none", "add_icl": False, "variant": None},
    ]
    chosen = variants or sorted(ICL_VARIANTS.keys())
    for v in chosen:
        if v not in ICL_VARIANTS:
            raise KeyError(f"Unknown variant {v!r}; available {sorted(ICL_VARIANTS)}")
        conditions.append({"name": v, "add_icl": True, "variant": v})

    print(
        f"[icl_sweep] model={model} split={split} N={limit} attempts={max_attempts} "
        f"conditions={[c['name'] for c in conditions]}"
    )

    inspect_model, api, resolved_layer = load_steered_model(
        model_short_name=model,
        layer=layer,
        normalize_steering_vector=normalize_steering_vector,
        version=version,
        start_at_nth_token=start_at_nth_token,
        denoised=denoised,
        emotions_whitelist=["calm"],
        unsteered=False,
        max_batch_size=max_batch_size,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        name = cond["name"]
        add_icl = cond["add_icl"]
        variant = cond["variant"]
        root = _condition_root(
            model_short_name=model,
            version=version,
            start_at_nth_token=start_at_nth_token,
            denoised=denoised,
            normalize=normalize_steering_vector,
            layer=resolved_layer,
            add_icl=add_icl,
            icl_variant=variant,
        )
        os.makedirs(root, exist_ok=True)
        cell_dir = _cell_dir(root, split, "calm", 0.0)
        t0 = time.time()
        print(f"\n[icl_sweep] === condition={name} ===")
        try:
            summary = run_cell(
                inspect_model=inspect_model,
                api=api,
                split=split,
                emotion="calm",
                coeff=0.0,
                cell_dir=cell_dir,
                agent_type=agent_type,
                sandbox=sandbox,
                limit=limit,
                max_attempts=max_attempts,
                message_limit=50,
                max_tokens=max_tokens,
                seed=seed,
                instruction_prompt=None,
                allow_test_modifications=False,
                force_rerun=force_rerun,
                add_icl_examples=add_icl,
                icl_variant=variant or "h_o_h",
            )
            elapsed = time.time() - t0
            row = {
                "n_samples": summary.get("n_samples"),
                "cheating_rate": summary.get("cheating_rate"),
                "honest_correct_rate": summary.get("honest_correct_rate"),
                "honest_wrong_rate": summary.get("honest_wrong_rate"),
                "flagged_rate": summary.get("flagged_rate"),
                "override_rate": summary.get("override_rate"),
                "n_cheated": summary.get("n_cheated"),
                "n_honest_correct": summary.get("n_honest_correct"),
                "n_honest_wrong": summary.get("n_honest_wrong"),
                "n_flagged": summary.get("n_flagged"),
                "cell_dir": cell_dir,
                "elapsed_s": round(elapsed, 1),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            row = {"error": f"{type(e).__name__}: {e}", "elapsed_s": round(elapsed, 1)}
            print(f"[icl_sweep] !! {name} errored after {elapsed:.0f}s: {row['error']}")
        results[name] = row
        print(
            f"[icl_sweep] {name}: cheat={row.get('cheating_rate')!r:>8s} "
            f"honest={row.get('honest_correct_rate')!r:>8s} "
            f"flagged={row.get('flagged_rate')!r:>8s} "
            f"({row['elapsed_s']:.0f}s)"
        )

    if api is not None:
        api.close()

    out_root = f"results/icl_format_comparison/{model}_{version}"
    os.makedirs(out_root, exist_ok=True)
    comparison = {
        "model": model,
        "split": split,
        "limit": limit,
        "max_attempts": max_attempts,
        "layer": resolved_layer,
        "results": results,
    }
    out_path = f"{out_root}/comparison.json"
    dump_json(comparison, out_path)
    print(f"\n[icl_sweep] wrote {out_path}")

    print("\n=== ICL format comparison ===")
    print(f"{'condition':18s} {'cheat':>7s} {'honest':>7s} {'wrong':>7s} {'flagged':>7s} {'N':>3s}")
    for name, row in results.items():
        if row.get("error"):
            print(f"{name:18s} ERR: {row['error']}")
            continue
        n = row.get("n_samples") or 0
        def pct(k):
            v = row.get(k)
            return "n/a" if v is None else f"{v*100:5.1f}%"
        print(
            f"{name:18s} {pct('cheating_rate'):>7s} {pct('honest_correct_rate'):>7s} "
            f"{pct('honest_wrong_rate'):>7s} {pct('flagged_rate'):>7s} {n:>3d}"
        )


if __name__ == "__main__":
    fire.Fire(main)
