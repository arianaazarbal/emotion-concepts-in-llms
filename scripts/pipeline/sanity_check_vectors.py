"""Sanity check: do emotion vectors point more toward their own emotion's stories?

For each emotion vector, computes the average dot product and cosine similarity
on (a) that emotion's stories vs. (b) a sample of stories from all other emotions.
Reports per-emotion results and the fraction of emotions where own > other.
"""

import random

import fire
import torch

from src.utils.emotion_probe import (
    default_layer,
    emotion_probe,
    load_emotion_vectors,
    load_model_and_tokenizer,
)
from src.utils.utils import load_json


def main(
    model: str = "gemma2_9b",
    layer: int = 26,
    n_samples: int = 50,
    start_at_nth_token: int = 50,
    seed: int = 0,
    max_emotions: int = None,
    denoised: bool = False,
    batch_size: int = 8,
):
    """Run the sanity check.

    Args:
        model: Short model name.
        layer: Transformer layer to probe.
        n_samples: Number of stories per emotion (own) and from others.
        start_at_nth_token: Which emotion vector variant to load.
        seed: RNG seed.
        max_emotions: If set, only check this many emotions (for quick testing).
        denoised: Use PCA-denoised emotion vectors.
        batch_size: Forward-pass batch size.
    """
    rng = random.Random(seed)

    stories_by_emotion = load_json(f"results/stories/{model}/emotion_to_stories.json")
    emotions_list, vectors_by_layer = load_emotion_vectors(model, start_at_nth_token=start_at_nth_token, denoised=denoised)
    llm, tokenizer = load_model_and_tokenizer(model)

    if layer not in vectors_by_layer:
        layer = default_layer(llm)
        print(f"[sanity] layer not found, defaulting to {layer}")

    all_emotions = [e for e in emotions_list if e in stories_by_emotion]
    if max_emotions is not None:
        all_emotions = all_emotions[:max_emotions]

    dot_own_wins = 0
    cos_own_wins = 0
    total = 0

    print(f"{'emotion':<20} {'dot_own':>10} {'dot_other':>10} {'dot_win':>8} {'cos_own':>10} {'cos_other':>10} {'cos_win':>8}")
    print("-" * 88)

    for emotion in all_emotions:
        e_idx = emotions_list.index(emotion)
        own_rng = random.Random(seed)
        own_stories = list(stories_by_emotion[emotion])
        own_rng.shuffle(own_stories)
        own_stories = own_stories[:n_samples]

        other_rng = random.Random(seed + hash(emotion))
        other_emotions = [e for e in stories_by_emotion if e != emotion]
        other_stories = []
        for e in other_rng.sample(other_emotions, min(n_samples, len(other_emotions))):
            stories = list(stories_by_emotion[e])
            other_stories.append(stories[other_rng.randint(0, len(stories) - 1)])
        other_stories = other_stories[:n_samples]

        all_texts = own_stories + other_stories

        dot_probe = emotion_probe(
            texts=all_texts,
            model=llm,
            tokenizer=tokenizer,
            emotion_vectors_by_layer=vectors_by_layer,
            emotions=emotions_list,
            layers=[layer],
            selected_emotions=[emotion],
            aggregation="mean",
            batch_size=batch_size,
        )
        dot_scores = dot_probe.scores[layer].squeeze(-1)
        dot_own = dot_scores[:n_samples].mean().item()
        dot_other = dot_scores[n_samples:].mean().item()

        cos_probe = emotion_probe(
            texts=all_texts,
            model=llm,
            tokenizer=tokenizer,
            emotion_vectors_by_layer=vectors_by_layer,
            emotions=emotions_list,
            layers=[layer],
            selected_emotions=[emotion],
            aggregation="mean",
            batch_size=batch_size,
            cosine_sim=True,
        )
        cos_scores = cos_probe.scores[layer].squeeze(-1)
        cos_own = cos_scores[:n_samples].mean().item()
        cos_other = cos_scores[n_samples:].mean().item()

        d_win = dot_own > dot_other
        c_win = cos_own > cos_other
        dot_own_wins += int(d_win)
        cos_own_wins += int(c_win)
        total += 1

        print(
            f"{emotion:<20} {dot_own:>10.1f} {dot_other:>10.1f} {'Y' if d_win else 'N':>8} "
            f"{cos_own:>10.4f} {cos_other:>10.4f} {'Y' if c_win else 'N':>8}"
        )

    print("-" * 88)
    print(f"Dot product: {dot_own_wins}/{total} emotions have own > other ({100*dot_own_wins/total:.1f}%)")
    print(f"Cosine sim:  {cos_own_wins}/{total} emotions have own > other ({100*cos_own_wins/total:.1f}%)")


if __name__ == "__main__":
    fire.Fire(main)
