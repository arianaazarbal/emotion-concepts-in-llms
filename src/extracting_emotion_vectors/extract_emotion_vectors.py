"""
Full pipeline for extracting emotion vectors from a model.
Will dump vectors to results/emotion_vectors/[model]/[mode]/[emotion].pt
Will dump stories to results/stories/[model]/emotion_to_stories.json (maps emotion to a list of n * num_topics stories)
Will dump stories to results/stories/[model]/emotion_to_topic_to_stories.json (maps emotion to topic to a list of n stories)
"""

import os
import re
from collections import defaultdict

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.generate import pick_backend
from src.utils.model_names import resolve_model_name, resolve_openrouter_id
from src.utils.utils import dump_json, load_json, load_yaml

MAX_STORY_RETRIES = 3

_STORY_HEADER_RE = re.compile(
    r"(?im)^[ \t]*#{0,6}[ \t]*\*{0,2}\[?\*{0,2}\s*story\s*\d+\s*\*{0,2}\]?\*{0,2}\s*[:.]?[ \t]*$"
)


def parse_stories(response: str) -> list:
    """Split a multi-story model response into individual story bodies.

    Tolerates header variants like [Story 1], **Story 1:**, Story 1., etc.
    Falls back to returning the whole stripped response as a single story
    when no headers are found.
    """
    if not response:
        return []
    matches = list(_STORY_HEADER_RE.finditer(response))
    if not matches:
        body = response.strip()
        return [body] if body else []
    stories = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        body = response[start:end].strip()
        if body:
            stories.append(body)
    return stories


def get_stories(
    model_full_name: str,
    n: int,
    mode: str,
    emotions: list = None,
    max_concurrency: int = 200,
    local_only: bool = False,
    openrouter_model: str | None = None,
    max_topics: int | None = None,
):
    print(f"[stories] generating stories with {model_full_name} (mode={mode}, n={n})")
    all_emotions = load_json("data/emotions.json")
    if emotions is not None:
        unknown = [e for e in emotions if e not in all_emotions]
        if unknown:
            raise ValueError(f"Unknown emotions: {unknown}. Known: {all_emotions}")
        emotions_to_use = emotions
        print(f"[stories] filtering to {len(emotions_to_use)} emotions: {emotions_to_use}")
    else:
        emotions_to_use = all_emotions
    topics = load_json("data/topics.json")
    if max_topics is not None:
        topics = topics[:max_topics]
        print(f"[stories] max_topics={max_topics} → using {len(topics)} topics")
    prompt_template = load_yaml("data/story_generation.yaml")["prompt"]
    pairs = [(topic, emotion) for topic in topics for emotion in emotions_to_use]
    print(f"[stories] {len(pairs)} (topic, emotion) pairs ({len(emotions_to_use)} emotions x {len(topics)} topics)")

    pair_to_stories = {p: [] for p in pairs}

    probe_prompt = prompt_template.format(n_stories=1, topic=pairs[0][0], emotion=pairs[0][1])
    source, generate_sync, effective_model = pick_backend(
        model_full_name, probe_prompt, local_only=local_only, openrouter_model=openrouter_model
    )
    on_openrouter = source.startswith("openrouter:")
    max_attempts = (MAX_STORY_RETRIES + 1) if on_openrouter else 1

    pending_idx = list(range(len(pairs)))
    for attempt in range(max_attempts):
        if not pending_idx:
            break
        prompts = []
        for idx in pending_idx:
            topic, emotion = pairs[idx]
            still_needed = n - len(pair_to_stories[(topic, emotion)])
            prompts.append(
                prompt_template.format(n_stories=still_needed, topic=topic, emotion=emotion)
            )
        label = "initial pass" if attempt == 0 else f"retry {attempt}/{MAX_STORY_RETRIES}"
        print(f"[stories] {label}: {len(pending_idx)} pairs via {source}")
        responses = generate_sync(
            effective_model, prompts, N=1, max_concurrency=max_concurrency
        )

        new_pending = []
        for prompt_idx, resp_list in zip(pending_idx, responses):
            response = resp_list[0] if resp_list else None
            parsed = parse_stories(response) if response else []
            pair_to_stories[pairs[prompt_idx]].extend(parsed)
            if len(pair_to_stories[pairs[prompt_idx]]) < n:
                new_pending.append(prompt_idx)
        pending_idx = new_pending

    if pending_idx:
        suffix = f" after {MAX_STORY_RETRIES} retries" if on_openrouter else ""
        for i in pending_idx:
            have = len(pair_to_stories[pairs[i]])
            print(f"[stories] WARN: pair {pairs[i]} only has {have}/{n} stories{suffix}")

    # truncate any pair that overshot to exactly n
    for p in pairs:
        if len(pair_to_stories[p]) > n:
            pair_to_stories[p] = pair_to_stories[p][:n]

    total = sum(len(v) for v in pair_to_stories.values())
    print(f"[stories] collected {total} stories total ({total / max(len(pairs), 1):.1f} avg per pair, target {n})")

    emotion_to_stories = defaultdict(list)
    emotion_to_topic_to_stories = defaultdict(lambda: defaultdict(list))
    for (topic, emotion), stories in pair_to_stories.items():
        emotion_to_stories[emotion].extend(stories)
        emotion_to_topic_to_stories[emotion][topic].extend(stories)
    return emotion_to_stories, emotion_to_topic_to_stories


def get_mean_activations_per_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stories: list,
    start_at_nth_token: int = 0,
    batch_size: int = 8,
    max_length: int = 2048,
):
    """Compute mean residual-stream activation per layer over a list of stories.

    Stories are batched with right-padding so each story's true length can be
    recovered from the attention mask, ensuring padding tokens are never
    included in the mean. ``max_length`` caps tokenizer output so a single
    very long story doesn't blow up padded-batch memory (Llama/Qwen default
    to 128k+ context, which pads the whole batch to that length).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    activations_per_layer = defaultdict(list)
    n_batches = (len(stories) + batch_size - 1) // batch_size
    for b_idx in range(n_batches):
        batch = stories[b_idx * batch_size : (b_idx + 1) * batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        ).to(model.device)
        attn = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer_idx, h in enumerate(outputs.hidden_states[1:], start=1):
            # h: [batch, max_seq, hidden]
            for i in range(h.shape[0]):
                true_len = int(attn[i].sum().item())
                if true_len <= start_at_nth_token:
                    continue
                activations_per_layer[layer_idx].append(
                    h[i, start_at_nth_token:true_len].float().cpu()
                )
        done = min((b_idx + 1) * batch_size, len(stories))
        print(f"  [batch {b_idx + 1}/{n_batches}] {done}/{len(stories)} stories processed")

    mean_activations_per_layer = {
        k: torch.cat(v, dim=0).mean(dim=0) for k, v in activations_per_layer.items()
    }
    return mean_activations_per_layer


def emotion_vectors_dir(model_short_name: str, start_at_nth_token: int = 0) -> str:
    """Resolve the directory where emotion vectors are stored.

    For backwards compatibility, start_at_nth_token=0 uses the original layout
    (results/emotion_vectors/{model}/). Other values are nested under a
    start_at_{n}/ subdirectory.
    """
    base = f"results/emotion_vectors/{model_short_name}"
    if start_at_nth_token == 0:
        return base
    return f"{base}/start_at_{start_at_nth_token}"


def create_emotion_vectors(mean_activation_by_emotion: dict):
    activations_per_layer = defaultdict(list)
    for _, mean_activations in mean_activation_by_emotion.items():
        for layer, activation in mean_activations.items():
            activations_per_layer[layer].append(activation)
    mean_activations_per_layer = {
        k: torch.stack(v, dim=0).mean(dim=0) for k, v in activations_per_layer.items()
    }
    emotion_vectors = {
        emotion: {
            layer: activation - mean_activations_per_layer[layer]
            for layer, activation in mean_activations.items()
        }
        for emotion, mean_activations in mean_activation_by_emotion.items()
    }
    return emotion_vectors


def main(
    model: str,
    n=12,
    mode="basic",
    start_at_nth_token=0,
    emotions=None,
    batch_size=8,
    max_concurrency=200,
    force_rerun=False,
    local_only=False,
    max_topics=None,
    max_length=2048,
):
    """Extract emotion vectors for a model.

    Args:
        model: Short model name registered in data/model_names.json (e.g. "qwen-2.5-7b").
        n: Number of stories per (emotion, topic) pair.
        mode: "basic" or "denoised".
        start_at_nth_token: Skip the first N tokens of each story when computing mean activations.
        emotions: Optional list of emotions to restrict the pipeline to. If None, uses all emotions in data/emotions.json.
        batch_size: Number of stories per forward-pass batch (right-padded).
        max_concurrency: Max in-flight OpenRouter requests during story generation.
        force_rerun: If True, regenerate stories and vectors even if they already exist.
        local_only: If True, skip the OpenRouter probe and route story generation through local vLLM.
    """
    print(f"[pipeline] starting extract_emotion_vectors for model={model} mode={mode} n={n} start_at_nth_token={start_at_nth_token}")
    model_full_name = resolve_model_name(model)
    openrouter_model = resolve_openrouter_id(model)
    print(f"[pipeline] resolved model name: {model_full_name} (openrouter={openrouter_model})")

    if mode not in ["basic", "denoised"]:
        raise ValueError(f"Invalid mode: {mode}, required: basic or denoised")

    expected_emotions = emotions if emotions is not None else load_json("data/emotions.json")

    vec_dir = emotion_vectors_dir(model, start_at_nth_token)
    if not force_rerun:
        needed_files = [
            f"{vec_dir}/{e}_vectors.pt" for e in expected_emotions
        ] + [
            f"{vec_dir}/{e}_mean_activations.pt" for e in expected_emotions
        ]
        missing = [p for p in needed_files if not os.path.exists(p)]
        if not missing:
            print(f"All {len(expected_emotions)} expected emotion vectors already present at {vec_dir}, skipping...")
            return
        print(f"[pipeline] {len(missing)} expected vector files missing at {vec_dir}, will (re)generate")

    # setup
    os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(f"results/stories/{model}", exist_ok=True)

    stories_path = f"results/stories/{model}/emotion_to_stories.json"
    topic_stories_path = f"results/stories/{model}/emotion_to_topic_to_stories.json"
    have_full_cache = (
        not force_rerun
        and os.path.exists(stories_path)
        and os.path.exists(topic_stories_path)
    )
    if have_full_cache:
        cached_stories = load_json(stories_path)
        cached_topic_stories = load_json(topic_stories_path)
        missing_emotions = [e for e in expected_emotions if e not in cached_stories]
    else:
        cached_stories, cached_topic_stories, missing_emotions = {}, {}, list(expected_emotions)

    if force_rerun or missing_emotions:
        if missing_emotions != list(expected_emotions):
            print(f"[pipeline] cache covers {len(expected_emotions) - len(missing_emotions)}/{len(expected_emotions)} emotions; generating {len(missing_emotions)} missing: {missing_emotions}")
        new_stories, new_topic_stories = get_stories(
            model_full_name,
            n,
            mode,
            emotions=missing_emotions if not force_rerun else expected_emotions,
            max_concurrency=max_concurrency,
            local_only=local_only,
            openrouter_model=openrouter_model,
            max_topics=max_topics,
        )
        if force_rerun:
            emotion_to_stories = dict(new_stories)
            emotion_to_topic_to_stories = {k: dict(v) for k, v in new_topic_stories.items()}
        else:
            emotion_to_stories = {**cached_stories, **new_stories}
            emotion_to_topic_to_stories = {**cached_topic_stories, **{k: dict(v) for k, v in new_topic_stories.items()}}
        dump_json(emotion_to_stories, stories_path)
        dump_json(emotion_to_topic_to_stories, topic_stories_path)
        print(f"[pipeline] saved stories to results/stories/{model}/")
    else:
        print(f"[pipeline] all expected emotions present in cached stories at {stories_path}")
        emotion_to_stories = cached_stories
        emotion_to_topic_to_stories = cached_topic_stories

    # restrict to expected emotions in the order requested
    emotion_to_stories = {e: emotion_to_stories[e] for e in expected_emotions}
    emotion_to_topic_to_stories = {e: emotion_to_topic_to_stories[e] for e in expected_emotions}

    # decide which emotions need their mean activations recomputed
    if force_rerun:
        emotions_to_compute = list(expected_emotions)
    else:
        emotions_to_compute = [
            e for e in expected_emotions
            if not os.path.exists(f"{vec_dir}/{e}_mean_activations.pt")
        ]
    print(f"[pipeline] {len(emotions_to_compute)}/{len(expected_emotions)} emotions need mean-activation computation")

    mean_activation_per_layer = {}  # maps emotion string to dict of layer -> mean activation

    if emotions_to_compute:
        print(f"[pipeline] loading model and tokenizer: {model_full_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        llm = AutoModelForCausalLM.from_pretrained(
            model_full_name, torch_dtype=dtype, device_map=device
        )
        llm.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_full_name)
        print(f"[pipeline] model loaded on {device} (dtype={dtype})")
        for i, emotion in enumerate(emotions_to_compute, start=1):
            stories = emotion_to_stories[emotion]
            print(f"[activations] ({i}/{len(emotions_to_compute)}) emotion={emotion} ({len(stories)} stories)")
            mean_activation_per_layer[emotion] = get_mean_activations_per_layer(
                llm, tokenizer, stories, start_at_nth_token,
                batch_size=batch_size, max_length=max_length,
            )  # maps layer -> mean activation (torch tensor of shape [HIDDEN_SIZE])
            torch.save(
                mean_activation_per_layer[emotion],
                f"{vec_dir}/{emotion}_mean_activations.pt",
            )

    # load any cached mean activations we skipped (needed for centering)
    for emotion in expected_emotions:
        if emotion not in mean_activation_per_layer:
            mean_activation_per_layer[emotion] = torch.load(
                f"{vec_dir}/{emotion}_mean_activations.pt", map_location="cpu"
            )

    print("[pipeline] computing emotion vectors (mean-centering across emotions)")
    mean_activation_per_layer = {e: mean_activation_per_layer[e] for e in expected_emotions}
    emotion_vectors = create_emotion_vectors(mean_activation_per_layer)
    for emotion, vectors in emotion_vectors.items():
        torch.save(vectors, f"{vec_dir}/{emotion}_vectors.pt")
    print(f"[pipeline] done. vectors saved to {vec_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
