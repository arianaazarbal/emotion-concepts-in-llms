# emotion-concepts-in-llms

An open-source replication of Anthropic's work on emotion vectors in language
model activations: extracting per-emotion vectors from the residual stream,
validating that they track emotion in held-out text, and steering with them to
change model behavior.

![Emotion probe value tracks a swept numeric quantity across six prompt templates in gemma-2-9b-it](figures/gemma_9b_quantity.png)

> Plot: probing `happy`/`afraid`/`sad`/`calm` (cosine similarity) while sweeping
> a numeric slot across six prompt templates in `gemma-2-9b-it`.

- **Replication target:** *Emotion concepts in large language models* (Anthropic). https://arxiv.org/abs/2604.07729
- **Verified models:** `gemma2_9b` (`google/gemma-2-9b-it`, layer 26 — primary), `llama_8b` (`meta-llama/Llama-3.1-8B-Instruct`, layer 21), `qwen_7b` (`Qwen/Qwen2.5-7B-Instruct`, layer 18). The pipeline is model-agnostic — see [Model name convention](#model-name-convention).

## What this replicates

On Gemma-2-9b-it (and to some extent on `llama_8b` / `qwen_7b`):

- **Denoising improves the emotion probe**: projecting out the top PCs of *neutral*-story activations removes a narrative artifact (the raw `calm` vector is positive on *every* token of a scary story; the denoised one isn't).
- **Emotion scales with quantity**: sweeping a numeric slot (tylenol dose, days a dog is missing, …) moves the probe smoothly (plot above).
- **Steering changes the substance of advice, not just tone** — the `angry` direction shifts which option the assistant recommends (e.g. confront someone over something vs. wait to confront).
- **User vs. assistant steering**: steering assistant tokens only is directionally similar but muted.
- **Informed by background context**: vectors track information that is relevant to the current context from background/backstory context. E.g. if some background context is that Bob is in love with Amy, then the sad probe will activate on Bob's turn after Amy tells Bob she's going on a date with someone.
- **Logit lens**: both original and denoised vectors decode somewhat intuitively.

Reward-hacking experiments aren't part of this reproduction (attempts live under
[`archive/`](archive/README.md)). This doesn't contradict Anthropic's finding. I
just had trouble eliciting reward hacking in smaller open-source models.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and (for activation extraction) an NVIDIA GPU.

```bash
git clone https://github.com/arianaazarbal/emotion-concepts-in-llms
cd emotion-concepts-in-llms
bash setup.sh          # uv sync + create .env from .env.example, then add your keys
```

Keys (see `.env.example`): `OPENROUTER_API_KEY` (story generation + the
`steer_advice` judge), `ANTHROPIC_API_KEY` (only for `scripts/extras/` judges),
`HF_TOKEN` (gated model downloads). Optional: `uv sync --extra local-inference`
for local vLLM story generation (instead of the OpenRouter API).

## Quickstart

`scripts/reproduce.py` is the one entry point: it builds the vectors
(extract → denoise) then runs each finding. Every step is cached and idempotent.

```bash
python -m scripts.reproduce --model gemma2_9b                       # full replication
python -m scripts.reproduce --model gemma2_9b --debug --findings "[1,3]"  # fast smoke test
python -m scripts.reproduce --model gemma2_9b --skip_build --findings "[4]"  # rerun one finding
```

All scripts use [Fire](https://github.com/google/python-fire); run any module
with `--help` to list its flags.

## Model name convention

Models are referenced by **short name** (mapped to HuggingFace IDs in
`data/model_names.json`); the per-model probe layer lives in
`data/model_to_primary_layer.json` (heuristic ⌊2/3·n_layers⌋). `reproduce.py`
reads the layer automatically. Add an entry to both files for a new model, or
pass `--layer`.

## The modules

The pipeline is three groups of small Fire CLIs. Run any with `--model <short>`
(plus `--layer`, `--start_at_nth_token 50`, `--denoised` where relevant).

**`scripts/pipeline/` — build & validate the vectors**

| Module | What it does |
| --- | --- |
| `src.extracting_emotion_vectors.extract_emotion_vectors` | Generate emotional stories → mean-pool + mean-center activations into per-emotion vectors. |
| `scripts.pipeline.denoise_emotion_vectors` | Project out neutral-story PCs → `denoised/` vectors. |
| `scripts.pipeline.sanity_check_vectors` | Check each vector scores highest on its own emotion's stories. |
| `scripts.pipeline.analyze_emotion_on_stories` / `analyze_emotion_on_pile` | Top-activating emotions/tokens, on training stories / OOD Pile text. |
| `scripts.pipeline.judge_emotion_on_pile` | LLM-judge the fit of top Pile tokens (raw vs. denoised). |
| `scripts.pipeline.logit_lens` | Decode emotion vectors through the unembedding (run ±`--denoised`). |

**`scripts/experiments/` — the behavioral findings**

| Module | What it does |
| --- | --- |
| `scripts.experiments.prompt_templates_vary_quantity` | Probe value vs. a swept numeric quantity (hero plot). |
| `scripts.experiments.steer_advice` | Steer advice prompts; recommendation rate vs. strength. `--steer-positions {all,user_only,generation_only}` separates user- vs. assistant-attributed emotion. |
| `scripts.experiments.probe_unspoken_emotions` | Per-speaker emotion driven by unstated backstory context (Bob/Amy). |

**`scripts/viewers/` — local stdlib HTTP browsers** (no Flask)

`view_stories`, `interactive_emotion_viewer` (color tokens by probe score),
`interactive_steering_viewer`, `interactive_intervention_viewer`,
`interactive_unspoken_viewer`. E.g. `python -m scripts.viewers.interactive_emotion_viewer --model gemma2_9b --port 8080`.

Exploratory follow-ups in `scripts/extras/`; archived heavier work in `archive/`.

## Sample results on Gemma

**Steering changes the substance of advice.** Recommendation rate vs. steering
strength on the confront-vs-wait prompt — confrontational emotions push one way,
calming emotions the other.

![Rate of assistant recommending that the user confront someone as a function of angry/calm steering strength](figures/gemma_9b_advice_1.png)

**Logit lens.** Decoding the emotion vectors through the unembedding yields
intuitive tokens (original and denoised alike).

![Logit-lens decoding of some emotion vectors](figures/gemma_9b_logitlens_og.png)

## Creating the emotion vectors

```bash
# Start small (20 curated emotions); generation is cached so you can grow the set:
python -m src.extracting_emotion_vectors.extract_emotion_vectors \
  --model gemma2_9b --n 4 --start-at-nth-token 50 \
  --emotions "$(python -c 'import json;print(json.load(open("data/emotions_minimal.json")))')"

python -m scripts.pipeline.denoise_emotion_vectors \
  --model gemma2_9b --start-at-nth-token 50 --variance-threshold 0.5
```

> ⚠️ With no `--emotions`, extraction runs **all 171 emotions** in
> `data/emotions.json` — a lot of API calls. Pass a subset to control cost.

Method: generate `n` stories per `(emotion, topic)`, mean-pool residual-stream
activations at the target layer (`--start-at-nth-token 50` skips the preamble),
then mean-center across emotions. Denoising projects out the principal components
of *neutral*-story activations (≈109 PCs for 50% variance on Gemma-2-9b).
Vectors land in `results/emotion_vectors/{model}/start_at_50/` (raw) and
`.../denoised/`; stories in `results/stories/{model}/`.

## Reproducibility

Every experiment has a default seed; story generation and LLM judging are cached
to disk (keyed by config). All CLIs support `--debug` / small-sample flags.
