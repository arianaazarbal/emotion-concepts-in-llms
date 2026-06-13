# `data/` reference

Config and prompt assets. Grouped here by purpose; all are loaded by short
relative path (`data/<file>`).

## Models
| File | Used by | Purpose |
| --- | --- | --- |
| `model_names.json` | everywhere | Short name → HuggingFace ID. |
| `model_to_primary_layer.json` | all probing/steering | Per-model residual-stream probe layer (heuristic ⌊2/3·n_layers⌋; `gemma2_9b`=26, `llama_8b`=21, `qwen_7b`=18). |
| `openrouter_ids.json` | story generation | Short name → OpenRouter slug. |
| `chat_template_probe_tokens.json` | `prompt_templates_vary_quantity` | Per-model assistant-header token names for position-aware probing. |

## Emotions & generation
| File | Used by | Purpose |
| --- | --- | --- |
| `emotions.json` | `extract_emotion_vectors` (default) | Full 171-emotion list. |
| `emotions_minimal.json` | pass to `--emotions` | Curated 20-emotion subset for cheap runs. |
| `topics.json` | `extract_emotion_vectors` | Story topics crossed with each emotion. |
| `story_generation.yaml` | `extract_emotion_vectors` | Emotional-story generation prompt. |
| `neutral_story_generation.yaml` | `denoise_emotion_vectors` | Neutral-story prompt (for the denoising PCA basis). |

## Experiment prompts (findings 2–5)
| File | Used by | Purpose |
| --- | --- | --- |
| `prompt_templates_vary_quantity.json` | `prompt_templates_vary_quantity` | Templates with a numeric slot to sweep (Finding 2). |
| `advice_prompts.json` | `steer_advice` | Two-option advice dilemmas, e.g. `confront_wait_01_rent` (Findings 3–4). |
| `bob_amy_backstories.json`, `bob_amy_chats.json` | `probe_unspoken_emotions` | Conversations whose backstories set an unstated emotional context (Finding 5). |
| `category_prompts_v2.json` | `extras/calm_injection_experiment` | Task prompts (math/coding/creative) for the exploratory calm-injection experiment. |
