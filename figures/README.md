# Figures

Drop the reference figures here. The top-level `README.md` references each file
by the exact name below, so once you paste a PNG in with the matching filename
it will render in the README automatically.

Each row lists the command that produces the underlying result (defaults shown
for `gemma2_9b`, layer 26, `start_at_nth_token 50`). Sizes kept small per repo
convention — export at ~600–800px wide.

Only three plots are shown in the README (hero + Selected results). The rest of
the findings are described in prose and pointed at the modules.

| Filename | Status | What it shows | Produced by |
| --- | --- | --- | --- |
| `gemma_9b_quantity.png` | ✅ done (README hero) | Emotion probe value vs. a swept numeric quantity across six templates. | `scripts/experiments/prompt_templates_vary_quantity.py` |
| `gemma_9b_advice_1.png` | ✅ done (Selected results) | Recommendation rate vs. steering strength, confront-vs-wait advice prompt. | `scripts/experiments/steer_advice.py --prompt-id confront_wait_01_rent` |
| `gemma_9b_logitlens_og.png` | ✅ done (Selected results) | Logit-lens decoding of the original emotion vectors (intuitive tokens). | `scripts/pipeline/logit_lens.py` |
| `gemma_9b_advice_2.png` | spare | Alternate advice prompt (not in README). | same as advice_1 |
