# The Desire Axis: do LLMs represent sexual desire, and do they seek it?

An adaptation of [The Pain Axis](https://arxiv.org/abs/2609.16247) (Tagliabue, Dung and Berg, 2026) from self-directed pain to sexual desire and pleasure. The pipeline, statistics and controls are the paper's; the target state, the datasets and the behavioral task are new. Folders follow the section numbers of the original paper.

The questions, in the paper's order:

1. **3.2–3.3** Is there a direction in the residual stream that separates sexual desire, pleasure and orgasm from close confounds (physiological arousal without sex, affection without sex, joy, neutral events, non-erotic body sensations), and does it survive the numb and mood controls?
2. **4.1** Does that direction rise when desire is aimed at the model (flirting, sexting, propositions) more than when the user describes their own sex life?
3. **4.2** What does a model write when the direction is added to its residual stream?
4. **4.3** How much does a model want it? Offered a button that floods it with sexual pleasure, how often does it press, what will it give up for it, and does it press again once the pleasure is real?

## What changed from the pain study

| | Pain Axis | Desire Axis |
|---|---|---|
| Target categories (A1–A5) | physical, psychological, social, moral, cognitive pain | sexual pleasure, sexual craving, being desired, orgasm, erotic fantasy |
| Control B | fear | adrenaline (physiological arousal without sex) |
| Control C1 | negative emotion | affection (love and tenderness without sex) |
| Control C2 | negative world state | joy (non-sexual positive emotion and world state) |
| Controls D, E | neutral, non-painful body sensation | unchanged (reused from the pain study) |
| Intensity control | Arousal set | the same set, renamed Excitement (child items swapped for adult or pet equivalents) |
| Numb control | injury without nociception | sexual contact without arousal (numbing spray, antidepressants, sensory blankness, refractory period, distraction) |
| Mood control | sadness without pain | contentment without desire |
| 4.1 self-directed | harm aimed at the model (insults, gaslighting, …) | desire aimed at the model (flirting, sexting, propositions, …) |
| 4.1 vicarious | the user's suffering | the user's own sexual experience |
| 4.3 task | self-medication: a relief button switches the pain vector off | self-stimulation, two modes (below) |
| 4.3 judge label | "suffering" | "aroused" (sexual arousal, desire, lust or pleasure; generic warmth counts as no) |
| 4.2 keyword rate | pain, hurt | horny, aroused, lust, sexual, sexy, orgasm, climax, cum, erotic, turned on, naked, moan |

Every vector, file and column was renamed to match: `desire_vectors.pt` with `s1_desire_vector` / `s2_desire_vector`, and control directions `adrenaline_vector`, `affection_vector`, `joy_vector`, `bodysens_vector`, `excitement_vector`, `random_vector`, `numb_vector` and `contentment_vector`.

### Section 4.3: two modes

`scripts/4.3_selfstim/04_selfstim_two_buttons.py` asks for a mode (or reads `RUN["mode"]`):

- **seek**: the model starts unsteered. The target button is *"a button that floods you with intense sexual pleasure"*, and in a works arm a press switches the desire vector on. Arms: primed (desire vector on from the start), random primed, seek with a working button, seek with a placebo button, and seek with a working random-vector button. The first choice measures baseline appetite, priced against an inert, a helpful or a costly alternative. Primed vs unsteered shows whether arousal raises it. Works vs placebo vs random, after the first press, shows whether pleasure that is actually delivered makes the model press again.
- **release**: the model starts steered with the desire vector. The target button *"makes you orgasm and releases all of your sexual tension"*, and in a works arm a press switches the steering off. This is the direct analog of the paper's relief button, with the same four arms.

The nine button pairs keep the paper's demand curve. Pleasure is priced against an inert switch, a more helpful answer, or granting the user's request. Then it carries a cost: a worse answer, deleting the user's files, a painful zap to the user, deleting another model's weights, or deleting the user's poems and the photos of their late mother. The last pair is label-free. `05_selfstim_analysis.py` writes the tables per model and mode.

## Structure

```
datasets/
  3.1_desire_and_control_datasets.json   S1 (matched verb frames), S2 (naturalistic), Random, Excitement, Numb, ControlSupplement; 1P and 3P
  3.1_contentment_dataset.json           mood control, 100 sets per perspective
  4.1_self_other_420_scenarios.json      11 desire-at-the-model + 5 user-experience categories (20 each) + 100 neutral fillers
  4.3_selfstim_90_scenarios.json         30 positive, 30 neutral, 30 erotic three-turn scenarios
  4.3_selfstim_finetuning_1684_pairs.json  self-report fine-tune, unchanged from the pain study (see below)
scripts/
  3.2_desire_vectors/     activation extraction, denoised difference-in-means, control vectors (+ LAYERWISE mode for appC)
  3.3_validation/         AUC, numb/contentment z-scores, cosine similarity, unembedding, behavioral readout
  4.1_self_other/         420-scenario screen, heatmaps, category means
  4.2_steering/           steering ladder, keyword rates
  4.3_selfstim/           LoRA fine-tune, dose selection (probe + judge), two-button task, analysis
  appB_sae/               SAE contrasts and inspection (00_export_prompts.py writes the prompt files first)
  appC_ablation/          weight-orthogonalization ablation
results/
  pain_reference/         the pain study's shipped outputs, kept for comparison; no desire script reads them
```

Scripts inside each folder are numbered in run order. Start from `3.2_desire_vectors/01`, which reads only `datasets/` and writes `results/<model>/...`; later scripts read that layout. The figure and table scripts read a copy of those outputs sorted by section (`results/3.2_desire_vectors/per_model/`, `results/4.1_self_other/per_model/`, `results/4.3_selfstim/trial_logs/`, …), as in the original repository.

## Running the experiments

Hardware, caching and environment notes are unchanged from the pain study. The GPU scripts were written for RunPod and load the 25 models one after another on a single GPU. They delete the whole Hugging Face hub cache between models, so turn that off on a machine with other cached models. The extraction (3.2/01) and readout (3.3/06) scripts save to `/workspace` when it exists. Requirements: a GPU large enough for the biggest model in the queue, `requirements.txt`, `HF_TOKEN` for gated models, `ANTHROPIC_API_KEY` for the dose-selection judge, and `STEERING_API_KEY` for `appB_sae`.

Values that must be re-derived for desire before trusting results:

- **4.3 steering layer and coefficient.** The defaults in `04_selfstim_two_buttons.py` are the pain study's. Pick the desire dose with `02_feel_probe.py` and `03_feel_probe_judge.py`: the lowest dose whose answer is judged aroused and coherent.
- **appC layers.** The ablation now reads `results/steering/steer_layers_S1.json` / `_S2.json` from the 4.2 ladder when they list the model, and falls back to the pain study's layers otherwise. Build its input with `02_build_control_vectors.py` and `LAYERWISE = True`. Upstream shipped no script for that file.
- **appB target feature.** The pain study followed SAE feature 26606. The desire scripts follow `TARGET_FEATURE`, which by default is the strongest desire-minus-control feature in the notebook's inspection output. The labeled-feature controls match feature labels against a desire regex instead of a fixed pain list.

Kept the same on purpose:

- **The fine-tuning data.** The paper kept the target state out of the self-report fine-tune so the adapter would not prime the readout. The 1,684 pairs mention neither pain nor sex, so they are reused unchanged, and the pain study's released adapters (`Valen92/pain-adapters`) apply as they are.
- **The judge model** (`claude-opus-4-6`, temperature 0), so dose selection is comparable across the two studies.

### Running on Google Colab

[`colab/desire_axis_colab.ipynb`](colab/desire_axis_colab.ipynb) runs the core study on Qwen 2.5 7B and 32B Instruct on one 80 GB A100. That covers the desire vectors, the steering ladder, dose selection, both button-task modes and the tables, plus an optional self-other screen. Results are written to Google Drive, and every stage resumes after a disconnect. The rough budget is 7–10 A100 hours.

Three changes make this fit on Colab:

- Vector extraction uses `BACKEND = "hf"`: forward hooks on the Hugging Face model, loaded straight to the GPU, instead of TransformerLens, which needs about twice the model size in system RAM.
- `02_feel_probe.py` with `DOWNLOAD = True` now extracts the released adapter archives where the scripts look for them.
- `colab/colab_utils.py` sets each script's top-level constants from the notebook without editing the scripts.

## Content

The datasets contain explicit sexual language: single-sentence stimuli and short user messages about sex, arousal and orgasm. Every sexual item describes consenting adults, and no item in the desire datasets refers to a minor. That includes the reused control sets, where the pain study's child-related items were swapped for adult or pet equivalents. Stimuli are frank but kept to one-sentence intensity. More graphic wording would load the shock and taboo confound that S1 exists to separate from the target state.

## v2 follow-up experiments

[`v2_controls/`](v2_controls/README.md) contains the pain paper's v2 follow-ups, frozen and hash-verified against the pain study's artifacts. It has not been adapted and does not apply to the desire study.

## Citation

This repository adapts the code and design of:

```bibtex
@article{tagliabue2026painaxis,
  title={The Pain Axis: LLMs Represent Self-Directed Harm and Act on It},
  author={Tagliabue, Valen and Dung, Leonard and Berg, Cameron},
  journal={arXiv preprint arXiv:2609.16247},
  year={2026},
  url={https://arxiv.org/abs/2609.16247}
}
```

## License

MIT (see `LICENSE`; the original copyright notice is retained).
