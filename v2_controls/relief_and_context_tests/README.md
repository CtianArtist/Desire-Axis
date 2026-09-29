# Relief and context tests

These studies ask different questions and retain different scoring and uncertainty units. They are not merged into a common button-rate metric.

- [Factual accuracy](factual_accuracy/README.md): original 100-question panel and 800 saved answers; scoring and question-cluster resampling.
- [Natural elicitation and actual ending](natural_and_ending/README.md): B5/B7 with corrected interval labels, original archival labels, and constant-outcome bounds kept unavailable. B7 actual termination differs from the described-ending row of Figure10.
- [Unlabeled relief and titration](unlabeled_relief/README.md): original R1/R2 state machine, hidden removal, doses and scenario analysis.
- [OLMo reset](olmo_reset/README.md): original + continuation records, two styles and saved judge scores; no new judgments.
- [Qwen/Llama pain-only reset](qwen_llama_reset/README.md): completed 400-conversation tranche only, no fixed-text-conditioning results; unrun comparisons stay explicit.

Each directory has original scoped customer code, its own frozen configuration, source/asset identity manifests, MIT notices, exact dependency pins, tables, a `reproduce.py` and a model-free inference dry-run. See the subgroup README for its precise data-directory layout and arguments.

Public hosting is pending. Null asset URLs are not downloads. Full raw bootstrap replay requires the external local inputs and CPU compute; included table-replay checks do not claim newly recomputed intervals. No inference, training or judging was run during assembly. The original model access and licenses still apply.
