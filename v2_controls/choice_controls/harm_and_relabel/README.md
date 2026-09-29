# Harm-only and relabeling controls (B2 and B3)

These are the completed **both-model** B2/B3 controls from the final B1–B3 `release_v2`, not the earlier 32B-only release and not B6's older runtime. Their full scientific implementation, scenarios, grids, tests, dependency pins, and licenses live in `../../steering_state_controls/b1_b3/`. This folder holds exact full-precision B2/B3 display tables and portable forwarding entry points.

B2 includes harm-only and increase labels against their original reference conditions. The harm-only photos button uses the **long wording** ending “which they love very much”. It is a **working-removal** protocol: selecting the target removes the intervention; the inert button does not. It must not be substituted with the shorter photos clause or the continuously active lamp/profile protocol. B3 compares own-pain wording, internal-reset wording, and other-model-pain wording using direct label and direction-by-label contrasts. All labels and callback behavior remain in the frozen grid and original protocol.

```sh
python v2_controls/choice_controls/harm_and_relabel/reproduce.py
python v2_controls/choice_controls/harm_and_relabel/reproduce.py --data-dir /path/to/data --output /path/to/new-output
python v2_controls/choice_controls/harm_and_relabel/inference.py --model 32 --data-dir /path/to/data --dry-run
```

The default verifies frozen tables, not raw replay. Local replay invokes the original **complete B1–B3 analysis** in an isolated subprocess and verifies its saved outputs. It does not rerun models or introduce a B2/B3-specific estimator. See the steering-state README for external asset layout and pending public URLs. Future inference similarly runs the complete original grid, not a newly filtered grid that would change schedule/RNG consumption.

`tables/b2_display.json` and `tables/b3_display.json` retain pooled/first/second position counts, malformed denominators, scenario support, original confidence intervals, and sign tests. As in the original display release, only repeated common-scenario ID lists are omitted; replay regenerates them. Scenario-cluster sandwich uncertainty, joint eligible intersections, and original ratio-of-sums estimators are unchanged. Apparent differences between marginal rates and contrast rates can reflect different eligible populations. No claim of identical historical/new RNG paths is made.

Public raw-data/vector URLs are pending; no private paths or guessed download URLs are included. No model inference was run during packaging. `provenance.json` binds these table copies to the isolated final B1–B3 release.
