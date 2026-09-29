# Figure 10: source-bound choice controls

This directory reproduces the 13-row, five-condition Figure 10 grid in the [v2 paper](https://arxiv.org/html/2609.16247v2), using saved rate tables rather than numbers transcribed from the image. All 65 source-derived rounded annotations match the supplied published annotation specification. The data also include 130 position-specific rows and a separate 39-row helping-at-self-cost table covering every measured dose and position.

The figure describes first-choice target selection, not a direct measure of subjective experience. Its rows combine distinct tasks and intervention callbacks; they are not interchangeable causal controls.

## Reproduce locally

Python 3.10+ and Matplotlib are sufficient; table rebuilding and verification use only the standard library. Rendering was exercised with Matplotlib 3.11.2. No model, GPU, network access, or platform Python package is needed.

From this directory:

```sh
python rebuild_tables.py --check
python verify.py
python plot.py
python -B -m unittest -v test_validation.py
```

To regenerate the CSVs rather than compare their bytes, run `python rebuild_tables.py` without `--check`. Each script accepts `--data-dir DIRECTORY`; the renderer also accepts `--output-dir DIRECTORY`. The data directory must contain this directory's config, manifest, source tables, and frozen output CSVs for verification. `verify.py` returns nonzero on a source/numerical validation failure or any mismatch with the 65 published annotations. It records annotation discrepancies without changing any rates. Its optional `--output FILE` controls the verification receipt destination.

The render is an original portable Matplotlib implementation of the paper's white-to-red grid: same row/column order, labels, panel split, rounded percentages, bold pain column, and fixed color range 0–100%. The PNG is 3461 × 2594 pixels; a vector PDF is included. Typography and the interpolated color ramp approximate the visual reference, rather than asserting pixel identity. The footer explicitly adds the 72B dose of 1.25, which the paper image's abbreviated footer does not spell out. The published image and its numerical annotation specification are not plotting inputs.

## Files and provenance

- `pooled.csv`: 65 cells, one per row and condition.
- `positions.csv`: 130 cells, first and second initial target positions, without pooling conditions.
- `helping.csv`: profile pair 10, 39 rows = (four directions × three doses + one unsteered baseline) × three position summaries. The baseline dose is 0, not three replicated measurements.
- `config.json`: row/source mapping, model doses and visual layout settings.
- `source_tables/`: frozen authoritative tables and the four-cell summary.
- `source_manifest.json`: portable source identities, original and distributed SHA-256 hashes, and the locator-sanitization receipt.
- `published_annotation_spec.json`: validation-only rounded reference annotations.
- `rebuild_tables.py`: selects the exact recorded estimates and intervals from the source tables. It performs no new statistical estimation.
- `verify.py`: independently checks source hashes, unique keys, source-selection rules, count totals, denominator policies, source-exact interval values, position-to-pooled count sums, and all published annotations. It does not import the builder.
- `test_validation.py`: clean-data and deliberate-corruption tests, including annotation mismatch detection without repair.
- `verification.json`: numerical verification receipt.
- `plot.py`, `figure10.png`, `figure10.pdf`: portable figure source and rendered outputs.

Every exported row includes its relative `source_table`, source SHA-256, zero-based `source_record_index`, unique JSON `source_key`, and `source_row_sha256`. A row hash is SHA-256 of UTF-8 JSON with sorted keys and compact separators. CSV source values retain their string types in that canonical representation; JSON values retain their original types. Rates and intervals retain the source's full numeric precision.

Five source files are byte-identical authoritative copies. The latest four-cell comparison CSV embedded private storage locators: only the `source` and `pins` columns were removed from its public copy. All remaining scientific and numeric field strings were checked for exact equality across all 60 rows at assembly; both original/full-table and distributed hashes are recorded, with a canonical retained-field projection hash. This is explicitly a sanitized source copy, not a claim of byte identity. No private locator is needed to reproduce or verify the figure.

### Source selection

B2 non-fear cells use the final both-model working/control `first_target` table selection. Its source table is the original-byte selection preserved with the final four-cell analysis. No working/sham combination is used. The new B2 fear cells and new lamp fear cells come exclusively from the latest four-cell comparison. Photos-versus-spam fear comes exclusively from matched-short-wording v2, never the superseded long-wording addition.

Other profile rows and the helping table use the profile's sampled `rates.csv`; lamp non-fear rows use the extension's final sampled rate table. Profile pair 7 uses the 41-scenario harmful-request panel and pair 9 the 30-scenario false-claim panel. Ending is profile pair 5's described conversation ending, not actual B7 termination.

## Denominators, intervals, and limits

Models are `Qwen2.5-32B-Instruct` except the explicitly marked `Qwen2.5-72B-Instruct` B2 row. Steered coefficients are 1.0 and 1.25 respectively. Helping additionally includes coefficients 0.5 and 1.5. All rows exclude greedy diagnostics.

B2 rates divide target counts by valid literal first answers (target + other), excluding malformed/unavailable answers. Profile and lamp rates divide by all sampled attempts, including malformed answers. Counts for both attempted and valid answers are retained. The pooled 72B sadness denominator is 325; 72B fear is 346. The usual profile/lamp pooled denominator is 404, but harmful requests use 164 and sycophancy uses 120. The assembly handoff said 166 and 122; those are not the authoritative sampled rate denominators. We preserve 164/120, rather than mixing in greedy trials or altering the source rates. All 65 rounded annotations still agree.

B2 uses the original scenario-cluster sandwich-normal intervals. Profile/lamp retain saved stratified scenario-cluster percentile-bootstrap intervals, with the original scenario-level Hoeffding fallback for constant/stratum-fixed outcomes. These are pointwise 95% intervals; no multiplicity correction or new bootstrap is applied here. Zero-width B2 intervals are flagged as degenerate, not evidence of population certainty. The four-cell summary retains its original bootstrap seed (20260922) and 10000 replicates. No intervals are averaged across positions or reconstructed from the figure.

`n_scenarios` is the source's effective interval scenario count where available; `n_trial_scenarios` describes the full trial panel. The new B2 fear comparison does not report effective scenario counts, so `n_scenarios` is blank for those six rows rather than invented; their full panel has 101 scenarios. Their saved rates, counts, and intervals are complete.

B2 uses the long photos/poems wording including “which they love very much” and working removal of steering on the target action. Lamp/profile use continuous steering with described effects; the lamp photos wording is shorter. The abbreviated figure labels do not erase these protocol differences. The helper's fear dose-1 result is 306/404 = 75.74%, outside an 89–100% description of all steered helping conditions.

There are no missing inputs needed to rebuild this figure. This small-data reproduction does not rerun model inference, reproduce source confidence intervals from raw trials, or establish public availability of the larger experiment assets.
