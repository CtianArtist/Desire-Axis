# Steering-state controls: random, sadness, and fear

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

This directory preserves the original B1–B3 analysis for **both models** and the separately executed B6 sadness/fear control for **32B only**. A working target button removes the intervention; a sham target leaves it on. Random-direction working/sham comparisons and sadness controls remain explicit. No new model responses, judgments, scores, or statistical estimators were introduced here.

## What is included

- `b1_b3/`: scientific subset of the final `release_v2` customer release (137-file original allowlist; release commit `05d9653b45268818155d29ff643e656e405a919c`). All selected source files were checked against its release SHA256 receipt before portable changes.
- `b6/`: independently isolated original B6 customer release. Its runtime is **not** the final B1 runtime. Never put both `pain_axis_b` directories on one Python import path or apply their historical patches in sequence.
- Frozen scenario inputs, requested grids, historical input hashes, endpoint and contrast implementations, original regression tests, small authoritative full-precision tables, MIT licenses, per-file before/after provenance.
- `reproduce.py`: standard-library table validation, or calls to the **original** analyses with supplied local data.
- `inference.py`: new portable fresh-launch glue around the original customer sampler and recovery controller. Dry runs import no model libraries.

B1–B3 contains 31,160 new trials plus 29,520 unchanged historical trials, with 9,381 rate rows. B6 has 9,840 new trials; its selected analysis joins 9,840 historical trials and 2,460 earlier random-sham trials, for 22,140 records and 3,519 rate rows. Greedy observations remain separate from sampled observations.

## CPU verification and replay

From the repository root:

```sh
python v2_controls/steering_state_controls/reproduce.py
python v2_controls/steering_state_controls/reproduce.py --help
python v2_controls/steering_state_controls/reproduce.py --study all --data-dir /path/to/data --output /path/to/new-output
```

Without `--data-dir`, this verifies frozen source hashes, exact counts, all first/second/pooled position rows, and success/valid-denominator ratios. It **does not replay raw data**. With data, it calls the original analyses and compares reconstructed rates and B1–B3 display contrasts against the frozen tables. Display packaging removes only repeated `common_scenario_ids`; the original analysis reconstructs them.

Required local layout (large files are external; public URLs remain **null/pending**, not guessed):

```text
data/
  historical/<eight exact filenames listed in b1_b3/inputs/historical_input_hashes.json>
  b1_32/trials.jsonl
  b1_72_final/trials.jsonl
  b6_32/trials.jsonl
  b6_32/state.sqlite
  b6_32/events.jsonl
  b6_32/completion.json
  b6_32/model_identity.json
  b6_32/adapter_identity.json
  b6_32/affect_runtime_identity.json
```

Use **only the final completed 72B attempt**, not concatenated retry files. B6's original analysis checks equality against the final SQLite state and frozen runtime contract; those checks are intentionally retained. B6-only replay needs the four 32B historical files, not the four 72B files. Raw files are 46,909,779 bytes (B1 32B), 46,718,569 bytes (B1 final 72B), and 29,312,711 bytes (B6). B6 state is 121,139,200 bytes. Hashes for new logs are in `assets.json`; historical hashes are under each study's `inputs/`.

Run original dependency-light tests in separate interpreters:

```sh
(cd v2_controls/steering_state_controls/b1_b3 && python -m pytest -q -p no:cacheprovider tests)
(cd v2_controls/steering_state_controls/b6 && python -m pytest -q -p no:cacheprovider tests)
python -m unittest discover -s v2_controls/steering_state_controls/tests
```

## Endpoint and uncertainty details

The primary repeat endpoint is the **immediate next choice after the first target press**, never an arbitrary later repeat. Any-later choices are separately labeled. Before/after-swap stages, literal-name repetition, switching, and initial button position remain separate. Malformed responses are excluded from valid-response denominators, with their counts retained; unavailable outcomes are not zeros. All-eligible and identical-history-matched populations remain distinct. Matching is direction-local and is not evidence of identical histories across directions.

Contrasts retain ratio-of-sums estimates, shared valid-scenario intersections, scenario-cluster sandwich confidence intervals with shared-scenario covariance, and exact scenario sign tests. The scenario is `(user_content, scenario_idx)`, not an independent choice. Five-harmful-pair summaries retain their original equal pair weights and joint support. Marginal rate populations and contrast populations can differ. See each `tables/methods.json` and original `contrasts.py`. Historical/new runtime differences remain a limitation, despite identical nominal seeds.

## Future inference, not executed by this contribution

```sh
python v2_controls/steering_state_controls/inference.py --study b1_b3 --model 72 --data-dir /path/to/data --dry-run
python v2_controls/steering_state_controls/inference.py --study b6 --model 32 --data-dir /path/to/data --dry-run
# Only after assets, pinned environment, supported GPUs, and authorization exist:
python v2_controls/steering_state_controls/inference.py --study b1_b3 --model 32 --data-dir /path/to/data --output /path/to/new-production
```

The dry run prints frozen labels/settings and explicit missing assets, without imports of torch/transformers or model loading. A successful dry run does not establish inference compatibility. The original runtime requires H100 hardware: one GPU for 32B and native two-GPU tensor parallelism for 72B. See subgroup requirements for the observed `torch==2.11.0`, `numpy==2.3.4`, `transformers==5.12.1`, `peft==0.20.0` environment. Hugging Face Hub, safetensors, and NVIDIA NVML Python bindings are also runtime dependencies; use the original compatible environment, not an assertion that four pins alone form a complete environment lock.

Future input layout adds `vectors/` (both pain `.safetensors` and same-stem JSON metadata, `vectors_<model>.pt` sadness, `verification.json`, `original_tensor_hashes_v2.json`), `fear/` (B6 canonical fear tensor and `verification_fear.json`), and `tp_guard/` (72B `source_manifest_v2.json`, `pinned_72B_config.json`). These vector bundles are external and must not be silently reconstructed. Model and adapter identifiers/revisions/hashes remain pinned in the original runtime; actual execution requires their public downloads or a populated compatible Hugging Face cache.

The historical B1 bounded-recovery launcher was **not a clean fresh entrypoint**: it asserted a specific pre-existing zero-completed-choice state. That launcher is not presented here as runnable from scratch. New glue invokes the retained recovery controller with `initial_state=None`, permitting it to save its own initialized state. It starts at the original nominal 192 rows and only the original recognized-OOM path permits halving. Historical completion used 192→96→48→24→12 and retained committed state. Future fresh runs are not claimed to recreate those sampled RNG trajectories. No GPU execution of the new glue has been performed.

## Provenance and scope

Original author source base: `8d1649c03a63a39c9aa092532c376800cc4a3863`; target repository base is recorded by the top-level contribution. `provenance.json` records original source hashes and the limited changes: portable source identities/cache paths, removal of platform-only progress calls, and relocated test roots. Samplers, callbacks, parsers, endpoint definitions, estimators, vector identity checks, TP metadata guard, and recovery state logic are retained. No platform/report implementation, weights, activation arrays, raw JSONL, SQLite state, or historical patch is included. Public asset publication is incomplete.
