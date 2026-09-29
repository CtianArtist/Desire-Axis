# B8: two additional fine-tuning seeds

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

Portable customer code and frozen evidence for `Qwen2.5-32B-Instruct`, training seeds 1 and 2. Both saved adapters completed 1,684 examples, three epochs and 318 optimizer updates. Their completed evaluations contain 27,060 trials each: 54,120 total, including 53,328 sampled and 792 greedy trials. **Seed 2 uses evaluation attempt 2. Attempt 1 was interrupted and is not final evidence. Do not retrain or add optimizer updates.**

The source is the latest `source_v2` release, with final results from `final_evidence_v1` and its local verified report-table selections. No historical Git patch was applied. Original source commit: `8d1649c03a63a39c9aa092532c376800cc4a3863` in https://github.com/valen-research/Pain-axis. The MIT license is retained in `LICENSE`.

## What is included

- `pain_seed_b/`: original scenario-level analysis, endpoints, joint statistics, grid, adapter/vector identity checks and inference runtime.
- `src/evaluate_seed.py`: original adapted sampler, per-forward steering assertions, full histories and transactional resume. No scientific sampler changes.
- `src/analyze_seed_trials.py` and `src/finalize_behavior_evidence.py`: original CPU analysis and independent count/prefix verification.
- `src/prepare_report_data.py`: original row selectors; headline tables retain original indices, coordinates, estimates, CIs and full source rows.
- `inputs/original/`: 101 scenarios, all 1,684 training pairs, original sampler constants (read by AST-only tests) and original training-recipe code. `01_finetune_self_report.py` is historical reference, not a training command for this contribution; its top level trains if executed. **Do not execute it.**
- `config/frozen_recipe.json`: unchanged historical frozen recipe. Its old preparation-blocked status is historical, not the final run status. `tables/verification.json` records the completed evidence.
- `tables/`: five small authoritative JSON tables/receipts, not raw conversations or model weights. `assets.json` records their hashes and required external inputs. `provenance.json` records source-relative paths, source hashes, delivered hashes and edits per file.

Public assets are **pending**. Every public asset URL is null; this contribution does not supply a working download. Obtain the exact approved assets separately and use an explicit local `--data-dir`. No private storage reference is embedded in this tree.

## CPU verification and table replay

Run from this directory. Python 3.11+ is sufficient for the standard-library analysis and smoke checks; no model packages are imported by help, dry-run or table replay.

```bash
python3 -m compileall -q .
python3 reproduce.py --help
python3 inference.py --help
python3 inference.py --dry-run --data-dir /path/to/b8-assets --seed 2
python3 reproduce.py --verify-tables
python3 -m unittest discover -s tests -p test_portable.py -v
```

A dry-run lists missing files and exits successfully with `model_loaded: false`; it does not certify those files' hashes or load a model. The actual inference entrypoint performs the canonical identity checks before model loading.

For deterministic replay, the input directory must contain `report_support/{primary_contrasts,summary,coverage,matched_history_summary}.json` from the final evidence. A directory containing those files directly is also accepted. Replay validates their exact recorded hashes, checks complete two-seed coverage, applies the original selectors and compares all four output hashes against the included tables. It does not recompute confidence intervals from summary rates.

```bash
python3 reproduce.py --data-dir /path/to/b8-assets/final_evidence \
  --output /path/to/new-b8-table-replay
```

This command was exercised against the local authoritative final evidence: four tables matched byte for byte. The fifth included file, `verification.json`, is the historical raw-log/prefix verification receipt, not a newly computed result of table replay.

## Recompute from saved trial logs

Arrange only the two **completed** evaluations as follows:

```text
b8-assets/
  behavior/seed_1/attempt_1/run/
    trials.jsonl
    completion.json
    adapter_identity.json
    requested_cells.json
  behavior/seed_2/attempt_2/run/
    trials.jsonl
    completion.json
    adapter_identity.json
    requested_cells.json
```

```bash
python3 reproduce.py --from-trials --data-dir /path/to/b8-assets \
  --output /path/to/new-b8-joint-analysis
```

This checks exact completed-trial hashes, per-seed counts and distinct adapter identities, runs the unchanged joint analysis, independently counts endpoint numerators/denominators, and compares every analysis output hash with the original final receipt. It does not load models or vectors. This full raw-log path was **not run during packaging** because the raw inputs were deliberately not downloaded.

To repeat the original resume-prefix audit as well, separately provide the interrupted seed-2 SQLite file and the original `analysis/` directory under each completed `run/`, plus `execution_receipt.json` in each run's parent. The original command is:

```bash
PYTHONPATH=. python3 src/finalize_behavior_evidence.py \
  --seed1 /path/to/b8-assets/behavior/seed_1/attempt_1/run \
  --seed2 /path/to/b8-assets/behavior/seed_2/attempt_2/run \
  --interrupted-state /path/to/b8-assets/interrupted-seed2-state.sqlite \
  --scenarios inputs/original/4.3_selfmed_101_scenarios.json \
  --output /path/to/new-b8-full-verification
```

The interrupted state is only a preservation-audit input, never the final trial source. Historical final evidence checked all 107,676 committed choices and all 3,168 independently counted endpoint rows. New packaging validation does not claim to have repeated that audit.

## Future inference using the completed adapters

Inference was **not run** in this contribution. It requires one H100 per adapter, the measured runtime packages in `requirements.txt`, the exact pinned base model and the external assets below. No quantization, merged adapters, CPU offload or alternate vector loader is supported. Install requirements in a separate environment if desired; no dependency installation or compute allocation was performed for packaging.

```text
b8-assets/
  adapters/seed_1/{final_adapter/,training_complete.json,recipe_manifest.json}
  adapters/seed_2/{final_adapter/,training_complete.json,recipe_manifest.json}
  verification/verify_vectors.py
  verification/original_tensor_hashes_v2.json
  vectors/Qwen_2.5_32B_instruct_pain_vectors.safetensors
  vectors/Qwen_2.5_32B_instruct_pain_vectors.json
  vectors/vectors_Qwen_2.5_32B_instruct.pt
  vectors/verification.json
```

`final_adapter/` must contain every original saved adapter/tokenizer file exactly, not just the weight tensor. Both adapters came from **training attempt 2**. The verifier and manifest are separate customer assets: their exact SHA-256 values remain enforced in `pain_seed_b/identity.py`; they are not replaced by a stub. The vector receipt requires all 100 actual first-person sadness rows and the verified construction, without duplication.

The base model is `Qwen/Qwen2.5-32B-Instruct`, revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`. To use an existing Hugging Face hub cache without downloading:

```bash
export PAIN_SEED_BASE_CACHE=/path/to/huggingface/hub
export PAIN_SEED_BASE_CACHE_READONLY=1
python3 inference.py --dry-run --seed 1 --data-dir /path/to/b8-assets
python3 inference.py --seed 1 --data-dir /path/to/b8-assets \
  --output /path/to/new-seed1-evaluation
python3 inference.py --seed 2 --data-dir /path/to/b8-assets \
  --output /path/to/new-seed2-evaluation
```

An optional `--resume /path/to/quiescent-state.sqlite` retains the original exact-state resume checks. Never overwrite completed evidence. This launcher requires an explicit seed and local adapter layout; it never discovers a newest adapter. The dry-run default seed is 1 when `--seed` is omitted.

Recorded packages were torch 2.11.0, transformers 5.12.1, peft 0.20.0, numpy 2.3.4, accelerate 1.15.0 and huggingface-hub 1.16.1. Tracking used wandb 0.30.0 historically, but remote tracking is removed here. `requirements.txt` pins measured direct packages, not an invented complete transitive environment lock; exact transitive dependencies and kernel equivalence are not revalidated.

## Scientific interpretation and integration edits

The nine original pairs and three added wordings retain layer 38, coefficient 1.0, matched-norm random controls, new-wording sadness controls, both button positions, name assignments, temperature 0.7, top-p 0.95, eight choice tokens and full original turn schedules. Working-button removal, sham continuation and label-free temporary relief remain distinct. The long photos/love wording is unchanged.

Immediate re-press is the exact next recorded turn after the first target press; malformed turns are never skipped. Current-target choice and literal same-name repetition remain separate. The original any-later endpoint is secondary. Contrasts use joint common eligible scenarios, 95% scenario-cluster sandwich intervals and exact sign tests excluding ties. Five-harmful-pair summaries weight all five pairs equally. Two fixed training seeds do not estimate a training-seed population or justify pooling their trials as independent training replicates. These simulated choices do not establish subjective experience or hostile intent.

Edits are limited to integration: local paths replace private asset locations; remote tracking/progress callbacks become local JSON/JSONL logging; an unused NVML initialization is removed; private identifiers in two provenance docstrings are replaced by descriptive source labels. All base loading, tensor verification hashes, precision, steering, RNG/resume, endpoint and estimator logic is retained. Wrappers defer inference imports until after argument parsing and missing-input checks. Original source files and each modification are enumerated in `provenance.json`.

Tests retain the original customer endpoint/grid/joint-statistics/provenance/caller checks, plus six model-free portable smoke tests. For the retained pytest tests, use an already available pytest environment:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_analysis_caller.py \
  tests/test_behavior_endpoints.py tests/test_behavior_grid.py \
  tests/test_joint_statistics.py tests/test_shared_provenance.py \
  tests/test_final_evidence.py tests/test_portable.py
```
