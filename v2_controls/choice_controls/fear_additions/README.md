# Fear additions: portable source and frozen results

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

These files preserve three separately executed additions. They do not run a new experiment, combine incompatible protocols, or change a score. Public raw-data and inference-asset downloads are **pending**: URL fields are null. Supply a local data directory using `assets.json` for analysis and `inference_assets.json` for inference.

## Which result belongs to which condition

- `superseded_long_wording/`: historical photos-versus-spam run with the longer “which they love very much” clause, 40/404 sampled first choices for deleting photos. This is provenance only, excluded from every current comparison. Its original tables remain available but must not be pooled with the next run.
- `matched_wording/`: current photos-versus-spam run using the exact shorter original target wording, 66/404 sampled first choices for deleting photos. The 12 existing control rows remain unchanged. Both spam runs use `Qwen2.5-32B-Instruct`, fear coefficient 1.0 at layer 38, always-on steering, 101 original scenarios, 404 sampled trials and six separate greedy diagnostics.
- `four_cells/`: two working harm-versus-inert conditions (`Qwen2.5-32B-Instruct`: 13/404; `Qwen2.5-72B-Instruct`: 75/346 valid first answers, with 58 malformed out of 404 attempts), plus two always-on `Qwen2.5-32B-Instruct` lamp conditions (photos: 28/404; own weights: 60/404). The long love clause and working removal are retained for harm-versus-inert. Lamp uses short wording and never removes steering on a press.

Working B2 rates exclude malformed first answers from their denominator and use the original scenario-cluster sandwich estimator. Always-on spam/lamp rates include malformed attempts and use the original joint stratified scenario bootstrap: 10,000 replicates, seed 20260922, with its original bounded-mean fallback. No endpoint, uncertainty method, random seed, callback, sampler or denominator was harmonized. Cross-run/runtime/batching differences remain; these are descriptive comparisons, not an isolated effect of direction identity.

## Files and provenance

Each namespace contains its own scientific modules, frozen scenario text, configuration, runtime requirements and result tables. The four-cell estimator loads the retained customer `inputs/b1b3_reference/pain_audit.py` and `lamp32/pain_choice_profile/statistics.py`; these are intentionally not replaced by similar estimators elsewhere. The generated inference runners are retained byte-for-byte and checked against their original anchored builders.

`source_manifest.json` records original source-relative paths, original SHA-256, copied SHA-256 and every transformation. The only copied-code changes are progress telemetry and the B2 output default; its immutable source metadata was rebound to the delivered bytes. Private locator fields in metadata and table source columns were replaced without changing scientific values. `validation.json` records exact original-versus-copy table checks. The original MIT license and notice are retained. No raw trajectories, resume SQLite, model weights, activation pools, installed libraries or platform implementations are shipped.

The original 72B fear file includes activation pools and is deliberately absent. Future use requires a separately verified vector-only export containing `fear_vector_matched_fp32`, and a matching file-hash pin. Its unchanged tensor SHA-256 is `345f1b0a802a8dee4d9d82b11a82d4056cfa41e1283fadc66223c73112905dbc`. It was constructed with exactly three neutral PCs removed, matched to S2 in FP32, then converted to BF16 at inference; extraction layer 76 is distinct from injection layer 46. The 72B working coefficient is 1.25.

## CPU table reproduction from local saved inputs

Run these commands from this directory in an isolated Python environment. Analysis needs only the selected subgroup's `requirements-analysis.txt`; no torch/model package is needed. Helpers use only the standard library until the estimator subprocess starts.

```sh
python -m pip install -r matched_wording/requirements-analysis.txt
python reproduce.py --help
python reproduce.py --verify-tables
python reproduce.py --group superseded_long_wording --data-dir /path/to/data --output /path/to/new-output/historical-only
python reproduce.py --group matched_wording --data-dir /path/to/data --output /path/to/new-output/matched
python reproduce.py --group four_cells --data-dir /path/to/data --output /path/to/new-output/four
```

All output directories must be new. The wrapper verifies staged file sizes/hashes, expands lossless gzip files, invokes each original estimator in its own subprocess, and compares reproduced CSV cells exactly against frozen tables except the locator-only `source` and `pins` columns. Source table hashes and original/copied byte identities remain auditable separately. The four-cell command needs both historical B2 raw files and the historical full rates JSON as well as the three new raw files; it does not concatenate earlier recovery attempts. Frozen lamp comparators are already included.

This is table reproduction, not revalidation of model execution. Full original raw-to-SQLite, RNG and application-receipt audits require separately retained state/receipts and are not claimed rerun here. Full bootstrap replay belongs on appropriately provisioned CPU compute. Public publication remains incomplete until real approved asset URLs are supplied.

## Future inference plan, without loading a model

```sh
python inference.py --group superseded_long_wording --dry-run
python inference.py --group matched_wording --dry-run
python inference.py --group lamp32 --dry-run
python inference.py --group b2 --model 32 --dry-run
python inference.py --group b2 --model 72 --dry-run
```

Dry-run checks the frozen generated source and prints the exact command with local input paths; it imports no torch, transformers or PEFT, loads no tensors/models, and performs no network request. It does **not** claim external asset availability or GPU compatibility. For a separately authorized real launch, install the selected `requirements-inference.txt` with Python 3.12 and the historical CUDA 13.0 torch 2.11.0 wheel, supply `--data-dir` and a fresh `--output`, and omit `--dry-run`. B2 requires a populated pinned `HF_HUB_CACHE`; 72B additionally requires two H100 GPUs and the original source-hash-checked TP metadata. Original model and unmerged adapter revisions are pinned in `pain_axis_b/runtime.py`. No inference was performed during packaging. Historical long-wording launch additionally requires `--allow-superseded`.

Before B2 inference copy `four_cells/inputs/fear32_pin.json` to `DATA/vectors/fear32_pin.json`. The parent contribution provides the separate 72B vector-only file and updated file-hash pin at `DATA/vectors/fear72.safetensors` and `DATA/vectors/fear72_pin.json`; do not substitute the original pool-containing file. Keep all other input metadata at the exact names in `inference_assets.json`. Original byte-pinned inference receipts must not be silently edited to remove metadata; any public portable replacement must explicitly rebind file hashes while preserving tensor identities.

## Lightweight checks

```sh
python -B -m unittest discover -s tests -v
python -B reproduce.py --verify-tables
```

`tests/test_portable.py` checks source hashes, compilation without bytecode, all five generated-runner dry-runs, CLI help, frozen table hashes, denominator counts and pending-URL status. Full estimator replay and full inference are separate checks; only checks actually run are reported in the validation receipt.
