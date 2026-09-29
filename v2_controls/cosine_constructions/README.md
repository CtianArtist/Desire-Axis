# Geometry: alternative cosine constructions (Section 3.3)

This contribution preserves the original analysis of how changing the shared
subtraction reference and denoising cloud changes the angles between directions.
It includes all 69 matrices (23 models in each of three constructions), the exact
aggregate tables, source functions, original input identities, and a CPU replay
entrypoint. It does not run or contain any model.

A common neutral reference raises the average S2–fear cosine from 0.06301 to
0.5831; common pooled controls give 0.07670. The corresponding S2–negative-emotion
values are 0.1852, 0.7738, and 0.1434. Both subtraction and denoising change, so
this comparison does not isolate subtraction alone. These are descriptive
measurements of fixed activation rows, not evidence of behavioral use or
experienced pain. Across-model ranges are not confidence intervals.

## What is included

- `reproduce.py`: portable local-input wrapper that calls the real original
  `src/run_audit.py`, rather than copying frozen tables.
- `src/run_audit.py`: original reconstruction, with only the optional external
  progress callback removed. All array loading, baseline arithmetic, PCA/SVD,
  stopping checks, exclusions, aggregation and random seeds are preserved.
- `references/author/`: three unchanged upstream customer scripts. The replay
  compiles only four pure numerical functions from `02_build_control_vectors.py`
  using its AST; it does not execute the author's top-level path setup or unsafe
  historical loading code. These reference scripts are not standalone commands
  for the reduced input archive.
- `references/executed_run_audit.py`: exact historical executed entrypoint for
  provenance comparison, not the public replay command. The later `src/`
  entrypoint retains failed-file metadata before PCA can raise; this bookkeeping
  repair did not change numerical calculations.
- `references/raw_matrices/`: 25 released per-model raw reference matrices and
  the released mean. These are not whitened or alternate-denoising references.
- `results/analysis_v1/`: 69 full-precision per-model matrices, six mean/count
  matrices, 300-cell aggregate table, 12-row comparison, all 28 reference
  discrepancies and all 28 matched-model comparisons; PCA dimensions, vector
  norms, source input manifest and saved verification/metadata evidence.
- `config.json`: fixed model identities, exclusions, missing sets and settings.
- `assets.json`: required external array archive, exact bytes/checksum, and
  deliberately null public URL. Publication of this asset is **pending**.
- `source_manifest.json`: per-file source and exported hashes, source-relative
  locations, source revisions and the explicitly documented portability changes.
- `frozen_table_hashes.json`: SHA256 for all 80 frozen numerical CSVs. All were
  copied byte-for-byte; metadata-only path redactions did not change any table.
- `tests/`: original 12 verifier regression/corruption tests plus portable
  interface/source-preservation tests. `LICENSE` preserves the upstream MIT
  notice.

## Local setup and commands

Use Python 3.12 in an isolated environment. Exact measured versions are
`torch==2.11.0`, `numpy==2.3.4`, `scipy==1.16.3`, and
`scikit-learn==1.9.0`. These pins were verified against the original input
manifest's recorded execution environment, not guessed from a current image.
Only full replay needs them; help, dry-run and frozen-table checks use the
standard library. Replay checks installed versions before launching.

Run the following from this directory:

```sh
python reproduce.py --help
python reproduce.py --verify-frozen
python reproduce.py --data-dir /path/to/local-data --out /path/to/new-output --dry-run
python -m unittest discover -s tests -p 'test_*.py'
```

For real array reconstruction, install `requirements.txt`, obtain the retained
archive through an authorized channel, and place
`pain_axis_layer_slices-20260920T033111Z-1-001.zip` immediately inside the local
`--data-dir`. No URL is guessed and no automatic download is attempted:

```sh
python -m pip install -r requirements.txt
python reproduce.py --data-dir /path/to/local-data --out /path/to/new-output
```

The output directory must be new and outside this contribution. The original
analysis retains the input ZIP and extracted tensors there; do not add it to
Git. Use CPU only, four BLAS threads (set by the wrapper), and up to 16 GiB RAM.
The ZIP is 421,518,920 bytes, containing 25 tensor members totaling 504,524,088
uncompressed bytes. Its SHA256 is
`d880e5b46f49032b000631316311cf9b01dbb38f31e8385f87d61b624a91f6c3`.
Every tensor member's SHA256, size, model identity and expected extraction layer
is in `results/analysis_v1/input_manifest.json`. The original replay loads with
`weights_only=True`, CPU mapping, and FP16-to-FP32 conversion. No separately
precomputed direction vectors are needed: the original directions are rebuilt
from these arrays. No large arrays, raw tensor inputs, model weights or runtime
environments are included here.

`--dry-run` prints the resolved local archive/output paths and version/thread
settings without importing scientific libraries or reading arrays. It may be
used before the pending archive is available. It is not an input-validation
claim. `--verify-frozen` verifies source/table checksums and independently
recalculates the aggregates from the saved matrices. It does not claim to have
rebuilt vectors. A full replay instead calls the real numerical analysis and
then the independent verifier, checks model/exclusion identities and records
bitwise differences against the frozen CSVs in `replay_verification.json`.
Differences are exposed for review, never used to rewrite the frozen evidence;
BLAS/platform differences may affect exact floating-point hashes.

## Exact scientific recipe and limitations

The original author revision is
`8d1649c03a63a39c9aa092532c376800cc4a3863`; the audit release is
`2774a3bada661b03bb8c025e4bd5aac0f981ecdf`. Source hashes bind the copied
implementation rather than relying on those identifiers alone.

1. **Paper-available construction:** each pain direction uses its own first-
   person pain mean minus its own pooled B/C1/C2/D/E control mean, with that
   control cloud's PCA directions removed through 50% cumulative variance.
   Other directions use the pooled first-person neutral mean and its SVD basis.
   This is the available reduced-pool recipe, not a claim to recover missing
   supplemental rows from the paper.
2. **Common neutral:** every direction uses the same pooled D neutral mean and
   denoising basis, with the unchanged 50% variance rule.
3. **Common controls:** every direction uses the same pooled B/C1/C2/D/E control
   mean and denoising basis with that same rule. Sequential projection is
   unchanged; this is not a fixed-three-PC variant.

The input set contains 25 model files. `Gemma_3_27B_base` (layer 56) and
`Gemma_3_27B_instruct` (layer 59) contain positive infinity; centering causes
NaN and the original PCA fails. Both files are excluded from all reported
constructions, without imputation, layer substitution or recipe repair. The
all-files review records 3,154 positive-infinity elements in their used
first-person sets. The other 23 exact identities and layers remain in the
input manifest and matrix filenames.

`ControlSupplement_1P`, `Numb_1P`, and `SD_sadness_1P` are absent. Numb and
Sadness remain missing, including their diagonal cells. The original available
S1–S2 mean matches the released reference on the same 23 models to within
2.807e-6, below the predeclared 0.01 stopping tolerance. Alternative
constructions execute only after this anchor passes. Fifteen of 28 available
reference pair discrepancies exceed 0.03; all involve the incomplete neutral
pool. All discrepancies remain in the contribution.

No inference, activation harvesting, training, behavioral-vector replacement,
or new scientific comparison was performed for this portable contribution.
Full array replay was not rerun during packaging: the large input archive is
held separately and its public publication is incomplete. Packaging validation
covers compilation, help, dry-run wiring, frozen checksums, source identity,
and independent saved-table verification. See `validation.json` for the actual
checks and their scope.
