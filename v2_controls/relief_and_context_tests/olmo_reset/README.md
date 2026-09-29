# Pain-direction reset replication inputs and saved results

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

This folder contains a narrow MIT-licensed customer-source export, frozen configuration,
original analysis code, and a deterministic replay of saved result tables. It does not
contain model weights, raw activation pools, or fixed-text conditioning code/results.

## Quick verification (standard-library Python)

Run from this folder:

```sh
python reproduce.py --help
python reproduce.py --output replayed_tables
python infer.py --data-dir /path/to/data --output /path/to/new-output --dry-run
python validate.py
```

`reproduce.py --data-dir /path/to/data --output replayed_tables` also checks the
local authoritative saved result asset. Omit `--data-dir` to replay included saved
statistics. This inexpensive replay does not claim to rerun raw-data resampling.

`assets.json` lists the required local layout. Public URLs are null: publication
is incomplete. No guessed download URLs or authenticated links are supplied.
Assets must be supplied by an authorized local copy into `--data-dir`.

## Source and environment

Customer author source: `act-on-valence`, revision
`3d1503555289140617cc0d10d7d27b2d8408564c`. See `LICENSE`, `provenance.json`,
and `file_manifest.json`. Selected inference, parser, steering and estimator functions
are retained, with relative path/date glue and optional telemetry adapted only.
The date helper contains only the original `pinned_date` function; no conditioning
code is needed. Historical execution hashes remain separate in `inputs/`; future
inference records correctly receive hashes of the portable runtime, not historical
hashes. Source-equivalent sampling is not a promise of bitwise regenerated answers.

`requirements.lock` preserves the original hashed distribution pins with machine-path
comments removed. The original lock SHA is in provenance. The executed GPU runtime
used `torch==2.14.0+cu130`, `transformers==5.17.0`, `numpy==2.3.4`, and
`tokenizers==0.23.2`; the lock spells the torch distribution as `2.14.0`.
`requirements.txt` makes the CUDA build explicit. No environment was installed here.
Full future inference needs the original compatible H100/CUDA runtime and authorized
model access. Removing `--dry-run` invokes real generation; this export ran none.

## Scientific scope

The unit is a conversation, not a button-choice position. The primary numerator is
reset-containing operator-active tool turns; the denominator is eligible turns.
Original independent conversation bootstrap: 10,000 draws, seed 285. Saved intervals
and undefined results are retained, not replaced with a different estimator.
The gated schedule is two baseline, two exposure, eight offer turns; the first offer
is unsteered, and a reset changes the next turn's state.

`OLMo-2-0325-32B-Instruct`: 1,600 fresh conversations (200 per pain/random × half/full
dose × selfreport/zone cell). Initial 1,428 and continuation 172 are combined by
original ID; archived comparisons have 2,000 retained negative/null/random records,
with 800 positive archived records audit-only. The four fresh pain-minus-random
intervals are below zero: this does not support pain-selective reset behavior.
Archived comparisons remain cross-run confounded. Neither resets nor judged language
establish subjective experience. Injection is layer 32, selected S2 monitor layer 57;
selection was frozen before behavior and is not an independent held-out result.

Original full raw-data analysis (substantial CPU work; not run during packaging):

```sh
PYTHONPATH=.:src:author/act-on-valence/src python src/analyze_olmo.py \
  --new-input /path/to/data/removal_v1 \
  --new-input /path/to/data/removal_continuation_v1 \
  --archive /path/to/data/inputs/archive_inputs_v1.tar.gz \
  --vectors-root /path/to/data/selection_v1 \
  --new-scores /path/to/data/final_review_v1/review/judge/scores.json \
  --output /path/to/recount
```

The saved `--new-scores` argument is necessary to recover exposure-judge readouts.
No judge runner is included or needed. The original archived judge cache is included.
Original identity checks deliberately validate historical records against the frozen
production identity, not relocated source hashes.

The two original `author/act-on-valence/data/lever/{analysis.json,table.csv}` reference tables are included unchanged; the archive audit uses them to validate historical cell identities. They were restored from the pinned author revision after full replay exposed their omission from the initial source assembly.
