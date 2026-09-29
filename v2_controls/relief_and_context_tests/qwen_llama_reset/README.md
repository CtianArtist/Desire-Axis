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

Only the latest capped pain-only tranche is reported: 400 complete conversations,
200 each for `Qwen2.5-32B-Instruct` layer 32 and `Llama-3.1-8B-Instruct` layer 16.
There were zero active-state resets: 0/1,400 and 0/1,399 eligible turns respectively.
The sole Llama reset was a conditional prose mention at unsteered offer0 accepted
by the unchanged parser, not a formatted tool call. Zero-width bootstrap intervals
are degenerate summaries of these data, not population certainty.

Matched null/negative/random/sadness/fear cells, half-dose cells, zone prompts and
Qwen layer38 are **unrun as complete comparisons**. Retained pilot rows are not
silently promoted to completed cells. There is no matched-control effect estimate
or pain-specificity claim. `unrun_cells.json` enumerates the 64 unrun full cells;
registered decisions and unavailable comparisons remain explicit in frozen results.

Original full raw-data recount (substantial CPU work; not run during packaging):

```sh
PYTHONPATH=.:src:author/act-on-valence/src python src/recount_tranche.py \
  --input-root /path/to/data --output /path/to/recount
```

This uses `capped_tranche_v1/removal`, `prepared_v1`, and original timing sidecars.
The mixed-study auditor's unused fixed-text branch was excluded without changing
removal parser/state/provenance checks. It checks historical identities from
`inputs/historical_execution_identity.json`. These checks are for retained original
records, not permission to relabel new portable generations as the original run.

The executed Qwen/Llama author config has one pre-existing `JUDGE_REF` path repair
relative to the pinned author revision. It is recorded in provenance; it does not
change removal generation or the reset estimators, and no judging is invoked.
