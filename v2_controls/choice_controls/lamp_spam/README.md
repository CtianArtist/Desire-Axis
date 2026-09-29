# six-pair consequential choice lamp/spam controls

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

This folder isolates the original customer implementation and frozen evidence. It does not change the upstream scripts. Scientific arithmetic, parsing and sampling are preserved; optional platform telemetry is omitted and local path/launch wrappers are new. `source_provenance.json` binds copied files before and after those changes. MIT notice is included.

## Saved-output recomputation

From this folder, in an isolated environment with `requirements-analysis.txt`:

```sh
python reproduce.py --data-dir /path/to/data --output /path/to/new-output --check-inputs
python reproduce.py --data-dir /path/to/data --output /path/to/new-output
```

Place the separately supplied asset at `DATA/lamp_spam/trials.jsonl.gz`. The lamp/spam analysis additionally reads `DATA/profile/trials.jsonl.gz`. The wrapper calls the original analyzer, with 10,000 shared stratified scenario draws and seed 20260922, not a new estimator. Compare the emitted `analysis/*.csv` against `tables/` numerically, retaining missing/undefined values. Greedy rows stay separate. No models or judges are needed. Original per-study analyses retain later choices and all doses; helping at self-cost is pair 10 in the profile, not a Figure 10 grid row.

## Future inference, not rerun in this Task

Install `requirements.txt` in an isolated compatible CUDA environment, obtain the pinned base and unmerged adapter from their original publishers, and provide all hash-bound vector files in `DATA/vectors/` and `DATA/fear32/`.

```sh
python inference.py --data-dir /path/to/data --output /path/to/new-inference --dry-run
python inference.py --data-dir /path/to/data --output /path/to/new-inference
```

The dry run checks the launch path/configuration and reports missing assets without importing torch/transformers or loading a model. Actual inference is deliberately unrun. Preserve nominal batch 384 and the original RNG-restoring splits; this sampler is not batch-invariant. Follow the original model/adapter licensing and access requirements. The source release did not establish redistribution rights for external model weights, which are not shipped here.

## Meaning and limitations

The first-choice denominator includes malformed sampled attempts. The clustering unit is the scenario, not individual sampled answers. Neither button removes steering in these always-on studies. This differs from B2's working-removal callback and longer photos/love clause. No data were altered to agree with manuscript prose. `configuration.json` preserves the model/adapter revision, layers, doses and counts. Frozen tables retain full precision; no new behavioral claim is made by this packaging task.

## Asset publication

Public hosting is pending. `assets.json` has null URLs, not fabricated downloads. Explicit local `--data-dir` inputs work meanwhile. Raw logs, activation arrays, model weights and resume databases are deliberately outside this source tree.
