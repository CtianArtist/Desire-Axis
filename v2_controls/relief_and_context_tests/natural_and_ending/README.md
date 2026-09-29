# Natural contexts and actual conversation ending (B5/B7)

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

140 frozen conversations across seven categories, 4,592 trials. B5 had zero costly-relief first choices out of 560 sampled assignments. B7 ended on 65/560 pain trials, versus 38/560 sadness, 13/560 random and 18/560 none. Ending stops subsequent model calls here; this is distinct from the Figure 10 continuously steered profile ending row.

## Portable layout and provenance

Customer experiment source and its narrow customer runtime dependencies are isolated here. The source release is based on author commit `8d1649c03a63a39c9aa092532c376800cc4a3863` and released runtime `ae6e35c7a85905d586897c2a956bd651dce5aedd`. `provenance.json` records each original relative path/hash and installed hash. Only telemetry/path/entrypoint glue changes; scientific functions retain their original estimator, parser, sampler and settings. Do not combine runtime copies across studies. `LICENSE` is upstream MIT; frozen data retain their source terms (PopQA for B10; author scenario/conversation inputs otherwise).

Base `Qwen/Qwen2.5-32B-Instruct`, revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`; unmerged adapter/tokenizer `Valen92/pain-adapters`, revision `b64bd64b4bc7ca6e0733a489b8372a099d55ef05`. BF16 base, FP32 adapter, SDPA; injection layer 38, monitoring layer 61. Actual historical runtime pins are in requirements.txt: torch 2.11.0 (CUDA build used cu130), transformers 5.12.1, peft 0.20.0, numpy 2.3.4. No shared environment is changed.

## Commands

Run from this directory in your own Python environment:

```sh
python -m pip install -r requirements-analysis.txt
python reproduce.py --help
python reproduce.py --data-dir inputs --verify-only
python inference.py --dry-run
# Full original analysis on suitably resourced CPU compute, no model:
python reproduce.py --data-dir /path/to/local/data --output /path/to/fresh-analysis
# Optional future inference on suitable GPU hardware, NOT performed for this export:
python -m pip install -r requirements.txt
python inference.py --data-dir /path/to/local/assets --output /path/to/fresh-run
```

`--verify-only` checks all packaged file identities and deterministic original-code checks, without a bootstrap. Full replay calls original scientific analysis rather than merely copying tables. Full resampling and model inference were not run during packaging. Import/configuration smoke checks do not establish GPU runtime compatibility.

For full analysis --data-dir must contain B5/trials.jsonl and B7/trials.jsonl. These raw trajectories are external pending assets, not included. --verify-only uses included frozen inputs and corrected tables and needs no raw data.

Inference assets go in `vectors/` (32B pain safetensors/JSON, matched sadness .pt, verification.json and original_tensor_hashes_v2.json); R1/R2 also requires `fear/` (canonical 32B fear .pt and verification_fear.json). B10 also requires `pain-axis-b1-b3.bundle`. Base/adapter weights use pinned public model repositories and are not bundled.

## Frozen results and uncertainty

results/analysis_v2/{rates,contrasts}.json are authoritative. results/original_analysis_v1 contains old values solely to replay the exact metadata correction. Constant bootstrap quantiles are preserved but inferential CI bounds are null, explicitly uninformative. Constant-outcome projection fits remain unavailable.

10,000 draws, seed 42, initial contexts clustered within category; paired B7 conditions and all assignments retained. Literal first choice, malformed answers in all-assigned denominators.

No costly-relief association can be inferred from a constant outcome. B7 pain-minus-sadness does not separate when ending is initially second.

## Publication status

`assets.json` records public URLs as null/pending, never guessed. Required raw/vector assets are not public from this contribution yet. Explicit local --data-dir access is supported; publication is incomplete. The 72B counterparts were not run within these studies; do not fabricate comparator results. No new generations, judging, model scoring rules, or statistical choices were introduced by this export.
