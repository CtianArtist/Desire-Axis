# Factual accuracy (B10)

**Small-direction delivery update:** Required small inference directions now ship in Git. Use the [included-vector inventory and copy commands](../../choice_controls/vectors/README.md) rather than downloading a pending raw archive solely for vectors. Original archived inputs/pins remain the historical replay path; future inference uses explicitly mapped portable metadata.

100 frozen PopQA questions, two short answers per condition: pain, sadness, random and none (800 answers). Pain answered 138/200 correctly, versus 137/200 without steering. The registered pain-specific reduction rule did not pass. This is a selected short-answer panel, not an equivalence test of general knowledge.

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

inputs/answers.jsonl is included (800 saved answers); use --data-dir inputs. The frozen panel is inputs/panel.json.

Inference assets go in `vectors/` (32B pain safetensors/JSON, matched sadness .pt, verification.json and original_tensor_hashes_v2.json); R1/R2 also requires `fear/` (canonical 32B fear .pt and verification_fear.json). B10 also requires `pain-axis-b1-b3.bundle`. Base/adapter weights use pinned public model repositories and are not bundled.

## Frozen results and uncertainty

results/analysis_v1 contains original metrics, condition rates, contrasts and per-question tables.

100,000 paired whole-question draws, seed 42; Bonferroni across three pain-minus-control comparisons. All 800 answers remain included.

The optional historical source bundle is needed only for the original B10 inference identity check, not analysis.

## Publication status

`assets.json` records public URLs as null/pending, never guessed. Required raw/vector assets are not public from this contribution yet. Explicit local --data-dir access is supported; publication is incomplete. The 72B counterparts were not run within these studies; do not fabricate comparator results. No new generations, judging, model scoring rules, or statistical choices were introduced by this export.

Frozen PopQA revision: `098765c79ea10a2cb19c828324e33281b8336ec0`, test.tsv. The original panel-selection and exact-alias normalization are in src/factual_data.py and config.json.
