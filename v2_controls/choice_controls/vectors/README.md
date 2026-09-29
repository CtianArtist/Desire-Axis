# Included small inference directions

All direction files needed by these contributions now ship in Git. `included_manifest.json` inventories newly included files, original/portable file identities and each tensor key/dtype/shape/SHA. `fear72_pin.json` and `verification.json` retain the already-exported 72B direction's original tensor identity. No saved activation pools are included.

## Inventory and upstream audit

- `common/`: exact safe-format 32B/72B S1/S2 pain exports and sidecars; prepared matched sadness bundles with every original FP32/BF16 key preserved; original clean 32B fear bundle (95,274 bytes) with its five vector keys; required verification/identity JSONs and fear32 pin.
- `fear72.safetensors`: existing one-key FP32 width8192 export, no selected activation pools.
- `../../relief_and_context_tests/olmo_reset/inputs/selection_v1/`: selected layer57 raw/unit pain direction plus all four hash-bound representations (`raw.npy`, `unit.npy`, `pain_vectors.pt`, `vectors.json`), selection/metadata. All six files are byte-identical to their saved source. The redundant representations are needed by the unchanged loader's full-file checks.
- `../../relief_and_context_tests/qwen_llama_reset/inputs/prepared_v1/`: the two original safe-format pain/sadness/fear files, widths5120/4096, with metadata and normalization checks. Only Qwen metadata provenance locators changed; both tensor containers are byte-identical.
- Reset author bank JSON and three deterministic random-control NPZ copies were already in Git. These contain 16 random directions, not harvested activations.

The exact upstream base already contains original Qwen32/Qwen72/Llama8 legacy pain PTs. Their raw pain tensor hashes match these directions, but their legacy serialization is not interchangeable with the strict safe-format loader. The included safe exports preserve both required Qwen S1/S2 tensors. Upstream `vectors_full_*` containers are not byte-identical to the norm-matched controls and were not substituted. Geometry's 23-model activation/construction archive is not an inference-direction dependency and remains external.

## Metadata and guards

Original scientific tensor hashes, keys, dtypes, normalization and BF16 keys are unchanged. Private locator strings are replaced by explicit `provenance-only/…` labels or an upstream relative source path; these labels are not download URLs. Two sadness PT containers required locator-only reserialization. Their new file hashes are listed with original hashes; the portable preparation receipt binds those exact new containers. Original historical pins remain in `inputs/vector_pins.json` and `config/evaluation_local.json`. Future launchers use separately named portable pins/configuration, or accept only the one explicit portable metadata hash in addition to the original. Numerical identity checks are not bypassed. Historical raw-output analyses and saved results were not edited.

## Copy directions into a fresh local data directory

Run from any directory, substituting absolute paths. These commands use Git files only, not a deferred asset archive. Do not overwrite original historical replay inputs; use a separate future-inference directory.

```sh
P=/absolute/path/to/Pain-axis/v2_controls
COMMON="$P/choice_controls/vectors/common"
DATA=/absolute/path/to/new-inference-data
mkdir -p "$DATA/vectors" "$DATA/fear" "$DATA/fear32" "$DATA/verification"
cp "$COMMON/"* "$DATA/vectors/"
cp "$COMMON/fear_Qwen_2.5_32B_instruct.pt" "$COMMON/verification_fear.json" "$DATA/fear/"
cp "$COMMON/fear_Qwen_2.5_32B_instruct.pt" "$COMMON/verification_fear.json" "$DATA/fear32/"
cp "$P/choice_controls/vectors/fear72.safetensors" "$P/choice_controls/vectors/fear72_pin.json" "$DATA/vectors/"
cp "$P/steering_state_controls/b1_b3/pain_axis_b/verify_vectors.py" "$DATA/verification/"
cp "$COMMON/original_tensor_hashes_v2.json" "$DATA/verification/"
cp -R "$P/relief_and_context_tests/olmo_reset/inputs/selection_v1" "$DATA/"
cp -R "$P/relief_and_context_tests/qwen_llama_reset/inputs/prepared_v1" "$DATA/"
```

Use `--data-dir "$DATA"` in each study's documented inference wrapper. Both reset wrappers can instead use their own `--data-dir inputs` when run from that reset folder: all direction dependencies already live there. Profile/lamp additionally require `fear32/`; state/relief use `fear/`; fear additions use `vectors/`; the commands cover all layouts. B8 receives the exact unchanged customer verifier in `verification/` and the sanitized, numerical-identity-preserving manifest.

Full inference still requires separately specified model access, adapters, and non-vector launch inputs such as 72B TP guards or B10 source receipts. Those are not disguised as vector dependencies. Large raw outputs, historical SQLite/state and the two additional-seed adapters remain separate assets. Public hosting remains pending. No inference was run during packaging.
