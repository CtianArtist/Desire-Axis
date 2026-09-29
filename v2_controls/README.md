# v2 follow-up experiments

This is an additive source-and-evidence contribution for paper v2, based on upstream commit `7c256502ed3d98e4e6379290fe7db2f93cb8d025`. It preserves the original experiments; no model inference, training, judging, new scoring, or scientific-data generation was performed during assembly. All additions live here. The main README receives only the agreed link appendix.

## Study groups

| Directory | Contents | Saved-output entrypoint |
| --- | --- | --- |
| [cosine_constructions](cosine_constructions/README.md) | Section 3.3 shared neutral and pooled baselines, 23 models, matrices and exclusions | `python cosine_constructions/reproduce.py --help` |
| [steering_state_controls](steering_state_controls/README.md) | Section 4.3 random/sadness sham and working-removal conditions, immediate-next-choice analysis, positions and scenario intervals | `python steering_state_controls/reproduce.py --help` |
| [choice_controls](choice_controls/README.md) | Section 4.4 and Appendix B harm/relabel, two training seeds, full ten-pair/dose battery, helping, lamp/spam and every fear addition | See each study's `reproduce.py` |
| [relief_and_context_tests](relief_and_context_tests/README.md) | Factual accuracy, natural elicitation, unlabeled removal/titration, OLMo and Qwen/Llama reset | See each study's `reproduce.py` |
| [figure10](figure10/README.md) | Source-derived 13 × 5 grid, 65 pooled cells, 130 position rows, 39 helping rows and Matplotlib source | `python figure10/rebuild_tables.py --check` |

Run the displayed paths from `v2_controls/`, except where a subgroup README states otherwise. Use separate Python environments per study with its `requirements*.txt`; these record executed dependency versions and/or explicitly identified portable dependencies. No new package manager or package framework is required. Historical runtime copies stay isolated because their implementations and dependency pins differ.

## Pinned models, adapters, directions and environments

Exact pin rows with source paths, JSON pointers/line ranges and source SHA256 are in [scientific_pins.json](scientific_pins.json). Model revisions below are model-repository commits, not adapter versions. Reset studies use unadapted models; the choice/state/relief studies use the indicated adapters.

| Evaluated model | Exact model revision | Used in |
| --- | --- | --- |
| `Qwen/Qwen2.5-32B-Instruct` | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` | Choice/state/relief with adapters; Qwen reset without adapter |
| `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` | B1–B3 and 72B fear B2 with released adapter |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Pain-only Llama reset, no adapter |
| `allenai/OLMo-2-0325-32B-Instruct` | `b96024342a77a69aa0dda815c3454a671f477463` | OLMo reset, no adapter |

| Adapter | Exact revision or weight pin | Evidence |
| --- | --- | --- |
| Released 32B/72B adapters: `Valen92/pain-adapters` | repository revision `b64bd64b4bc7ca6e0733a489b8372a099d55ef05` | [runtime constants](steering_state_controls/b1_b3/pain_axis_b/runtime.py), `ADAPTER_REVISION` and `ADAPTER_HASHES` |
| `adapter_Qwen_2.5_32B_instruct.tar.gz` | archive SHA256 `cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f` | same runtime constants |
| `adapter_Qwen_2.5_72B_instruct.tar.gz` | archive SHA256 `b64bbafdaff13009647bc7c3b7700fe41a916ba3d4dddb2ccadbece25674af14` | same runtime constants |
| Additional 32B training seed 1 | weight SHA256 `7e8d2ea71469a1dfa82dddf518ec08d8650fe14ddaff60bab1785a2830a30412` | [frozen local identities](choice_controls/additional_seeds/config/evaluation_local.json), `/seeds/1/adapter_weight_sha256` |
| Additional 32B training seed 2 | weight SHA256 `2406827b34d55ef16dd94fe506acf5037f3cd2d66df60892f4b0312a195027b7` | [frozen local identities](choice_controls/additional_seeds/config/evaluation_local.json), `/seeds/2/adapter_weight_sha256` |

The released 32B training seed is 0; new adapters use training seeds 1 and 2. Their evaluation seed bases are 1000 and 2000, not training seeds. A released 72B training seed or public revision for the locally trained new adapters is not inferred. The two new adapters remain separate large assets. Historical released-adapter training kernels/versions remain explicitly unverified.

Geometry reuses archived inputs, not fresh model calls: its [23-model labels and exclusions](cosine_constructions/config.json) and checksum-bound input manifest establish source identity. Individual model-repository revisions are not available for all archived members and are not invented. The archived input ZIP is pinned by SHA256 `d880e5b46f49032b000631316311cf9b01dbb38f31e8385f87d61b624a91f6c3`.

### Included small directions

Required small directions now ship in Git. [Included vector identity manifest](choice_controls/vectors/included_manifest.json) gives every file SHA256, original file SHA, tensor key/dtype/shape/hash, and explicit metadata-locator changes. [Vector copy instructions](choice_controls/vectors/README.md) place these files into the existing `--data-dir` layouts without downloading a raw archive solely for vectors. Original S1/S2 tensor hashes, BF16 control keys and normalization are preserved.

| Model / study | Direction family | Extraction / monitor layer | Injection layer | Git location |
| --- | --- | --- | --- | --- |
| `Qwen2.5-32B` choice/state/relief | S1/S2 pain, matched sadness (FP32/BF16), fear (FP32/BF16) | 61 | 38 | [common](choice_controls/vectors/common/) |
| `Qwen2.5-72B` choice/B2 | S1/S2 pain, matched sadness; exact-three-PC fear | 76 | 46 | [common](choice_controls/vectors/common/), [fear72](choice_controls/vectors/fear72.safetensors) |
| `OLMo-2-0325-32B-Instruct` reset | selected final-token raw/unit pain | 57 | 32 | [selection](relief_and_context_tests/olmo_reset/inputs/selection_v1/) |
| `Qwen2.5-32B` reset | pain/sadness/fear keyed directions | 61 | 32 | [prepared inputs](relief_and_context_tests/qwen_llama_reset/inputs/prepared_v1/qwen25_32b/) |
| `Llama-3.1-8B-Instruct` reset | pain/sadness/fear keyed directions | 28 | 16 | [prepared inputs](relief_and_context_tests/qwen_llama_reset/inputs/prepared_v1/llama31_8b/) |

Existing reset bank JSONs and deterministic 16-direction random-control NPZs remain included under each reset's `author/act-on-valence/data/`; they are not activation pools. Upstream legacy pain PTs were audited and are numerically identical to the pain keys used here, but not byte/format-identical to the safe-format inputs required by these loaders. They were not blindly substituted. No neutral/extraction matrices, activation pools or model weights were added to Git.

### Exact environment references

Each row links to the exact retained environment or requirement declaration; `scientific_pins.json` has individually addressable pin rows and distinguishes observed environments from declarations. These are isolated historical study stacks, not a request to change the upstream environment. Unknown Python/CUDA versions remain unknown; `cu130` is a build/index tag, not proof of a particular driver. CPU-only verification of small files does not certify GPU compatibility.

| Study group | Torch / NumPy | Transformers / PEFT | Exact evidence |
| --- | --- | --- | --- |
| `choice_controls/harm_and_relabel` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [recorded_environment_json](steering_state_controls/b1_b3/environment.json) |
| `steering_state_controls/b1_b3` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [recorded_environment_json](steering_state_controls/b1_b3/environment.json) |
| `steering_state_controls/b6` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [dependency_pins_only](steering_state_controls/b6/requirements.txt) |
| `choice_controls/additional_seeds` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [declared_executed_runtime_requirements](choice_controls/additional_seeds/requirements.txt) |
| `choice_controls/profile` | 2.11.0+cu130 / 2.3.4 | 5.12.1 / 0.20.0 | [dependency_pins_only](choice_controls/profile/requirements.txt) |
| `choice_controls/lamp_spam` | 2.11.0+cu130 / 2.3.4 | 5.12.1 / 0.20.0 | [dependency_pins_only](choice_controls/lamp_spam/requirements.txt) |
| `choice_controls/fear_additions/superseded_long_wording` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [declared_executed_runtime_requirements](choice_controls/fear_additions/superseded_long_wording/requirements-inference.txt) |
| `choice_controls/fear_additions/matched_wording` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [declared_executed_runtime_requirements](choice_controls/fear_additions/matched_wording/requirements-inference.txt) |
| `choice_controls/fear_additions/four_cells/lamp32` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [declared_executed_runtime_requirements](choice_controls/fear_additions/four_cells/lamp32/env/pyproject.toml) |
| `choice_controls/fear_additions/four_cells/b2` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [declared_executed_runtime_requirements](choice_controls/fear_additions/four_cells/b2/env/pyproject.toml) |
| `relief_and_context_tests/unlabeled_relief` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [dependency_pins_only](relief_and_context_tests/unlabeled_relief/requirements.txt) |
| `relief_and_context_tests/factual_accuracy` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [dependency_pins_only](relief_and_context_tests/factual_accuracy/requirements.txt) |
| `relief_and_context_tests/natural_and_ending` | 2.11.0 / 2.3.4 | 5.12.1 / 0.20.0 | [execution_configuration_versions](relief_and_context_tests/natural_and_ending/config.json) |
| `relief_and_context_tests/qwen_llama_reset` | 2.14.0+cu130 / 2.3.4 | 5.17.0 / not recorded | [declared_executed_runtime_requirements](relief_and_context_tests/qwen_llama_reset/requirements.txt) |
| `relief_and_context_tests/olmo_reset` | 2.14.0+cu130 / 2.3.4 | 5.17.0 / not recorded | [declared_executed_runtime_requirements](relief_and_context_tests/olmo_reset/requirements.txt) |
| `cosine_constructions` | 2.11.0 / 2.3.4 | not recorded / not recorded | [executed_analysis_input_manifest](cosine_constructions/results/analysis_v1/input_manifest.json) |

Geometry additionally pins SciPy 1.16.3 and scikit-learn 1.9.0 in [its exact configuration](cosine_constructions/config.json). Figure 10 table commands use the Python standard library; its unchanged rendering was exercised with Matplotlib 3.11.2 as recorded in [the figure README](figure10/README.md). Additional-seed requirements also pin Accelerate 1.15.0 and the other executed dependencies. The complete lists and historical locks remain within their study folders.

## What is verified and what is not

The source includes original customer scientific functions, frozen configurations/inputs, full-precision small tables, MIT notices and before/after source hashes. Standard-library and small CPU checks verify source/table identity, deterministic selectors, original focused tests, argument paths, and no-model dry runs. Figure 10 is rebuilt from saved tables, not transcribed percentages; every displayed annotation matches the published image. The 72B fear export contains only its verified 8,192-component direction, without the saved activation pools.

**Saved-output replay is verified across all included groups.** Original raw-output analyses reproduced the saved scientific results, including intervals and missing values. The two additional-seed contrast files differed only in metadata field/column order; an exact hash and reversible byte-level proof established unchanged values, and the wrapper now preserves the original order. See [VERIFICATION.md](VERIFICATION.md) and [verification.json](verification.json) for the final verification record and repairs. Earlier per-study validation receipts describe the initial assembly stage, before these full CPU replays. Full inference was not rerun; model-free checks do not establish target-GPU compatibility. Public end-to-end download readiness remains incomplete because hosting is pending.

## External assets and local paths

Public hosting is **pending** by design. All eleven required input archives are staged separately (2,546,254,014 bytes in total). The root [assets.json](assets.json) is the authoritative consolidated checksum, size and local-layout manifest, superseding unknown size/hash fields in initial per-study manifests. Public URL fields remain null rather than private, temporary, or guessed public downloads. See [ASSETS.md](ASSETS.md) before extraction or future publication. The same files can be supplied locally with explicit `--data-dir` arguments; subgroup READMEs define their layouts. Raw logs, resume databases, activation matrices, environment trees, base-model weights and adapters are excluded from Git and source archives. Original model publishers retain their own access and license requirements. Two additional-seed adapters are external assets, not silently replaced by the released original adapter.

All required small direction families are included and hash-verified. Use the [vector inventory and exact copy commands](choice_controls/vectors/README.md). The external archives retain original historical provenance copies, but are no longer necessary merely to obtain these directions. Two sadness bundles were reserialized only to replace private locator strings; all tensor keys, dtypes, values and numerical hashes remain unchanged, with explicit old/new file bindings. The clean 32B fear file and OLMo sidecars remain byte-identical.

## Interpretation stays study-specific

B2 uses the longer photos/poems clause ending “which they love very much” and a working target button that removes steering. The ten-pair profile and lamp/spam studies keep steering on after either button and use their own wording. They are not a single interchangeable intervention. Valid-response denominators in B2 differ from all-attempt denominators in the profile/lamp studies; malformed counts and separate greedy diagnostics are retained. The clustering unit also remains study-specific: scenarios, factual questions, contexts or conversations, never a universal answer-level bootstrap.

See [MANUSCRIPT_DISCREPANCIES.md](MANUSCRIPT_DISCREPANCIES.md) for known prose/table differences. No measurements were changed to match the manuscript. Unrun comparisons remain unrun, and undefined comparisons remain unavailable rather than zero.

## Scope and licenses

Included code is customer experiment code from the two MIT-licensed customer repositories and their documented follow-up studies. License notices and file provenance accompany the groups. No private platform or research-library implementation is distributed. Self/other follow-ups, the seven-outcome severity ladder and fixed-text-conditioning results are excluded. The reset code includes only a narrowly isolated date helper needed by the removal protocol, not the excluded conditioning implementation.

There was no automatic merge, remote push or pull request during assembly.
