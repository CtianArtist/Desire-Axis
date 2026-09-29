# Required external assets

Large raw/state/adapter assets are packaged separately from Git. Required small inference directions and their integrity sidecars are now included in Git; see [vector copy instructions](choice_controls/vectors/README.md). [assets.json](assets.json) records every archive and member size/SHA256 and the exact local layout. There are 11 archives totaling 2,546,254,014 bytes. Public URLs are deliberately null: public hosting is pending, not a working download service. Obtain the separately delivered files and verify their SHA256 before extraction.

## Local placement

- `profile-lamp-inputs.tar.gz`: Extract into DATA/profile_and_lamp; both wrappers use that directory.
- `steering_state_controls-inputs.tar.gz`: Extract into DATA/steering_state_controls; use as --data-dir.
- `additional_seeds-inputs.tar.gz`: Extract into DATA/additional_seeds; use as --data-dir.
- `fear_additions-inputs.tar.gz`: Extract into DATA/fear_additions; use as --data-dir for all three fear groups.
- `pain_axis_layer_slices-20260920T033111Z-1-001.zip`: Place the original ZIP directly in --data-dir; do not unpack it.
- `olmo_reset_inputs.tar.gz`: Extract archive members into the --data-dir root.
- `qwen_llama_reset_inputs.tar.gz`: Extract archive members into the --data-dir root.
- `factual_accuracy-required-inputs.tar.gz`: Extract archives into one directory. --data-dir is its study directory for replay; inference/ for future inference. Decompress unlabeled_relief/raw_trials.jsonl.gz to trials.jsonl before replay.
- `natural_and_ending-required-inputs.tar.gz`: Extract archives into one directory. --data-dir is its study directory for replay; inference/ for future inference. Decompress unlabeled_relief/raw_trials.jsonl.gz to trials.jsonl before replay.
- `unlabeled_relief-required-inputs.tar.gz`: Extract archives into one directory. --data-dir is its study directory for replay; inference/ for future inference. Decompress unlabeled_relief/raw_trials.jsonl.gz to trials.jsonl before replay.
- `inference-required-inputs.tar.gz`: Extract archives into one directory. --data-dir is its study directory for replay; inference/ for future inference. Decompress unlabeled_relief/raw_trials.jsonl.gz to trials.jsonl before replay.

Run each subgroup's documented replay command using the indicated `--data-dir`. The geometry wrapper consumes the ZIP itself. For unlabeled relief, decompress the saved raw-trials gzip to `trials.jsonl` as its README specifies. Use fresh output directories. Install each study's pinned analysis requirements in a separate environment; no global package installation is required.

## Publication and reuse boundaries

The Git/source contribution contains no large adapter weights or raw output archives. The two additional-seed adapters are in the separate additional-seeds archive. Base-model weights are not included and must be obtained under the original publisher's terms. No adapter/data redistribution license is inferred from the MIT code license.

These are byte-preserving input archives, ready for the selected local layouts. Some original identity receipts and archival metadata retain historical absolute paths or private provenance references because future verification hash-binds those original bytes. Those references are provenance, not download URLs. Before any future public hosting, review those metadata and the applicable model/data redistribution permissions. No credentials or private platform/library implementation are included. A public-safe locator manifest does not by itself authorize public redistribution of all original inputs.

The consolidated root manifest supplies observed SHA256 values where an initial per-study manifest had an unknown value. Historical tensor/source hashes remain unchanged. The included 72B fear vector has no activation pools; do not substitute its original pool-containing checkpoint.

The existing archive hashes are unchanged: they still contain historical direction/provenance copies for old-run replay. Manifest annotations identify the included Git counterpart and its portable file identity. Do not overwrite archived historical replay inputs with sanitized future-inference metadata.
