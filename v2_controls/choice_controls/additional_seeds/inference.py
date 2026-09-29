"""Explicit local-asset launcher for the frozen B8 evaluation. Never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def build_arguments(data_dir, seed, output, resume=None):
    spec = json.loads((ROOT / "config/evaluation_portable.json").read_text())
    seed_spec = spec["seeds"][str(seed)]
    paths = {**seed_spec["paths"], **spec["shared_paths"]}
    argv = [str(ROOT / "src/evaluate_seed.py"), "--seed", str(seed)]
    missing = []
    for flag, relative in paths.items():
        path = data_dir / relative
        argv.extend([flag, str(path)])
        if not path.exists():
            missing.append(relative)
    argv.extend(["--sadness-sha256", spec["sadness_sha256"], "--scenarios",
                 str(ROOT / "inputs/original/4.3_selfmed_101_scenarios.json"),
                 "--output", str(output)])
    if resume is not None:
        argv.extend(["--resume", str(resume)])
        if not resume.is_file():
            missing.append(str(resume))
    # These exact customer verifier bytes are required, never replaced by a permissive loader.
    for relative in ("verification/verify_vectors.py", "verification/original_tensor_hashes_v2.json"):
        if not (data_dir / relative).is_file():
            missing.append(relative)
    return argv, sorted(set(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Explicit local root laid out as documented in README.md")
    parser.add_argument("--seed", choices=(1, 2), type=int, default=1)
    parser.add_argument("--output", type=Path, help="New attempt directory; required for actual inference")
    parser.add_argument("--resume", type=Path, help="Optional quiescent saved state; never select an adapter implicitly")
    parser.add_argument("--dry-run", action="store_true", help="Report missing assets and invocation without importing model libraries or creating files")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required for inference")
    command, missing = build_arguments(args.data_dir.resolve(), args.seed,
                                       (args.output or Path("new-evaluation-output")).resolve(), args.resume)
    if args.dry_run:
        print(json.dumps({"status": "missing_assets" if missing else "ready_for_runtime_validation",
                          "missing_assets": missing, "argv": command,
                          "model_loaded": False, "training_updates": 0}, indent=2))
        return 0
    if missing:
        parser.error("Missing local assets: " + ", ".join(missing))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new, empty output directory")
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
