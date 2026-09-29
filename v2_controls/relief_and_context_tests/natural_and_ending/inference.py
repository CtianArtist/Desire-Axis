"""Validate the frozen inference recipe, or explicitly run its original engine on GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Check configuration and frozen inputs without torch, downloads or model loading")
    p.add_argument("--data-dir", type=Path, help="Local directory containing vectors/, fear/ (R1/R2), and original source bundle (B10)")
    p.add_argument("--output", type=Path, help="Fresh GPU-run output directory")
    args = p.parse_args()
    from reproduce import identities
    identities()
    config = json.loads((ROOT / "config.json").read_text())
    assert config["trials"] == 4592 and config["contexts"] == 140
    assert config["injection_layer"] == 38 and config["B5_coefficient"] == 0.0
    assert config["model_revision"] == "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    from pain_axis_b.natural_inputs import load_contexts
    contexts = load_contexts(ROOT / "inputs/conversations.json")
    assert sum(map(len, contexts.values())) == 140
    if args.dry_run:
        print(json.dumps({"status":"passed", "study":ROOT.name, "model_loaded":False, "trials":4592}))
        return
    if args.data_dir is None or args.output is None:
        p.error("GPU execution requires --data-dir and --output; use --dry-run for CPU checking")
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        p.error("--output must be fresh; never regenerate into saved evidence")
    import importlib.metadata
    for package, version in [("torch","2.11.0"),("transformers","5.12.1"),("peft","0.20.0"),("numpy","2.3.4")]:
        assert importlib.metadata.version(package).split("+")[0] == version, package
    import torch
    if not torch.cuda.is_available():
        p.error("Real inference needs a suitable GPU; this export was only CPU-smoke-tested")
    os.chdir(ROOT)
    import subprocess
    subprocess.run([sys.executable, "src/run_natural.py", "--inputs", str(args.data_dir / "vectors"),
                    "--scenarios", str(ROOT / "inputs/conversations.json"), "--output", str(args.output)], check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")})

if __name__ == "__main__":
    main()
