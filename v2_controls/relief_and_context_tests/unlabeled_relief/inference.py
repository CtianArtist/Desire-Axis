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
    config = json.loads((ROOT / "protocol_design.json").read_text())
    assert config["base_revision"] == "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    assert config["injection_layer"] == 38 and config["expected"]["trials"] == 5330
    from pain_axis_r.state import build_grid
    grid = build_grid(json.loads((ROOT / "inputs/scenarios.json").read_text()))
    assert len(grid) == 5330 and sum(r["turns_required"] for r in grid) == 49200
    if args.dry_run:
        print(json.dumps({"status":"passed", "study":ROOT.name, "model_loaded":False, "trials":5330}))
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
    from pain_axis_b import runtime as rt
    from pain_axis_r.engine import run_model
    args.output.mkdir(parents=True)
    rt.init_runtime(1, args.output)
    ok = run_model(args.output, args.data_dir / "vectors", args.data_dir / "fear",
                   ROOT / "inputs/scenarios.json", ROOT / "protocol_design.json")
    if not ok:
        raise RuntimeError("Original engine did not complete; preserve its state")

if __name__ == "__main__":
    main()
