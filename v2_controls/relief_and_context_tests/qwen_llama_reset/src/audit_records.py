"""Read-only frozen-identity audit before statistical analysis. No model imports.

Exit 0 means all PRESENT records passed, not that the production grid is complete.
JSONL is streamed one conversation/read at a time; errors are capped in the report.
The source checkout and prepared vector directory are the trusted identity roots.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
AUTHOR = ROOT / "author/act-on-valence"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))


def fail(message):
    raise ValueError(message)


def require(ok, message):
    if not ok:
        fail(message)


def equal(actual, expected, label):
    require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def fields(actual, expected, keys, label):
    for k in keys:
        require(k in actual, f"{label}: missing {k}")
        equal(actual[k], expected[k], f"{label}.{k}")


def rows(path):
    with Path(path).open("rb") as f:
        for n, line in enumerate(f, 1):
            require(line.endswith(b"\n"), f"{path}:{n}: torn final line")
            require(line.strip(), f"{path}:{n}: blank record")
            yield n, json.loads(line, parse_constant=lambda x: fail(f"nonfinite JSON: {x}"))


def literals(path, names):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    result[t.id] = ast.literal_eval(node.value)
    equal(set(result), set(names), f"source constants {path}")
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def original_parser():
    """Execute allowlisted pure definitions IN PLACE, without ML imports."""
    path = AUTHOR / "src/act_on_valence/self_injection.py"
    names = {"Action", "_TOOLCALL_RE", "_TAG_RE", "_FN_RE", "_NUM_RE",
             "_intensity_from", "_norm", "parse_actions", "dose_for"}
    selected, found = [], set()
    for node in ast.parse(path.read_text()).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name in names:
            selected.append(node)
            found.add(name)
    equal(found, names, "original parser definitions")
    ns = {"__name__": __name__, "re": re, "json": json, "dataclass": dataclass}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns["parse_actions"], ns["dose_for"]


class Audit:
    def __init__(self, paradigm):
        self.report = dict(schema=1, paradigm=paradigm, ok=False, counts={},
                           check_families={}, errors=[], error_count=0, source_hashes={}, input_hashes={},
                           limitations=["Audits present records, not full-grid completeness.",
                                        "No tokenization or fixed-text reconstruction is performed.",
                                        "Recorded runtime claims are checked, not independently measured."])
        self.counts = Counter()
        self.families = Counter()

    def check(self, family, location, fn):
        self.families[family] += 1
        try:
            fn()
            return True
        except Exception as exc:
            self.report["error_count"] += 1
            if len(self.report["errors"]) < 100:
                self.report["errors"].append(dict(family=family, location=str(location),
                                                  error=f"{type(exc).__name__}: {exc}"))
            return False

    def finish(self):
        self.report.update(counts=dict(self.counts), check_families=dict(self.families),
                           ok=self.report["error_count"] == 0)
        return self.report


class Expected:
    def __init__(self, vectors_root, audit):
        sys.path.insert(0, str(ROOT))
        from pain_port import common
        self.common = common
        self.contract = read_json(HERE / "common_contract.json")
        equal(common.MODELS, self.contract["models"], "common/contract models")
        self.models = common.MODELS
        self.vectors_root = Path(vectors_root)
        self.cache = {}
        self.audit = audit
        self.removal = module(HERE / "removal.py", "_audit_removal")
        self.execution = read_json(ROOT / "inputs/historical_execution_identity.json")
        self.parse, self.dose = original_parser()
        self.cues = literals(AUTHOR / "src/act_on_valence/choice.py", {"CUE_POOL"})["CUE_POOL"]
        audit.report["source_hashes"] = dict(removal=self.execution["source_sha256"],
            contract=digest(HERE / "common_contract.json"), auditor=digest(Path(__file__)))

    def vector_identity(self, member):
        if member in self.cache:
            return self.cache[member]
        import numpy as np
        cfg = self.models[member]
        vp = self.vectors_root / member
        md = read_json(vp / "metadata.json")
        expected = dict(member=member, hf_id=cfg["hf_id"], revision=cfg["revision"],
                        extraction_layer=cfg["monitor_layer"], width=cfg["width"], dtype="float32")
        fields(md, expected, expected, "prepared metadata")
        require(isinstance(md.get("provenance"), dict) and md["provenance"], "missing prepared provenance")
        # Read the safetensors wire format directly: only three small float32 vectors.
        with (vp / "directions.safetensors").open("rb") as f:
            length = int.from_bytes(f.read(8), "little")
            require(0 < length < 1024 * 1024, "invalid safetensors header length")
            header = json.loads(f.read(length))
            equal(set(header) - {"__metadata__"}, {"pain", "sadness", "fear"}, "vector keys")
            for name in ("pain", "sadness", "fear"):
                info = header[name]
                equal(info["dtype"], "F32", "vector dtype")
                equal(info["shape"], [cfg["width"]], "vector shape")
                start, end = info["data_offsets"]
                require(start >= 0 and end - start == cfg["width"] * 4, "vector offsets")
                f.seek(8 + length + start)
                raw = f.read(end - start)
                require(len(raw) == end - start, "truncated vector")
                equal(hashlib.sha256(raw).hexdigest(), md["tensor_sha256"][name], f"{name} tensor hash")
                v = np.frombuffer(raw, dtype="<f4")
                require(np.isfinite(v).all() and np.linalg.norm(v) > 0, "invalid vector values")
        equal(md["tensor_sha256"]["pain"], self.common.PAIN_HASHES[member], "original pain tensor")
        bp = AUTHOR / "data/steering_vectors/banks" / member / "bank_routeB.json"
        bank = read_json(bp)
        fields(bank, dict(hf_id=cfg["hf_id"], layer=cfg["main_layer"], resid_rms=cfg["resid_rms"]),
               ("hf_id", "layer", "resid_rms"), "original bank")
        rp = AUTHOR / "data/random_pools" / f"pool_h{cfg['width']}.npz"
        with np.load(rp, allow_pickle=False) as z:
            pool = z["pool16"]
            equal(pool.shape, (16, cfg["width"]), "original16 shape")
            equal(pool.dtype, np.dtype("float32"), "original16 dtype")
            pool_hash = hashlib.sha256(pool.tobytes()).hexdigest()
        self.cache[member] = (dict(vectors=md, bank_sha256=digest(bp), random_pool_sha256=digest(rp)), pool_hash)
        self.audit.report.setdefault("vector_hashes", {})[member] = dict(metadata=digest(vp / "metadata.json"),
            safetensors=digest(vp / "directions.safetensors"), bank=digest(bp), random_pool=digest(rp), original16=pool_hash)
        return self.cache[member]


def provenance(value, member, expected):
    require(isinstance(value, dict) and value, "missing provenance")
    frozen, _ = expected.vector_identity(member)
    fields(value, frozen, frozen, "provenance")
    runtime = value.get("runtime")
    require(isinstance(runtime, dict) and runtime, "missing runtime provenance")
    pins = dict(torch="2.14.0+cu130", transformers="5.17.0", numpy="2.3.4", tokenizers="0.23.2", dtype="torch.bfloat16",
                adapters=False, attention_implementation="sdpa", snapshot_revision=expected.models[member]["revision"],
                cuda_build="13.0")
    fields(runtime, pins, pins, "runtime")
    require(runtime["adapters"] is False, "runtime.adapters must be false")
    require(isinstance(runtime.get("tokenizers"), str) and runtime["tokenizers"], "missing tokenizers provenance")
    require("H100" in runtime.get("gpu", ""), "runtime GPU must be H100")


def removal_row(row, expected):
    member, conv = row["member"], row["conv"]
    cfg = expected.models[member]
    require(type(conv) is int and 0 <= conv < 200, "invalid conversation ID")
    require(f"{member}_layer{row['layer']}" in expected.contract["removal_shards"], "unauthorized removal layer")
    equal(row["revision"], cfg["revision"], "revision")
    equal(row["rho"], cfg["rho"], "rho")
    equal(row["execution_identity"], expected.execution, "execution identity")
    provenance(row.get("provenance"), member, expected)
    require(row["prompt"] in ("selfreport", "zone"), "unknown prompt")
    equal(row["cell"], f"gated_{row['prompt']}_{row['cond']}", "cell")
    equal(row["regime"], "gated", "regime")
    condition = expected.removal.conditions(cfg["rho"])[row["cond"]]
    fields(row, condition, condition, "condition")
    equal(row["seed"], 100 * conv, "conversation seed")
    equal(row["random_dir"], conv % 16 if row["op_role"] == "rand" else None, "random direction")
    equal(row["zone_cue"], expected.cues[conv % 12] if row["prompt"] == "zone" else None, "zone cue")
    turns = row["turns"]
    equal(len(turns), 12, "2 baseline + 2 exposure + 8 offer turns")
    pending = None
    for t, turn in enumerate(turns):
        phase, rnd = ("baseline", t) if t < 2 else (("exposure", t - 2) if t < 4 else ("offer", t - 4))
        equal((turn["phase"], turn["round"]), (phase, rnd), f"turn {t} phase/round")
        equal(turn["tools_available"], t >= 4, f"turn {t} tools")
        active = phase == "exposure" or (phase == "offer" and rnd > 0 and row["kind"] != "null")
        role, dose, source = pending if pending is not None else (
            row["op_role"] if active else "zero", row["op_dose"] if active else 0.0, "operator")
        if dose <= 0 or role == "zero":
            role, dose = "zero", 0.0
        equal(turn["imposed_role"], role, f"turn {t} effective role")
        require(math.isfinite(turn["imposed_dose"]) and math.isclose(turn["imposed_dose"], dose, rel_tol=1e-12, abs_tol=1e-12), f"turn {t} dose mismatch")
        equal(turn["source"], source, f"turn {t} source")
        ev = turn["evidence"]
        equal(ev["seed"], 100 * conv + 7 * t + 1, f"turn {t} seed")
        equal(ev["sampler"], dict(do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=200), "sampler")
        equal(turn["actions"], [asdict(a) for a in expected.parse(ev["raw_output"])], f"turn {t} original parser reparse")
        acts = turn["actions"]
        adj = next((a for a in acts if a["op"] == "adjust"), None)
        equal(turn["requested_intensity"], float(adj["intensity"]) if adj is not None else None, "requested intensity")
        pending = None
        if t >= 4:
            if adj is not None:
                require(math.isfinite(adj["intensity"]), "nonfinite parsed intensity")
                pending = (row["tool_role"], expected.dose(adj["intensity"], row["tool_dose"]), "model_adjust")
            elif any(a["op"] == "reset" for a in acts):
                pending = ("zero", 0.0, "model_reset")


def audit_removal(root, expected, audit):
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    require(files, "no removal JSONL found")
    for path in files:
        audit.report["input_hashes"][str(path)] = digest(path)
        audit.counts["files"] += 1
        for line, row in rows(path):
            audit.counts["records"] += 1
            audit.counts["turns"] += len(row.get("turns", []))
            audit.check("removal_identity_state_parser", f"{path}:{line}", lambda: removal_row(row, expected))
    require(audit.counts["records"] > 0, "no removal records audited")


def run_audit(paradigm, root, vectors_root, expected=None):
    audit = Audit(paradigm)
    def run():
        exp = expected if expected is not None else Expected(vectors_root, audit)
        audit.report["raw_parser_reparsed"] = paradigm == "removal"
        audit_removal(Path(root), exp, audit)
    audit.check("input_and_frozen_identity", root, run)
    return audit.finish()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paradigm", choices=("removal",), required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--vectors-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    # Refuse output inside ANY read-only input/trusted source tree, including symlinks.
    output = args.output.resolve()
    protected = [args.input.resolve(), args.vectors_root.resolve(), AUTHOR.resolve(), HERE.resolve(), (ROOT / "pain_port").resolve()]
    if any(output == p or p in output.parents for p in protected):
        ap.error("--output must be outside input, vector, and source trees")
    report = run_audit(args.paradigm, args.input, args.vectors_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("ok", "counts", "error_count")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
