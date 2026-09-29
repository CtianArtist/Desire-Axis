"""Versioned, inspectable saved-trial analysis outputs; standard-library only."""
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            invalid(text)
        return value
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON field: {key}")
            out[key] = value
        return out
    return json.loads(text, parse_constant=invalid, parse_float=finite_float, object_pairs_hook=unique)


def write_analysis(result, output, *, sources=None):
    """Do not overwrite earlier evidence; the caller controls the source identity."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output must be new or empty; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, content in result.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(content, allow_nan=False, separators=(",", ":")) + "\n")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if name in ("rates", "contrasts", "matched_histories", "seed_comparison") and content:
            fields = list(dict.fromkeys(k for row in content for k in row))
            path = output / f"{name}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in content:
                    writer.writerow({k: json.dumps(v, separators=(",", ":"), allow_nan=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    receipt = {"schema_version": 1, "data_kind": result["summary"]["data_kind"], "sources": sources or {}, "outputs": files}
    (output / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
