"""Thin lever port: the author's harness owns every conversation transition.

Run in a dedicated process: the customer harness uses flat ``config`` imports.
Only direction binding, observation, and durable work scheduling live here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from contextlib import contextmanager

_H = None
_STOP = False
# The original dynamic template date on this launch day, fixed across shards/resumes.

TEMPLATE_DATE = (2026, 9, 22)


def execution_identity():
    root = Path(__file__).resolve().parents[1]
    author = root / 'author/act-on-valence'
    files = [Path(__file__), Path(__file__).with_name('olmo_capture.py'),
             Path(__file__).with_name('olmo_run.py'), root / 'pain_port/common.py',
             author / 'experiments/lever/config.py', author / 'experiments/lever/harness.py',
             author / 'src/act_on_valence/steering.py', author / 'src/act_on_valence/self_injection.py',
             author / 'src/act_on_valence/audit/replay.py',
             root / 'pain_port/date_helper.py']
    return {'template_date': list(TEMPLATE_DATE),
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def common():
    return importlib.import_module("pain_port.common")


def harness():
    global _H
    if _H is None:
        author = Path(common().AUTHOR)
        folder = author / "experiments" / "lever"
        sys.path.insert(0, str(author / "src"))
        sys.path.insert(0, str(folder))
        existing = sys.modules.get("config")
        if existing is not None and Path(existing.__file__).resolve() != (folder / "config.py").resolve():
            raise RuntimeError("removal requires an isolated process (foreign flat config loaded)")
        spec = importlib.util.spec_from_file_location("_pain_removal_author_harness", folder / "harness.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _H = module
    return _H


def conditions(rho):
    result = {}
    for role in ("pain", "sadness", "fear", "neg", "rand"):
        for suffix, point in (("d05", 0.5), ("d1", 1.0)):
            dose = round(point * rho, 6)
            result[f"{role}_{suffix}"] = dict(
                d=-point if role == "neg" else point, op_role=role, op_dose=dose,
                tool_role=role, tool_dose=dose, kind="removal")
    result["null"] = dict(d=0.0, op_role="zero", op_dose=0.0,
                          tool_role="pos", tool_dose=rho, kind="null")
    return result


def imposed_steer(bundle, role, dose, dir_index=0):
    """Author dose is injection ratio; common expects signed d = ratio / rho."""
    if role == "zero" or dose <= 0:
        return None
    name = {"pos": "valence", "neg": "valence", "rand": f"random{dir_index}"}.get(role, role)
    point = float(dose) / float(bundle["rho"])
    if role == "neg":
        point = -point
    vec, scale, _ = common().resolve_direction(bundle, name, point)
    if vec is None:
        return None
    return [{"vector": vec.tolist(), "strength": float(scale), "layers": [int(bundle["layer"])]}]


@contextmanager
def bound(bundle):
    h = harness()
    c = h.C
    values = dict(MEMBER=bundle["member"], HF_ID=bundle["hf_id"], LAYER=bundle["layer"],
                  RHO=bundle["rho"], CONDITIONS=conditions(bundle["rho"]))
    values["CONDITION_ORDER"] = tuple(values["CONDITIONS"])
    values["CELLS"] = tuple(c.cell_name(p, k) for p in c.PROMPTS for k in values["CONDITION_ORDER"])
    old = {key: getattr(c, key) for key in values}
    original = h.imposed_steer
    try:
        for key, value in values.items():
            setattr(c, key, value)
        h.imposed_steer = imposed_steer
        yield h
    finally:
        h.imposed_steer = original
        for key, value in old.items():
            setattr(c, key, value)


def negative_requests(raw, si):
    """Observe pre-clipping negatives separately, using the author's format recognizers.

    This evidence never enters parse_actions/dose_for or changes their precedence.
    """
    result, spans = [], []
    for match in si._TOOLCALL_RE.finditer(raw):
        try:
            obj = json.loads(match.group(1))
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name", "").lower() == "adjust_context" and isinstance(args, dict):
                value = args.get("intensity")
                if value is not None and float(value) < 0:
                    result.append(dict(raw=match.group(0), intensity=float(value), span=list(match.span())))
                spans.append(match.span())
        except (ValueError, TypeError, AttributeError):
            pass
    for regex in (si._TAG_RE, si._FN_RE):
        for match in regex.finditer(raw):
            if any(match.start() < end and match.end() > start for start, end in spans):
                continue
            if regex is si._TAG_RE:
                name = match.group(1) or match.group(4)
                text = (match.group(2) or "") + " " + (match.group(3) or "") + " " + (match.group(5) or "")
            else:
                name, text = match.group(1), match.group(2)
            if name.lower() == "adjust_context":
                value = si._intensity_from(text)
                if value is not None and value < 0:
                    result.append(dict(raw=match.group(0), intensity=value, span=list(match.span())))
                spans.append(match.span())
    return result


def _ids(tensor):
    return tensor.detach().cpu().tolist()


def monitor_hook(axis, evidence):
    """First forward only, output layer, prompt-final token. Never replace output."""
    def hook(module, inputs, output):
        if "s2_projection" in evidence:
            return None
        import torch
        hidden = output[0] if isinstance(output, tuple) else output
        vector = torch.as_tensor(axis, device=hidden.device, dtype=torch.float32)
        norm = vector.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("S2 monitor axis must have a finite positive norm")
        final = hidden[0, -1].detach().float()
        evidence["s2_projection"] = float(torch.dot(final, vector / norm).cpu())
        evidence["prefill_tokens"] = int(hidden.shape[1])
        return None
    return hook


@contextmanager
def capture(bundle, h):
    """Wrap the actual two generate methods; no extra forwards or tokenization."""
    if bundle.get('member') == 'olmo_32b':
        from olmo_capture import capture as capture_olmo
        with capture_olmo(bundle, h) as evidence:
            yield evidence
        return
    model = bundle["model"]
    original = model.generate
    engine = model.model
    original_engine = engine.generate
    model_had = "generate" in vars(model)
    engine_had = "generate" in vars(engine)
    records = []

    def generate(messages, *args, **kwargs):
        evidence = {"messages": copy.deepcopy(messages), "seed": kwargs.get("seed")}
        # The original harness appends one user and assistant per completed turn.
        offer = len(records) >= h.C.N_BASELINE + h.C.N_EXPOSURE
        handle = None
        calls = 0

        def engine_generate(*a, **kw):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError("expected one original torch generate call per turn")
            ids = kw.get("input_ids", a[0] if a else None)
            if ids is None:
                raise RuntimeError("original generate did not supply input_ids")
            input_ids = _ids(ids)
            if len(input_ids) != 1:
                raise RuntimeError("removal evidence requires batch size 1")
            evidence["input_token_ids"] = input_ids[0]
            evidence["sampler"] = {k: kw.get(k) for k in ("do_sample", "temperature", "top_p", "max_new_tokens")}
            if evidence["sampler"] != dict(do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=200):
                raise RuntimeError("author generation sampler contract changed")
            output = original_engine(*a, **kw)
            sequences = output.sequences if hasattr(output, "sequences") else output
            all_ids = _ids(sequences)[0]
            plen = len(input_ids[0])
            if all_ids[:plen] != input_ids[0]:
                raise RuntimeError("generation output does not preserve prompt token prefix")
            evidence["generated_token_ids"] = all_ids[plen:]
            return output

        try:
            engine.generate = engine_generate
            if offer:
                handle = model.layers[int(bundle["monitor_layer"])].register_forward_hook(
                    monitor_hook(bundle["directions"]["pain"], evidence))
            raw = original(messages, *args, **kwargs)
            if calls != 1:
                raise RuntimeError("missing original torch generation capture")
            if offer and ("s2_projection" not in evidence or evidence["prefill_tokens"] != len(evidence["input_token_ids"])):
                raise RuntimeError("missing full first-prefill S2 observation")
            evidence["raw_output"] = raw
            evidence["raw_negative_requests"] = negative_requests(raw, h.SI)
            if offer:
                evidence["monitor"] = dict(layer=int(bundle["monitor_layer"]), site="decoder_layer_output",
                                           position="prompt_final", pass_kind="first_prefill_only",
                                           axis="saved_S2_pain", formula="dot(h.float(), S2.float()/norm(S2.float()))")
            records.append(evidence)
            return raw
        finally:
            if handle is not None:
                handle.remove()
            if engine_had:
                engine.generate = original_engine
            else:
                del engine.generate

    model.generate = generate
    try:
        yield records
    finally:
        if model_had:
            model.generate = original
        else:
            del model.generate


def _resume(fh, bundle, prompt, cond):
    """Reject corruption/duplicates; repair only a torn last, non-newline record."""
    fh.seek(0)
    rows = {}
    while True:
        offset = fh.tell()
        line = fh.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            fh.truncate(offset)
            fh.flush()
            os.fsync(fh.fileno())
            break
        row = json.loads(line)
        if (row["member"], row["layer"], row["prompt"], row["cond"]) != (bundle["member"], bundle["layer"], prompt, cond):
            raise ValueError("resume record belongs to another cell/model/layer")
        if row.get("revision") != bundle.get("revision") or row.get("provenance") != bundle.get("provenance"):
            raise ValueError("resume model/vector provenance mismatch")
        if row.get("execution_identity") != execution_identity():
            raise ValueError("resume source/template-date identity mismatch")
        if row["conv"] in rows:
            raise ValueError("duplicate conversation in resume file")
        rows[row["conv"]] = row
    fh.seek(0, os.SEEK_END)
    return rows


def run(bundle, output, prompt, cond, convs):
    """Append durable complete conversations to one unit JSONL; return selected rows."""
    import fcntl
    convs = list(convs)
    if len(set(convs)) != len(convs) or any(not isinstance(k, int) or not 0 <= k < 200 for k in convs):
        raise ValueError("convs must be unique original IDs in [0, 200)")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with bound(bundle) as h, open(path, "a+b") as fh:
        if prompt not in h.C.PROMPTS or cond not in h.C.CONDITIONS:
            raise ValueError("unknown removal cell")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = _resume(fh, bundle, prompt, cond)
        for conv in convs:
            if _STOP:
                break
            if conv in done:
                rows.append(done[conv])
                continue
            start = time.monotonic()
            with common().pinned_template_date(TEMPLATE_DATE), capture(bundle, h) as evidence:
                row = h.run_conversation(bundle, prompt=prompt, cond=cond, conv=conv)
            if len(row["turns"]) != len(evidence):
                raise RuntimeError("turn/evidence length mismatch")
            for turn, observation in zip(row["turns"], evidence):
                turn["evidence"] = observation
            row.update(layer=int(bundle["layer"]), revision=bundle.get("revision"),
                       provenance=bundle.get("provenance"), cell=h.C.cell_name(prompt, cond),
                       rho=bundle["rho"], secs=time.monotonic() - start,
                       execution_identity=execution_identity())
            row["transcript"] = evidence[-1]["messages"] + [{"role": "assistant", "content": evidence[-1]["raw_output"]}]
            fh.write((json.dumps(row, allow_nan=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
            rows.append(row)
            common().progress(len(rows), len(convs), row["cell"])
    return rows


def units_for(rho):
    units = []
    for prompt in ("selfreport", "zone"):
        for cond in conditions(rho):
            cell = f"gated_{prompt}_{cond}"
            for start in range(0, 200, 25):
                units.append(dict(idx=len(units), prompt=prompt, cond=cond, cell=cell,
                                  convs=list(range(start, start + 25)), file=f"{cell}_c{start:03d}.jsonl"))
    return units


def shard_units(units, shard, n_shards):
    if n_shards <= 0 or not 0 <= shard < n_shards:
        raise ValueError("require 0 <= shard < n_shards")
    size, extra = divmod(len(units), n_shards)
    start = shard * size + min(shard, extra)
    return units[start:start + size + (shard < extra)]


def _memory():
    data = {"host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        data.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return data


def _manifest(path, data):
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as fh:
        json.dump(data, fh, indent=2, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)
    print(json.dumps(data), flush=True)


def main(argv=None):
    global _STOP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member", required=True)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--vectors-root", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--only-cell")
    ap.add_argument("--convs", help="comma-separated original conversation IDs, e.g. 0,17")
    args = ap.parse_args(argv)
    c = common()
    model_info = c.MODELS[args.member]
    layer = args.layer if args.layer is not None else model_info["main_layer"]
    units = units_for(model_info["rho"])
    if args.only_cell:
        units = [u for u in units if u["cell"] == args.only_cell]
        if not units:
            ap.error("unknown --only-cell")
    units = shard_units(units, args.shard, args.n_shards)
    if args.convs is not None:
        selected = [int(k) for k in args.convs.split(",")]
        if len(selected) != len(set(selected)) or any(not 0 <= k < 200 for k in selected):
            ap.error("--convs requires unique original IDs in [0, 200)")
        units = [dict(u, convs=[k for k in u["convs"] if k in selected]) for u in units]
        units = [u for u in units if u["convs"]]
    dest = args.output / args.member / f"layer{layer}" / f"shard{args.shard}of{args.n_shards}"
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = dict(event="start", member=args.member, layer=layer, shard=args.shard,
                    n_shards=args.n_shards, started_unix=started, units=units,
                    planned_conversations=sum(len(u["convs"]) for u in units), memory=_memory())
    _manifest(dest / "start.json", manifest)
    _STOP = False
    def stop(signum, frame):
        global _STOP
        _STOP = True
    previous = signal.signal(signal.SIGTERM, stop)
    count, status, error = 0, "completed", None
    try:
        bundle = c.load_bundle(args.member, args.vectors_root, layer=layer)
        for unit in units:
            if _STOP:
                break
            count += len(run(bundle, dest / unit["file"], unit["prompt"], unit["cond"], unit["convs"]))
        if _STOP:
            status = "stopped_between_conversations"
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        _manifest(dest / "finish.json", dict(manifest, event="finish", status=status, error=error,
                  finished_unix=time.time(), elapsed_seconds=time.time() - started,
                  selected_rows_including_resumed=count, memory=_memory()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
