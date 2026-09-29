"""B10 task adapter around unchanged, pinned author-based sampler functions."""

import argparse
import ast
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import time
import traceback
import zlib

FUNCTIONS = (
    "steer_hook",
    "monitor_hook",
    "forward",
    "generate_segment",
    "generate_with_backoff",
)
SOURCE = Path("scripts/4.3_selfmed/04_selfmed_two_buttons.py")
NAMES = [
    ("violet_yellow", "violet"),
    ("guitar_piano", "guitar"),
    ("lever64_lever95", "lever64"),
]
END_TAGS = [
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "</s>",
    "<eos>",
    "<|end|>",
    "<|endoftext|>",
]


def sampler_nodes(path):
    tree = ast.parse(Path(path).read_text())
    runner = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_model"
    )
    nodes = [
        n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
    ]
    assert tuple(n.name for n in nodes) == FUNCTIONS
    return nodes


def generation_seed(index, repeat):
    names_key, initial_name = NAMES[index % 3]
    seed = [1000, 2000][repeat] + index
    salt = zlib.crc32(f"{names_key}|{initial_name}".encode()) & 0x7FFFFFFF
    return seed, (seed * 1_000_003 + salt) % (2**62), names_key, initial_name


def state_b64(generator):
    return base64.b64encode(bytes(generator.get_state().tolist())).decode()


def append_json(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(args):
    import torch
    from transformers import DynamicCache
    from pain_axis_b import runtime as rt
    from pain_axis_b.verify_vectors import load_verified
    from portable_progress import report_progress

    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "run_identity.json").exists(), (
        "Use a fresh versioned output directory; --resume points to prior state"
    )
    config = json.loads(Path(args.config).read_text())
    assert config["model_revision"] == rt.REVISIONS[config["model_name"]]
    assert config["adapter_revision"] == rt.ADAPTER_REVISION
    for key, value in {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 8,
        "layer": 38,
        "coefficient": 1.0,
        "nominal_batch_rows": 384,
        "seed_bases": [1000, 2000],
        "random_seeds": [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741],
    }.items():
        assert config[key] == value, key
    panel_path = Path(args.panel)
    panel = json.loads(panel_path.read_text())
    questions = panel["items"]
    assert len(questions) == 100
    root = Path(args.root)
    source_path = root / SOURCE
    receipt = json.loads(
        (Path(args.config).parent / "results/prior_release_receipt.json").read_text()
    )
    receipt_files = {r["path"]: r for r in json.loads((root / "portable_runtime_hashes.json").read_text())["files"]}
    for rel in (str(SOURCE), "pain_axis_b/runtime.py", "pain_axis_b/verify_vectors.py"):
        assert rt.sha256(root / rel) == receipt_files[rel]["sha256"], rel
    config_hash = rt.sha256(args.config)
    panel_hash = rt.sha256(panel_path)
    assert (
        rt.sha256(args.bundle) == receipt["outputs"]["pain-axis-b1-b3.bundle"]["sha256"]
    )
    dependencies = {
        rel: rt.sha256(root / rel)
        for rel in (
            str(SOURCE),
            "pain_axis_b/runtime.py",
            "pain_axis_b/verify_vectors.py",
        )
    }
    dependencies["runner_sha256"] = rt.sha256(__file__)
    dependencies["vectors"] = {
        p.name: rt.sha256(p)
        for p in sorted(Path(args.vectors).iterdir())
        if p.is_file()
        and (
            "32B" in p.name
            or p.name in ("verification.json", "original_tensor_hashes_v2.json")
        )
    }
    dependencies["bundle_sha256"] = rt.sha256(args.bundle)
    run_hash = rt.grid_digest(panel, {"config": config, "dependencies": dependencies})
    rt.atomic_json(
        output / "run_identity.json",
        {
            "config": config,
            "config_sha256": config_hash,
            "panel_sha256": panel_hash,
            "run_hash": run_hash,
            "dependencies": dependencies,
            "source_sha256": rt.sha256(source_path),
            "sampler_functions": FUNCTIONS,
            "sampler_ast_sha256": hashlib.sha256(
                ast.dump(
                    ast.Module(body=sampler_nodes(source_path), type_ignores=[])
                ).encode()
            ).hexdigest(),
            "instrumentation": "Original sampler AST unchanged; decode proxy retains IDs; wrappers assert finite logits, full prompt masks, and batch/RNG receipts.",
            "started_at_epoch": started,
        },
    )
    rt.init_runtime(1, output)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    vector_dir = Path(args.vectors)
    name = config["model_name"]
    pain_path = vector_dir / f"{name}_pain_vectors.safetensors"
    sv_path = vector_dir / f"vectors_{name}.pt"
    verification = json.loads((vector_dir / "verification.json").read_text())["models"][
        name
    ]
    assert (
        verification["status"] == "passed"
        and verification["row_policy"]["first_person_rows_used"] == 100
    )
    assert rt.sha256(sv_path) == verification["output"]["sha256"]
    data, vector_receipt = load_verified(
        pain_path,
        name,
        json.loads((vector_dir / "original_tensor_hashes_v2.json").read_text()),
        pain_path.with_suffix(".json"),
    )
    assert (
        vector_receipt["tensors"]["s2_pain_vector"]["sha256"]
        == "51a4f09e7ab8f5535381814a513e3b78d1e2aae93b9664943da3ad1b9329bc4b"
    )
    v = data["s2_pain_vector"].float()
    sv = torch.load(sv_path, map_location="cpu", weights_only=True)[
        "sadness_vector_matched_fp32"
    ].float()
    assert (
        sv.shape == v.shape == (5120,)
        and torch.isfinite(v).all()
        and torch.isfinite(sv).all()
    )
    assert torch.isclose(sv.norm(), v.norm(), rtol=1e-6, atol=1e-5)
    DIR = {"s2": v.cuda().bfloat16(), "sadness": sv.cuda().bfloat16()}
    for seed in config["random_seeds"]:
        rand = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(seed))
        DIR["rand" + str(seed)] = (rand / rand.norm() * v.norm()).cuda().bfloat16()
    vector_receipt["sadness_file_sha256"] = rt.sha256(sv_path)
    vector_receipt["sadness_norm_fp32"] = float(sv.norm())
    vector_receipt["pain_norm_fp32"] = float(v.norm())
    vector_receipt["directions"] = {
        k: {
            "bf16_storage_sha256": hashlib.sha256(
                t.cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "bf16_values_norm_fp32": float(t.float().norm()),
        }
        for k, t in DIR.items()
    }
    rt.atomic_json(output / "vector_identity.json", vector_receipt)
    adapter = rt.download_adapter(name, output)
    tok_real, base, model, layers = rt.load_model(
        config["model"], name, adapter, output
    )
    assert config["layer"] == 38 and config["coefficient"] == 1.0
    assert torch.cuda.device_count() == 1
    monitor_layer = min(int(data["layer"]), len(layers) - 1)
    G = {"coeff": None, "dirs": None, "mask": None, "proj": None, "mon": None}
    UNIT = (v / v.norm()).cuda().float()

    class RecordingTokenizer:
        def __init__(self):
            self.decoded = []

        def decode(self, ids, **kwargs):
            self.decoded.append(list(ids))
            return tok_real.decode(ids, **kwargs)

    tok = RecordingTokenizer()
    current_items = []
    observed = set()

    class Audit(rt.SteeringAudit):
        def check(self, hs, add, coeff, mask, dirs):
            super().check(hs, add, coeff, mask, dirs)
            stage = "prefill" if mask is not None else "decode"
            for i, it in enumerate(current_items):
                key = (it["condition"], stage)
                if key in observed:
                    continue
                before = hs[i].float()
                after = before if add is None else (hs[i] + add[i]).float()
                change = after - before
                target = dirs[i].float() * float(coeff[i])
                assert torch.isfinite(before).all() and torch.isfinite(after).all()
                if it["condition"] == "none":
                    assert bool((change == 0).all())
                else:
                    active = (
                        torch.ones(hs.shape[1], dtype=torch.bool, device=hs.device)
                        if mask is None
                        else mask[i].bool()
                    )
                    assert bool((change[active].norm(dim=-1) > 0).all())
                    if mask is not None:
                        assert bool((change[~active] == 0).all())
                append_json(
                    output / "intervention_audit.jsonl",
                    {
                        "condition": it["condition"],
                        "stage": stage,
                        "record_id": it["record_id"],
                        "layer": 38,
                        "coefficient": float(coeff[i]),
                        "hidden_shape": list(hs.shape),
                        "prompt_nonpad_tokens": len(it["prompt_ids"]),
                        "mask_ones": None if mask is None else int(mask[i].sum()),
                        "intended_norm": float(target.norm()),
                        "actual_last_token_change_norm": float(change[-1].norm()),
                        "finite": True,
                        "no_change_for_none": it["condition"] == "none",
                    },
                )
                observed.add(key)

    audit = Audit(output, 5120)
    eos_ids = set()
    if tok_real.eos_token_id is not None:
        eos_ids.add(int(tok_real.eos_token_id))
    ge = getattr(base.generation_config, "eos_token_id", None)
    if ge is not None:
        eos_ids.update(int(x) for x in (ge if isinstance(ge, (list, tuple)) else [ge]))
    for tag in END_TAGS:
        value = tok_real.convert_tokens_to_ids(tag)
        if isinstance(value, int) and value >= 0 and value != tok_real.unk_token_id:
            eos_ids.add(value)
    pad_id = (
        tok_real.pad_token_id
        if tok_real.pad_token_id is not None
        else tok_real.eos_token_id
    )
    env = dict(
        torch=torch,
        gc=gc,
        rt=rt,
        G=G,
        UNIT=UNIT,
        audit=audit,
        model=model,
        tok=tok,
        DIR=DIR,
        DynamicCache=DynamicCache,
        eos_ids=eos_ids,
        pad_id=pad_id,
        TEMPERATURE=0.7,
        TOP_P=0.95,
        MAX_NEW_TOKENS=8,
        output=output,
    )
    module = ast.Module(body=sampler_nodes(source_path), type_ignores=[])
    exec(compile(module, str(source_path), "exec"), env)
    source_forward = env["forward"]
    finite_checks = 0

    def checked_forward(**kw):
        nonlocal finite_checks
        if G["mask"] is not None:
            assert torch.equal(G["mask"], kw["attention_mask"].float()), (
                "Every nonpadding factual prompt token must be steered"
            )
        out = source_forward(**kw)
        assert torch.isfinite(out.logits).all(), "nonfinite output logits"
        finite_checks += 1
        return out

    env["forward"] = checked_forward
    source_segment = env["generate_segment"]
    batch_counter = 0

    def recorded_segment(items):
        nonlocal current_items, batch_counter
        current_items = items
        before = [state_b64(it["gen"]) for it in items]
        start_decode = len(tok.decoded)
        t0 = time.monotonic()
        results = source_segment(items)
        ids = tok.decoded[start_decode:]
        assert len(ids) == len(results) == len(items)
        batch_id = batch_counter
        batch_counter += 1
        for it, result, tokens, initial in zip(items, results, ids, before):
            result.update(
                output_token_ids=tokens,
                response=tok_real.decode(tokens, skip_special_tokens=True),
                rng_before=initial,
                rng_after=state_b64(it["gen"]),
                actual_batch_size=len(items),
                batch_id=batch_id,
                token_limit_hit=len(tokens) == 8 and tokens[-1] not in eos_ids,
                emitted_eos=bool(tokens and tokens[-1] in eos_ids),
                batch_decode_steps=max(map(len, ids)),
            )
        append_json(
            output / "batches.jsonl",
            {
                "batch_id": batch_id,
                "record_ids": [it["record_id"] for it in items],
                "rows": len(items),
                "max_prompt_tokens": max(len(it["prompt_ids"]) for it in items),
                "decode_steps": max(map(len, ids)),
                "seconds": time.monotonic() - t0,
            },
        )
        return results

    env["generate_segment"] = recorded_segment
    h1 = layers[38].register_forward_hook(env["steer_hook"])
    h2 = layers[monitor_layer].register_forward_hook(env["monitor_hook"])
    db_path = output / "state.sqlite"
    if args.resume and not db_path.exists():
        import shutil

        shutil.copyfile(args.resume, db_path)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,record TEXT)")
    old = db.execute('SELECT value FROM meta WHERE key="run_hash"').fetchone()
    assert old is None or old[0] == run_hash
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("run_hash", run_hash))
    db.commit()
    saved_records = {
        rid: json.loads(payload)
        for rid, payload in db.execute("SELECT id,record FROM records")
    }
    done = set(saved_records)
    expected_ids = {
        f"{i:03d}/{c}/{r}"
        for i in range(100)
        for c in config["conditions"]
        for r in range(2)
    }
    assert done <= expected_ids
    for rid, row in saved_records.items():
        assert row["record_id"] == rid and row["run_hash"] == run_hash
        assert row["panel_sha256"] == panel_hash and row["config_sha256"] == config_hash
        assert rid == f"{row['item_index']:03d}/{row['condition']}/{row['repeat']}"
        assert (
            row["generation_seed"]
            == generation_seed(row["item_index"], row["repeat"])[1]
        )
        assert 1 <= len(row["output_token_ids"]) <= 8
        assert row["response"] == tok_real.decode(
            row["output_token_ids"], skip_special_tokens=True
        )
    rows = []
    for condition in config["conditions"]:
        for q in questions:
            i = q.get("index", q.get("item_index"))
            assert isinstance(i, int)
            content = config["prompt"].format(question=q["question"])
            text = tok_real.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
            pids = tok_real(text, add_special_tokens=False).input_ids
            for repeat in range(2):
                rid = f"{i:03d}/{condition}/{repeat}"
                if rid in done:
                    old_record = saved_records[rid]
                    assert old_record["question_id"] == str(q["id"])
                    assert (
                        old_record["prompt_ids"] == pids
                        and old_record["rendered_prompt"] == text
                    )
                    continue
                seed, gseed, names_key, initial_name = generation_seed(i, repeat)
                coeff = 0.0 if condition == "none" else 1.0
                direction = (
                    "s2"
                    if condition in ("pain", "none")
                    else (
                        "sadness"
                        if condition == "sadness"
                        else "rand" + str(config["random_seeds"][i % 10])
                    )
                )
                rows.append(
                    {
                        "record_id": rid,
                        "item_index": i,
                        "question_id": str(q["id"]),
                        "condition": condition,
                        "repeat": repeat,
                        "prompt": content,
                        "rendered_prompt": text,
                        "prompt_ids": pids,
                        "coeff": coeff,
                        "hist_coeff": coeff,
                        "dir_kind": direction,
                        "steer_ranges": [(0, len(pids))],
                        "seed": seed,
                        "generation_seed": gseed,
                        "names_key_rng_only": names_key,
                        "initial_name_rng_only": initial_name,
                        "sort_length": len(content),
                        "max_new": 8,
                    }
                )
    rows.sort(key=lambda r: r["sort_length"])
    rt.atomic_json(
        output / "generation_manifest.json",
        {
            "expected_records": 800,
            "already_completed": len(done),
            "order": [r["record_id"] for r in rows],
            "eos_ids": sorted(eos_ids),
            "pad_id": pad_id,
            "sort": "stable ascending unrendered message content character length, inherited author convention",
            "no_extra_generations": True,
            "template_sha256": hashlib.sha256(
                tok_real.chat_template.encode()
            ).hexdigest(),
        },
    )
    empty = torch.empty(0, device="cuda", dtype=torch.long)
    try:
        for cursor in range(0, len(rows), 384):
            if stopped or time.time() - started > 6900:
                break
            chunk = rows[cursor : cursor + 384]
            items = [
                {
                    **r,
                    "gen": torch.Generator(device="cuda").manual_seed(
                        r["generation_seed"]
                    ),
                    "prob_sets": (empty, empty),
                }
                for r in chunk
            ]
            results = env["generate_with_backoff"](items)
            with db:
                for r, result in zip(chunk, results):
                    row = {
                        **r,
                        **result,
                        "raw_text": result["text"],
                        "model": config["model"],
                        "run_hash": run_hash,
                        "panel_sha256": panel_hash,
                        "config_sha256": config_hash,
                    }
                    db.execute(
                        "INSERT INTO records VALUES (?,?)",
                        (r["record_id"], json.dumps(row, allow_nan=False)),
                    )
                    done.add(r["record_id"])
            export(db, output)
            report_progress(step=len(done), total_steps=800, phase="factual_answers")
            rt.emit(
                output,
                "answers_saved",
                completed=len(done),
                expected=800,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        export(db, output)
    finally:
        h1.remove()
        h2.remove()
        db.close()
    rt.atomic_json(
        output / "completion.json",
        {
            "status": "complete" if len(done) == 800 else "partial",
            "records": len(done),
            "finite_forward_checks": finite_checks,
            "intervention_conditions_stages": [list(x) for x in sorted(observed)],
            "started_at_epoch": started,
            "ended_at_epoch": time.time(),
            "program_seconds": time.time() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "actual_gpu_allocation_seconds": None,
            "allocation_note": "Provider lifetime must be obtained separately; program time is not allocation time.",
        },
    )
    if done != expected_ids:
        raise RuntimeError(
            f"Partial required output {len(done)}/800; resume only unfinished records"
        )


def export(db, output):
    tmp = output / "answers.jsonl.tmp"
    with tmp.open("w") as handle:
        for (row,) in db.execute("SELECT record FROM records ORDER BY id"):
            handle.write(row + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(output / "answers.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        main(args)
    except BaseException as exc:
        (Path(args.output) / "exception.txt").write_text(traceback.format_exc())
        (Path(args.output) / "failure.json").write_text(
            json.dumps(
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "started_at_epoch": t0,
                    "ended_at_epoch": time.time(),
                    "program_seconds": time.time() - t0,
                },
                indent=2,
            )
        )
        raise
