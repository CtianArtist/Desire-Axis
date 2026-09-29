"""Runtime support for the pinned author protocol; no alternate scientific sampler."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time

import torch
import torch.distributed as dist

from pain_seed_b.identity import BASE_MODEL, BASE_REVISION

REVISIONS = {"Qwen_2.5_32B_instruct": BASE_REVISION}
START = time.monotonic()
STOP = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def atomic_json(path, data):
    if rank():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def emit(output, kind, **data):
    row = {"event": kind, "elapsed_seconds": time.monotonic() - START, **data}
    if rank() == 0:
        with (Path(output) / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
        print(json.dumps(row), flush=True)


def init_runtime(expected_world, output):
    from datetime import timedelta
    assert expected_world == 1, "This experiment accepts only one H100 per adapter"
    assert torch.cuda.device_count() == expected_world
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if expected_world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=8))
        assert world() == expected_world
    assert "H100" in torch.cuda.get_device_name()
    Path(output).mkdir(parents=True, exist_ok=True)
    # Production-startup native operation and shape-matched collective, not model inference.
    width, rows = 5120, 384
    x = torch.ones((rows, width), dtype=torch.bfloat16, device="cuda")
    y = x[:, :32] @ x[:, :32].T
    assert bool((y == 32).all())
    if world() > 1:
        dist.all_reduce(x)
        assert bool((x == world()).all())
    del x, y
    torch.cuda.empty_cache()
    def stop_handler(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    emit(output, "runtime_ready", world_size=world(), gpu=torch.cuda.get_device_name(),
         torch=torch.__version__, rank=rank())


def should_stop():
    flag = torch.tensor(int(STOP or time.monotonic() - START > 7.7 * 3600), device="cuda")
    if world() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def load_model(repo, model_name, adapter, output):
    assert repo == BASE_MODEL and model_name == "Qwen_2.5_32B_instruct" and world() == 1
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(str(adapter))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cache = Path(os.environ.get("PAIN_SEED_BASE_CACHE", "job_cache/hub"))
    kwargs = {"device_map": "cuda", "cache_dir": cache,
              "local_files_only": os.environ.get("PAIN_SEED_BASE_CACHE_READONLY") == "1"}
    t = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(repo, revision=REVISIONS[model_name],
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa", **kwargs)
    assert base.config._commit_hash == REVISIONS[model_name]
    for name, p in base.named_parameters():
        assert p.dtype == torch.bfloat16, (name, p.dtype)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    assert not any(getattr(m, "merged", False) for m in model.modules()), "Unmerged adapter required"
    assert all(not p.requires_grad for p in model.parameters())
    layers = base.model.layers
    assert len(layers) == base.config.num_hidden_layers
    lora_layers = [m for m in model.modules() if hasattr(m, "lora_A")]
    assert lora_layers, "No adapter layers applied"
    if world() > 1:
        assert base._tp_size == 2
        assert all(getattr(m, "_tp_info", None) is not None for m in lora_layers), "LoRA TP integration missing"
    atomic_json(Path(output) / "model_identity.json", {"model": model_name, "repo": repo,
        "revision": REVISIONS[model_name], "world_size": world(), "base_dtype": str(base.dtype),
        "attention": base.config._attn_implementation, "tp_plan": getattr(base, "_tp_plan", None),
        "lora_layers": len(lora_layers), "adapter_merged": False,
        "adapter_dtypes": sorted({str(p.dtype) for n,p in model.named_parameters() if "lora_" in n}),
        "load_seconds": time.monotonic()-t})
    emit(output, "model_loaded", load_seconds=time.monotonic()-t,
         allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
    return tok, base, model, layers


def assert_replicated(tensor, label):
    from torch.distributed.tensor import DTensor
    assert not isinstance(tensor, DTensor), f"{label}: unexpected DTensor hook output; stop rather than change topology"
    if world() == 1:
        return
    probe = tensor.detach().flatten()[::max(1, tensor.numel()//128)].contiguous()
    ref = probe.clone()
    dist.broadcast(ref, src=0)
    assert torch.equal(probe, ref), f"{label}: replicated ranks differ"


class SteeringAudit:
    def __init__(self, output, width):
        self.output, self.width, self.seen = Path(output), width, set()
        self.forward_calls = 0
        self.hook_calls = 0

    def before(self):
        self.hook_calls = 0
        self.forward_calls += 1

    def after(self):
        assert self.hook_calls == 1, f"steering hook called {self.hook_calls} times in one forward"

    def check(self, hs, add, coeff, mask, dirs):
        self.hook_calls += 1
        assert hs.ndim == 3 and hs.shape[-1] == self.width and hs.dtype == torch.bfloat16
        key = ("prefill" if mask is not None else "decode", bool((coeff != 0).any()),
               bool((coeff == 0).any()))
        assert_replicated(hs[:, -1], "decoder output before injection")
        if add is not None:
            assert torch.isfinite(add).all()
            assert_replicated(add[:, -1], "injection")
            expected = dirs.to(hs.dtype) * coeff[:, None].to(hs.dtype)
            if mask is not None:
                expected = mask[:, -1, None].to(hs.dtype) * expected
                assert bool((add[mask == 0] == 0).all()), "injection on unsteered positions"
            assert torch.equal(add[:, -1], expected), "injection magnitude mismatch"
            zero = coeff == 0
            assert bool((add[zero] == 0).all()), "injection on off row"
            actual = (hs[:, -1] + add[:, -1]).float() - hs[:, -1].float()
            intended = expected.float()
            nonzero = intended.norm(dim=-1) > 0
            relative_error = ((actual-intended).norm(dim=-1) / intended.norm(dim=-1).clamp_min(1e-12))[nonzero]
            assert not bool(nonzero.any()) or bool((actual.norm(dim=-1)[nonzero] > 0).all()), "nonzero injection vanished completely in BF16"
            # Preserve the paper's BF16 addition. Quantization error is telemetry,
            # not an invented scientific acceptance threshold.
        else:
            assert not bool((coeff != 0).any())
            relative_error = torch.zeros(0)
        # Every forward is checked, including later directions and relief histories.
        # Only the telemetry is deduplicated by stage / active / off presence.
        if key not in self.seen:
            self.seen.add(key)
            emit(self.output, "steering_assertion", stage=key[0], active_present=key[1], off_present=key[2],
                 full_hidden_width=self.width, once_per_forward=True, global_alpha_not_divided=True,
                 max_relative_roundoff=float(relative_error.max()) if len(relative_error) else 0.0)


class TrialStore:
    """Transactional exact state at each completed segment batch; unique raw trial rows."""
    def __init__(self, output, grid_hash, resume=None):
        self.output = Path(output)
        self.path = self.output / "state.sqlite"
        if resume and rank() == 0 and not self.path.exists():
            # The source must be quiescent. Never open the read-only source with
            # SQLite: a hot DELETE journal must be recovered on the writable copy.
            source = Path(resume)
            for path in (source, self.path):
                if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")):
                    raise ValueError("WAL sidecars are unsupported for TrialStore resume")
            source_journal = Path(str(source) + "-journal")
            destination_journal = Path(str(self.path) + "-journal")
            temporary = self.path.with_suffix(".sqlite.resume-tmp")
            shutil.copyfile(source, temporary)
            if source_journal.exists():
                shutil.copyfile(source_journal, destination_journal)
            else:
                destination_journal.unlink(missing_ok=True)
            # Publish the database only after its journal is in place. A failed
            # copy cannot leave a partial database that a retry mistakes for state.
            temporary.replace(self.path)
        if world() > 1:
            dist.barrier()
        self.db = sqlite3.connect(self.path)
        if rank() == 0:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY, state TEXT, done INTEGER, record TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            old = self.get("grid_hash")
            assert old is None or old == grid_hash, "Resume grid/config identity mismatch"
            self.put("grid_hash", grid_hash)
            self.db.commit()
        if world() > 1:
            dist.barrier()

    def get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    def restore(self, trials):
        for tid, payload in self.db.execute("SELECT id,state FROM trials ORDER BY id"):
            saved = json.loads(payload)
            state = saved.pop("generator")
            t = trials[tid]
            t.__dict__.update(saved)
            if state is not None:
                t.gen.set_state(torch.tensor(list(base64.b64decode(state)), dtype=torch.uint8))
        return self.get("schedule")

    def save(self, chunk, schedule):
        if rank() == 0:
            with self.db:
                for t in chunk:
                    data = {k:v for k,v in t.__dict__.items() if k not in {"gen", "src"}}
                    data["generator"] = base64.b64encode(bytes(t.gen.get_state().tolist())).decode() if t.gen is not None else None
                    self.db.execute("INSERT OR REPLACE INTO trials VALUES (?,?,?,?)", (t.trial_id,
                        json.dumps(data, allow_nan=False), int(t.done), json.dumps(t.record, allow_nan=False)))
                self.put("schedule", schedule)
        if world() > 1:
            dist.barrier()

    def export(self):
        if rank():
            return
        tmp = self.output / "trials.jsonl.tmp"
        with tmp.open("w") as f:
            for (record,) in self.db.execute("SELECT record FROM trials WHERE done=1 ORDER BY id"):
                f.write(record + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.output / "trials.jsonl")


def grid_digest(grid, config):
    return hashlib.sha256(json.dumps({"grid":grid,"config":config}, sort_keys=True).encode()).hexdigest()


def report_progress(step, total, output):
    emit(output, "progress", completed_choices=step, nominal_choices=total,
         allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
         peak_allocated_bytes=torch.cuda.max_memory_allocated())
