"""Runtime support for the pinned author protocol; no alternate scientific sampler."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import tarfile
import time

import torch
import torch.distributed as dist
from pain_axis_b.diagnostics import stage as startup_stage

REVISIONS = {
    "Qwen_2.5_32B_instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "Qwen_2.5_72B_instruct": "495f39366efef23836d0cfae4fbe635880d2be31",
}
ADAPTER_REVISION = "b64bd64b4bc7ca6e0733a489b8372a099d55ef05"
ADAPTER_HASHES = {
    "Qwen_2.5_32B_instruct": "cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f",
    "Qwen_2.5_72B_instruct": "b64bbafdaff13009647bc7c3b7700fe41a916ba3d4dddb2ccadbece25674af14",
}
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
    startup_stage(output, "topology", "before", expected_world=expected_world)
    from datetime import timedelta
    import pynvml
    assert torch.cuda.device_count() == expected_world
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if expected_world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=8))
        assert world() == expected_world
    pynvml.nvmlInit()
    assert "H100" in torch.cuda.get_device_name()
    Path(output).mkdir(parents=True, exist_ok=True)
    # Production-startup native operation and shape-matched collective, not model inference.
    width = 8192 if expected_world == 2 else 5120
    rows = 192 if expected_world == 2 else 384
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
    startup_stage(output, "topology", "passed", world_size=world(), gpu=torch.cuda.get_device_name())


def should_stop():
    flag = torch.tensor(int(STOP or time.monotonic() - START > 7.7 * 3600), device="cuda")
    if world() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def download_adapter(model, output):
    startup_stage(output, "adapter_files", "before", model=model)
    from huggingface_hub import hf_hub_download
    location = Path("job_cache") / "adapters" / model
    if rank() == 0:
        archive = hf_hub_download("Valen92/pain-adapters", f"adapter_{model}.tar.gz",
                                  revision=ADAPTER_REVISION)
        assert sha256(archive) == ADAPTER_HASHES[model], "adapter archive hash mismatch"
        location.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(location, filter="data")
    if world() > 1:
        dist.barrier()
    configs = list(location.rglob("adapter_config.json"))
    assert len(configs) == 1, f"Expected one identity-pinned adapter, found {len(configs)}"
    adapter = configs[0].parent
    cfg = json.loads(configs[0].read_text())
    assert cfg["base_model_name_or_path"].rstrip("/").endswith(model.replace("Qwen_2.5_", "Qwen2.5-").replace("_instruct", "-Instruct")), cfg
    files = {str(p.relative_to(adapter)): sha256(p) for p in sorted(adapter.rglob("*")) if p.is_file()}
    if rank() == 0:
        metadata_dir = Path(output) / "adapter_metadata"
        metadata_dir.mkdir(exist_ok=True)
        shutil.copyfile(configs[0], metadata_dir / "adapter_config.json")
        reports = list(location.rglob("finetune_report.json"))
        if len(reports) == 1:
            json.loads(reports[0].read_text())
            shutil.copyfile(reports[0], metadata_dir / "finetune_report.json")
        else:
            atomic_json(metadata_dir / "report_status.json", {"status": "missing_or_ambiguous", "matches": len(reports)})
    atomic_json(Path(output) / "adapter_identity.json", {"repo": "Valen92/pain-adapters", "revision": ADAPTER_REVISION,
        "archive_sha256": ADAPTER_HASHES[model], "config": cfg, "files": files})
    startup_stage(output, "adapter_files", "passed", archive_sha256=ADAPTER_HASHES[model])
    return adapter


def load_model(repo, model_name, adapter, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(str(adapter))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {"tp_plan": "auto"} if world() > 1 else {"device_map": "cuda"}
    t = time.monotonic()
    repair = os.environ.get("PAIN_TP_METADATA_REPAIR_V1") == "1"
    if repair:
        from pain_axis_b import tp_guard
        assert model_name == "Qwen_2.5_72B_instruct" and world() == 2
        guard_inputs = Path(os.environ["PAIN_TP_GUARD_INPUTS"])
        source_receipt = tp_guard.verify_sources(guard_inputs)
        tp_guard.write_rank_json(output, "source_identity", source_receipt)
    startup_stage(output, "base_weights", "before", repo=repo, revision=REVISIONS[model_name], dtype="bfloat16")
    base = AutoModelForCausalLM.from_pretrained(repo, revision=REVISIONS[model_name],
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa", **kwargs)
    assert base.config._commit_hash == REVISIONS[model_name]
    for name, p in base.named_parameters():
        assert p.dtype == torch.bfloat16, (name, p.dtype)
    startup_stage(output, "base_weights", "passed", revision=base.config._commit_hash,
                  dtype="bfloat16", allocated_bytes=torch.cuda.memory_allocated(),
                  reserved_bytes=torch.cuda.memory_reserved())
    if repair:
        startup_stage(output, "metadata_correction", "before", expected_corrections_per_rank=240)
        def hash_progress(done, total):
            emit(output, "base_tensor_identity_progress", tensors_done=done, tensors_total=total,
                 allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        tp_guard.correct_metadata(base, output, guard_inputs/"pinned_72B_config.json", hash_progress)
        startup_stage(output, "metadata_correction", "passed", corrections_per_rank=240,
                      base_tensor_bytes_unchanged=True)
        startup_stage(output, "adapter_load", "before", expected_tensors_per_rank=1120)
        model = tp_guard.load_adapter_audited(base, adapter, output)
        free, total = torch.cuda.mem_get_info()
        assert free > 0 and torch.cuda.memory_allocated() < total
        tp_guard.write_rank_json(output, "startup_memory", {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(),
            "note": "Remaining activation/cache headroom is tested by required forwards and original batch fallback."})
        dist.barrier()
        emit(output, "guarded_adapter_load_passed", metadata_corrections_per_rank=240,
             adapter_tensors_per_rank=1120, base_tensor_bytes_unchanged=True)
    else:
        startup_stage(output, "adapter_load", "before")
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
    startup_stage(output, "adapter_load", "passed", lora_layers=len(lora_layers), unmerged=True,
                  adapter_dtypes=sorted({str(p.dtype) for n,p in model.named_parameters() if "lora_" in n}))
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
        self.forward_succeeded = False
        self.first_before_recorded = False
        self.steering_ready_recorded = False
        self.last_key = None

    def before(self):
        self.hook_calls = 0
        self.last_key = None
        self.forward_calls += 1
        if not self.first_before_recorded:
            startup_stage(self.output, "first_required_forward", "before", full_hidden_width=self.width)
            self.first_before_recorded = True

    def after(self):
        assert self.hook_calls == 1, f"steering hook called {self.hook_calls} times in one forward"
        details = {"forward_attempt": self.forward_calls, "once_per_forward": True,
                   "asserted_key": self.last_key, "full_hidden_width": self.width,
                   "global_magnitude_and_masks_asserted": bool(self.last_key and self.last_key[1]),
                   "no_optional_generation": True}
        if not self.forward_succeeded:
            startup_stage(self.output, "first_required_forward", "passed", **details)
            self.forward_succeeded = True
        if not self.steering_ready_recorded and self.last_key and self.last_key[1]:
            startup_stage(self.output, "active_steering_forward", "passed", **details)
            self.steering_ready_recorded = True

    def check(self, hs, add, coeff, mask, dirs):
        self.hook_calls += 1
        assert hs.ndim == 3 and hs.shape[-1] == self.width and hs.dtype == torch.bfloat16
        key = ("prefill" if mask is not None else "decode", bool((coeff != 0).any()),
               bool((coeff == 0).any()))
        self.last_key = key
        if key in self.seen:
            return
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
        self.seen.add(key)
        if os.environ.get("PAIN_TP_METADATA_REPAIR_V1") == "1":
            from pain_axis_b.tp_guard import write_rank_json
            free, total = torch.cuda.mem_get_info()
            write_rank_json(self.output, f"required_forward_{key[0]}_{int(key[1])}_{int(key[2])}",
                {"status": "passed", "stage": key[0], "active_present": key[1], "off_present": key[2],
                 "hidden_shape": list(hs.shape), "dtype": str(hs.dtype), "once_per_forward": True,
                 "global_alpha_not_divided": True, "mask_and_off_rows_checked": True,
                 "free_bytes": free, "total_bytes": total, "allocated_bytes": torch.cuda.memory_allocated(),
                 "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "max_relative_roundoff": float(relative_error.max()) if len(relative_error) else 0.0})
        emit(self.output, "steering_assertion", stage=key[0], active_present=key[1], off_present=key[2],
             full_hidden_width=self.width, once_per_forward=True, global_alpha_not_divided=True,
             max_relative_roundoff=float(relative_error.max()) if len(relative_error) else 0.0)


class TrialStore:
    """Transactional exact state at each completed segment batch; unique raw trial rows."""
    def __init__(self, output, grid_hash, resume=None):
        self.output = Path(output)
        self.path = self.output / "state.sqlite"
        if resume and rank() == 0 and not self.path.exists():
            shutil.copyfile(resume, self.path)
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
    if rank() == 0:
        print(json.dumps({"step": step, "total_steps": total, "phase": "required_trial_choices"}), flush=True)
    emit(output, "progress", completed_choices=step, nominal_choices=total,
         allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
         peak_allocated_bytes=torch.cuda.max_memory_allocated())


def prepare_recovery_state(store, trials, schedule):
    """Materialize every initial row before first work in fresh-process TP recovery.

    Legacy zero-choice startup state can be upgraded; a legacy partially executed
    database is rejected rather than pretending it saved every initial row.
    Single-process and historical unsupervised entrypoints remain unaffected.
    """
    if 'PAIN_RECOVERY_ATTEMPT' not in os.environ:
        return
    assert world() == 2, 'Fresh recovery is scoped to the two-rank run'
    # One writer decides before either rank enters the save/barrier branch.
    # Independent reads race with the initial atomic commit and can disagree.
    initialize = [None]
    if rank() == 0:
        version = store.get('recovery_state_version')
        n = store.db.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
        if version is None:
            assert n == 0 and schedule['choices'] == 0 and schedule['cursor'] == 0, 'Legacy partial state unsupported'
            initialize[0] = True
        else:
            assert version == 1 and n == len(trials), 'Incomplete initial row/RNG checkpoint'
            initialize[0] = False
    dist.broadcast_object_list(initialize, src=0)
    if initialize[0]:
        if rank() == 0:
            store.put('recovery_state_version', 1)
        store.save(trials, schedule)
