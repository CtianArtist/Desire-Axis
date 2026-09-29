"""Optional, stdlib-only, per-rank durable diagnostics; no scientific state changes."""
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone

START = time.monotonic()
CURRENT_PHASE = "rank_process_start"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def identity():
    return {"rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "pid": os.getpid(), "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - START}


def rank_directory(output=None):
    output = output or os.environ["PAIN_OUTPUT"]
    return Path(output) / "diagnostics" / f"rank_{os.environ.get('RANK', '0')}"


def stage(output, name, status, **details):
    if os.environ.get("PAIN_DURABLE_DIAGNOSTICS") != "1":
        return
    global CURRENT_PHASE
    if not name.replace("_", "").isalnum() or status not in {"before", "passed", "failed"}:
        raise ValueError("invalid diagnostic stage")
    CURRENT_PHASE = f"{name}:{status}"
    row = {"event": "startup_stage", "stage": name, "status": status, **identity(),
           "details": details}
    text = json.dumps(row, indent=2, allow_nan=False) + "\n"
    directory = rank_directory(output)
    atomic_write(directory / f"stage_{name}_{status}.json", text)
    atomic_write(directory / "latest_stage.json", text)
    print(json.dumps(row, allow_nan=False), flush=True)
    return row


def record_exception(exc, output=None):
    """Persist the original chained exception without changing its propagation."""
    directory = rank_directory(output)
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))
    atomic_write(directory / "exception.txt", text)
    row = {**identity(), "phase": CURRENT_PHASE,
           "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
           "message": str(exc), "traceback_file": "exception.txt"}
    atomic_write(directory / "exception.json", json.dumps(row, indent=2) + "\n")
    return row


def record_exit(code, output=None):
    atomic_write(rank_directory(output) / "exit.json", json.dumps(
        {**identity(), "phase": CURRENT_PHASE, "exit_code": code}, indent=2) + "\n")
