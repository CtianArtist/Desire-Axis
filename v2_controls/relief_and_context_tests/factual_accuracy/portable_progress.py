"""Optional console progress; no sampling or model operations."""
import json
def report_progress(**fields):
    print(json.dumps({"event": "progress", **fields}), flush=True)
