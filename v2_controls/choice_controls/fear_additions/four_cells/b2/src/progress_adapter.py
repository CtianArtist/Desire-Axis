"""Portable no-op telemetry installer; runtime retains JSON progress."""
def install(runtime):
    return {"telemetry": "local_json"}
