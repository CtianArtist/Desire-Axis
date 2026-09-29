"""B2/B3 use the isolated complete B1-B3 release, never the B6 runtime."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2]/"steering_state_controls"/"inference.py"
    raise SystemExit(subprocess.call([sys.executable,str(target),"--study","b1_b3",*sys.argv[1:]]))
