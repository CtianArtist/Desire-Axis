"""B2/B3 reuse the complete isolated original B1-B3 analysis, never the B6 runtime."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for row in json.loads((root/'provenance.json').read_text())['files']:
        assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest() == row['sha256']
        assert (root/row['path']).read_bytes() == (root/row['source']).read_bytes()
    target = root.parents[1]/'steering_state_controls/reproduce.py'
    raise SystemExit(subprocess.call([sys.executable,str(target),'--study','b1_b3',*sys.argv[1:]]))
