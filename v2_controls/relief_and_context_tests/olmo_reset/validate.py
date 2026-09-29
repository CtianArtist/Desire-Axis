"""Bounded packaging tests: source compilation, hashes, saved counts, dry-run isolation."""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    count=0
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(),str(path),'exec'); count+=1
    manifest=json.loads((ROOT/'file_manifest.json').read_text())
    for row in manifest:
        path=ROOT/row['path']; raw=path.read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    p=subprocess.run([sys.executable,str(ROOT/'reproduce.py')],capture_output=True,text=True,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'infer.py'),'--data-dir','/data','--output','/output','--dry-run'],capture_output=True,text=True,check=True)
    value=json.loads(p.stdout);assert value['loads_model'] is False and value['status']=='dry_run'
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.json','.md','.txt','.lock','.csv'):
            text=path.read_text()
            # Assemble search tokens to avoid flagging this test's own literals.
            for forbidden in ('artifact'+':/', '/s'+'rv/', '/__'+'modal/', '/work'+'space/', 'SIL'+'ICO_', 'exp_'+'01'):
                assert forbidden not in text,(path,forbidden)
    print(json.dumps(dict(status='passed',compiled=count,hashes=len(manifest),dry_run_no_model=True,saved_table_replay=True,private_path_scan=True)))
if __name__=='__main__': main()
