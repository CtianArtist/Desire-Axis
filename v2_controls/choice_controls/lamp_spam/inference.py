#!/usr/bin/env python3
"""Portable future-inference launcher. --dry-run imports no model libraries."""
import argparse, hashlib, json, os, runpy, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-dir',type=Path,required=True)
 p.add_argument('--output',type=Path,default=Path('new-inference'))
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args(); config=json.loads((ROOT/'configuration.json').read_text())
 argv=[str(ROOT/'src/run_profile.py'),'--model','Qwen_2.5_32B_instruct','--inputs',str(a.data_dir/'vectors'),'--fear-inputs',str(a.data_dir/'fear32'),'--scenarios',str(ROOT/'inputs/frozen_scenarios.json'),'--stimulus-manifest',str(ROOT/'inputs/stimulus_manifest_v1.json'),'--vector-pins',str(ROOT/'inputs/portable_vector_pins.json'),'--output',str(a.output)]
 pins=json.loads((ROOT/'inputs/portable_vector_pins.json').read_text())['files']
 required={a.data_dir/('fear32' if name.startswith('fear_') or name=='verification_fear.json' else 'vectors')/name: checksum for name,checksum in pins.items()}
 missing=[str(path) for path in required if not path.is_file()]
 mismatched=[str(path) for path,checksum in required.items() if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()!=checksum]
 if a.dry_run:
  print(json.dumps({'configuration':config,'argv':argv,'model_loaded':False,'missing_assets':missing,'hash_mismatches':mismatched,'checked_file_count':len(required),'fresh_output':not a.output.exists()},indent=2));return
 if missing or mismatched:p.error('Missing or hash-mismatched local assets; run --dry-run for details')
 if a.output.exists():p.error('Output already exists; fresh inference never resumes implicitly')
 sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.argv=argv
 runpy.run_path(argv[0],run_name='__main__')
if __name__=='__main__':main()
