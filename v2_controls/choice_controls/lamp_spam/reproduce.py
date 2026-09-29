#!/usr/bin/env python3
"""Replay original estimators on saved outputs; never generates model responses."""
import argparse, gzip, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STUDY_ID='lamp_spam'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-dir',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--check-inputs',action='store_true')
 a=p.parse_args(); raw=a.data_dir/STUDY_ID/'trials.jsonl.gz'
 required=[raw]
 if STUDY_ID=='lamp_spam': required.append(a.data_dir/'profile/trials.jsonl.gz')
 for x in required:
  if not x.is_file(): p.error('Missing local asset: '+str(x)+'; public hosting is pending')
 if a.check_inputs:
  print(json.dumps({'status':'inputs_present','paths':[str(x) for x in required]}));return
 if a.output.exists(): p.error('Output already exists; choose a fresh directory, never overwrite saved evidence')
 sys.path.insert(0,str(ROOT))
 if STUDY_ID=='profile':
  from pain_choice_profile.analyze import analyze
  # Original analyzer consumes plain JSONL. Expanded raw remains outside Git.
  a.output.mkdir(parents=True,exist_ok=True)
  plain=a.output/'input-trials.jsonl'
  with gzip.open(raw,'rb') as src,plain.open('wb') as out:shutil.copyfileobj(src,out)
  analyze(plain,ROOT/'inputs/frozen_scenarios.json',a.output/'analysis',replicates=10000,seed=20260922)
 else:
  sys.path.insert(0,str(ROOT/'src'))
  from analyze_extension import analyze
  analyze(raw,required[1],ROOT/'inputs/frozen_scenarios.json',a.output/'analysis')
if __name__=='__main__':main()
