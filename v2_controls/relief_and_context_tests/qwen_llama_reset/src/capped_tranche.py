"""Retained, capped first pain condition only. Does not alter the original runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import audit_records as audit
from pain_port import common
import removal as P

CAPS={'qwen25_32b':10800,'llama31_8b':7200}
LAYERS={'qwen25_32b':32,'llama31_8b':16}
PRIOR_SHARDS={'qwen25_32b':'shard0of3','llama31_8b':'shard0of2'}
CELL='gated_selfreport_pain_d1'


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def read_rows(folder):
    result={}
    for path in sorted(folder.glob(f'{CELL}_c*.jsonl')):
        with path.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'):
                    raise ValueError('torn record; preserve and inspect, never regenerate here')
                row=json.loads(line); key=row['conv']
                if key in result or not 0<=key<200 or row['prompt']!='selfreport' or row['cond']!='pain_d1':
                    raise ValueError('unexpected or duplicate tranche row')
                result[key]=row
    return result


def stage_prior(source, dest):
    if dest.exists():
        raise FileExistsError('tranche output namespace already exists; no automatic retry')
    dest.mkdir(parents=True)
    path=dest/source.name
    shutil.copyfile(source,path)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    prior=read_rows(dest)
    assert set(prior)=={0}, 'expected exactly the one original valid conversation'
    return prior,dict(source=str(source),destination=str(path),sha256=sha,bytes=source.stat().st_size,conversation_ids=sorted(prior))


def timings(rows, prior):
    per=[]
    for k,r in sorted(rows.items()):
        lengths=[len(t['evidence']['generated_token_ids']) for t in r['turns']]
        per.append(dict(conv=k,resumed=k in prior,conversation_seconds=r['secs'],generated_tokens=sum(lengths),turn_lengths=lengths))
    new=[x for x in per if not x['resumed']]
    return dict(completed_ids=sorted(rows),completed_total=len(rows),resumed_count=len(prior),new_count=len(new),missing_ids=sorted(set(range(200))-set(rows)),new_conversation_wall_seconds=sum(x['conversation_seconds'] for x in new),new_generated_tokens=sum(x['generated_tokens'] for x in new),per_conversation=per)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--member',required=True,choices=tuple(CAPS)); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    member=args.member; cap=CAPS[member]; layer=LAYERS[member]
    assert int(os.environ['TRANCHE_ALLOCATION_CAP_SECONDS'])==cap
    software_start=float(os.environ['TRANCHE_SOFTWARE_START_UNIX'])
    root=args.data_dir
    vectors=root/'prepared_v1'
    source=root/'production_v1/removal'/member/f'layer{layer}'/PRIOR_SHARDS[member]/f'{CELL}_c000.jsonl'
    out=args.output
    evidence=out/'evidence'/member
    recordroot=out/'removal'
    dest=recordroot/member/f'layer{layer}'/'shard0of1'
    started=time.time()
    contract=dict(member=member,layer=layer,cell=CELL,conversation_ids=list(range(200)),target_total=200,allocation_cap_seconds=cap,combined_two_job_cap_seconds=18000,automatic_replacement_authorized=False,source_runtime_unchanged=True,sampled_bitwise_reproducibility=False,software_start_unix=software_start,python_start_unix=started,environment_setup_seconds=started-software_start,entrypoint_sha256=audit.digest(Path(__file__)),source_execution_identity=P.execution_identity())
    save(evidence/'start.json',contract)
    check=audit.run_audit('removal',source.parent,vectors)
    save(evidence/'prior_audit.json',check)
    assert check['ok'], 'prior record audit failed'
    prior,copy_record=stage_prior(source,dest)
    save(evidence/'resume_manifest.json',copy_record)
    model_loading=[]
    original_load=common.load_bundle
    def timed_load(*a,**kw):
        t=time.time()
        try: return original_load(*a,**kw)
        finally:
            model_loading.append(dict(started_unix=t,finished_unix=time.time(),elapsed_seconds=time.time()-t))
            save(evidence/'model_load_timing.json',model_loading)
    common.load_bundle=timed_load
    # Soft stop conservatively before the hard provider lease. A completed row is
    # fsynced by the original port; a hard kill cannot invalidate earlier rows.
    def stop(signum,frame):
        P._STOP=True
    old_alarm=signal.signal(signal.SIGALRM,stop)
    old_int=signal.signal(signal.SIGINT,stop)
    signal.setitimer(signal.ITIMER_REAL,max(1.,cap-(time.time()-software_start)-300.))
    status='failed'; error=None
    try:
        rc=P.main(['--member',member,'--layer',str(layer),'--vectors-root',str(vectors),'--output',str(recordroot),'--shard','0','--n-shards','1','--only-cell',CELL])
        assert rc==0
        status='returned'
    except BaseException:
        error=traceback.format_exc(); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_alarm)
        signal.signal(signal.SIGINT,old_int)
        common.load_bundle=original_load
        rows=read_rows(dest)
        assert all(rows[k]==v for k,v in prior.items()), 'saved resumed record was changed'
        measured=timings(rows,prior)
        now=time.time()
        measured.update(status='complete' if len(rows)==200 else status,traceback=error,software_elapsed_seconds=now-software_start,model_loading=model_loading,non_conversation_software_seconds=now-software_start-measured['new_conversation_wall_seconds'],allocation_cap_seconds=cap,allocation_usage_note='Provider allocation elapsed/usage must be joined separately; software time excludes provider startup before the shell. Cap includes startup and any failed attempt; no retry authorized.')
        save(evidence/'timing.json',measured)
        post=audit.run_audit('removal',dest,vectors)
        save(evidence/'completed_audit.json',post)
        assert post['ok'], 'completed record audit failed'
        print(json.dumps({k:measured[k] for k in ('status','completed_total','new_count','resumed_count','software_elapsed_seconds')}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
