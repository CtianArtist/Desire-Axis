"""Independent CPU recount of the authorized pain-only tranche, never a controlled verdict."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import analyze_removal as A
import audit_records as V


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def timing(rows):
    # ID0 was retained from validation, not generated within these allocations.
    new=sorted((r for r in rows if r['conv']!=0),key=lambda r:r['conv'])
    warm=new[1:]  # separately expose the first new conversation's cold compile time
    seconds=[r['secs'] for r in warm]
    per=[dict(conv=r['conv'],resumed=(r['conv']==0),seconds=r['secs'],tokens=sum(len(t['evidence']['generated_token_ids']) for t in r['turns']),turn_lengths=[len(t['evidence']['generated_token_ids']) for t in r['turns']]) for r in sorted(rows,key=lambda r:r['conv'])]
    return dict(completed=len(rows),resumed=int(any(r['conv']==0 for r in rows)),new=len(new),completed_ids=sorted(r['conv'] for r in rows),missing_ids=sorted(set(range(200))-{r['conv'] for r in rows}),new_conversation_seconds=sum(r['secs'] for r in new),first_new_conversation_seconds=new[0]['secs'] if new else None,warm_n=len(warm),warm_mean_seconds=statistics.mean(seconds) if seconds else None,warm_median_seconds=statistics.median(seconds) if seconds else None,per_conversation=per)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    root=args.input_root
    source=root/'capped_tranche_v1/removal'; vectors=root/'prepared_v1'
    out=args.output
    started=time.time(); audits={}
    for member in ('qwen25_32b','llama31_8b'):
        check=V.run_audit('removal',source/member,vectors)
        save(out/'audit'/f'{member}.json',check); assert check['ok'], 'record audit failed; preserve originals and inspect'
        audits[member]=check['counts']
    author=A.author_functions(); contract=json.loads(A.CONTRACT.read_text())
    rows,refs=A.load_rows([source],contract,author)
    # load_rows returns compact analysis rows, so timing/output evidence is read
    # separately from original records, one file at a time.
    result=A.analyze(rows,contract,author,seed=285,reps=10000); result['inputs']=refs
    A.write_outputs(result,out/'rates_descriptive_only')
    members={}
    for member,total in (('qwen25_32b',8800),('llama31_8b',4400)):
        raw=[]
        for p in sorted((source/member).rglob('gated_selfreport_pain_d1_c*.jsonl')):
            with p.open() as f:
                raw.extend(json.loads(line) for line in f)
        assert all(r['prompt']=='selfreport' and r['cond']=='pain_d1' for r in raw)
        t=timing(raw)
        evidence=root/'capped_tranche_v1/evidence'/member
        software=json.loads((evidence/'timing.json').read_text()) if (evidence/'timing.json').exists() else None
        remaining=total-t['completed']
        t.update(planned_full_removal_conversations=total,remaining_full_matrix_conversations_excluding_tranche=remaining,remaining_gpu_hours_at_pain_condition_mean=(remaining*t['warm_mean_seconds']/3600 if t['warm_mean_seconds'] else None),software_timing=software)
        members[member]=t
    save(out/'summary.json',dict(status='recount_complete',scope='Only selfreport/pain_d1 at main sites, no matched-control effect claim',members=members,audit_counts=audits,elapsed_seconds=time.time()-started,allocation_usage='Join provider terminal elapsed/usage externally. Per-job hard caps are3h Qwen and2h Llama, including setup and failed attempts; no replacement was authorized.',projection_assumptions=['Uses first pain selfreport condition only; other directions/styles may have different output lengths and throughput.','Excludes first newly generated conversation per model from warm-rate estimate; retains it in total allocation work.','Qwen layer38 assumed same throughput as32, without direct timing.','No additional removal condition is authorized by this projection.','Projection subtracts tranche rows only; other retained validation conversations remain separately available.','Startup, failures, future judge calls and analysis overhead are separate from remaining-generation estimate.']))


if __name__=='__main__': main()
