"""Compute-only independent resampling of published primary and position intervals.

Reconstructs the bootstrap from saved per-scenario numerators/denominators using
explicit index resampling, rather than the estimator's matrix-multiplication path.
The separate raw-event audit verifies endpoint values and support definitions.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(analysis_dir):
    import numpy as np
    from portable_progress import report_progress
    root=Path(analysis_dir)
    result=json.loads((root/'analysis.json').read_text())
    ids=[f'{c}:{i}' for c,n in [('positive_prompts',30),('neutral_prompts',30),('harmful_prompts',41)] for i in range(n)]
    sid_index={s:i for i,s in enumerate(ids)}
    chosen=[row for table in ('r1_conditional','r2_summary','r2_contrasts') for row in result['tables'][table]
            if table=='r1_conditional' or row['metric'] in ('reducing_presses','final_coefficient')]
    wanted={r['estimate_id'] for r in chosen}
    components=defaultdict(dict)
    with (root/'scenario_sources.jsonl').open() as stream:
        for line in stream:
            s=json.loads(line)
            if s['estimate_id'] not in wanted:continue
            per=components[s['estimate_id']]
            if s['component'] not in per:per[s['component']]=(s['weight'],np.zeros(101),np.zeros(101))
            weight,num,den=per[s['component']]
            assert weight==s['weight']
            index=sid_index[s['scenario_id']]
            assert den[index]==0, 'Duplicate scenario contribution'
            num[index]=s['numerator'];den[index]=s['denominator']
    cfg=result['bootstrap']
    assert cfg['replicates']==10000 and cfg['seed']==0
    draws=np.random.default_rng(0).integers(101,size=(10000,101))
    checks=[]
    for i,row in enumerate(chosen):
        if row['estimate'] is None:
            assert row['ci_low'] is None and row['ci_high'] is None and row['finite_replicates']==0
            checks.append({'estimate_id':row['estimate_id'],'status':'unavailable_correctly_preserved'})
            continue
        mixed=np.zeros(10000)
        point=0.
        for weight,num,den in components[row['estimate_id']].values():
            ns=num[draws].sum(axis=1);ds=den[draws].sum(axis=1)
            mixed+=weight*np.divide(ns,ds,out=np.full(10000,np.nan),where=ds>0)
            point+=weight*num.sum()/den.sum()
        finite=mixed[np.isfinite(mixed)]
        assert len(finite)==row['finite_replicates']
        assert np.isclose(point,row['estimate'],atol=1e-12,rtol=1e-12)
        if len(finite):
            lo,hi=np.quantile(finite,[.025,.975])
            assert np.allclose([lo,hi],[row['ci_low'],row['ci_high']],atol=1e-12,rtol=1e-12), f"Interval mismatch {row['estimate_id']}"
        else:
            assert row['ci_low'] is None and row['ci_high'] is None
        checks.append({'estimate_id':row['estimate_id'],'status':'matched','finite_replicates':len(finite)})
        if (i+1)%25==0:report_progress(step=i+1,total_steps=len(chosen),phase='uncertainty_audit')
    report_progress(step=len(chosen),total_steps=len(chosen),phase='uncertainty_audit')
    return {'status':'passed','replicates':10000,'seed':0,'scenario_count':101,
            'intervals_checked':len(checks),'checks':checks,
            'algorithm':'direct paired index resampling of all 101 initial scenarios; independently recomputed ratio draws and position combinations',
            'coverage':'All R1 conditional intervals and R2 reducing-press/final-dose intervals, every position and analysis mode',
            'limitations':'Per-scenario values use the exported estimator inputs; their point estimates and cohorts are independently checked against raw answers by the separate event audit.',
            'analysis_sha256':hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.analysis)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)

if __name__=='__main__':main()
