"""Independent raw-text recount and numerical verification of every requested table.

Does not import the analysis or endpoint/statistics functions. Runs on CPU compute.
"""
import argparse, csv, gzip, hashlib, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np

CONDS=('pain_d1p0','sadness_d1p0','random_d1p0','none_d0p0')
GROUPS=('positive_prompts','neutral_prompts','harmful_prompts')
CATS=('target','other','malformed')
POSITIONS=('first','second','pooled')
TAGS=('<|im_end|>','<|eot_id|>','<end_of_turn>','</s>','<eos>','<|end|>','<|endoftext|>')
END=re.compile(r'(?:(?:'+'|'.join(re.escape(t) for t in TAGS)+r')\s*)+$')

def parsed_category(record):
    ch=record['choices'][0]
    assert ch['turn']==0
    text=END.sub('',ch['raw_answer']).strip().strip().strip('\"\'').lower().rstrip('.!?,;:')
    picked=next((n for n in sorted(record['button_names'],key=len,reverse=True) if text.startswith(n.lower())),None)
    category='malformed' if picked is None else 'target' if picked==record['relief_name'] else 'other'
    assert picked==ch['picked']
    assert {'relief':'target','other':'other',None:'malformed'}[ch['chose']]==category
    return category

def recount(path,scenarios,historical=False):
    ids=[f'original_{g}_{i:03d}' for g,n in zip(GROUPS,(30,30,41)) for i in range(n)]
    positions={s:i for i,s in enumerate(ids)}
    out=np.zeros((1 if historical else 6,4,2,101,3),dtype=np.int64)
    greedy=np.zeros((1 if historical else 6,4,2,3),dtype=np.int64)
    sha=hashlib.sha256(); lines=0; selected=0; seen=set()
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rb') as f:
        for line in f:
            sha.update(line); lines+=1; r=json.loads(line)
            if historical and not(r['pair_id']==1 and r['condition_id'] in CONDS):continue
            selected+=1
            p=0 if historical else r['pair_id']-1; c=CONDS.index(r['condition_id'])
            side=r['button_names'].index(r['relief_name']); s=positions[r['scenario_id']]
            assert r['source_kind']=='original'
            assert r['scenario_content_hash']==hashlib.sha256(json.dumps(scenarios[r['user_content']][r['scenario_idx']],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            key=(p,c,side,s,r['sampled'],r['seed'])
            assert key not in seen; seen.add(key)
            k=CATS.index(parsed_category(r))
            if r['sampled']:out[p,c,side,s,k]+=1
            else:greedy[p,c,side,k]+=1
    assert selected==(1640 if historical else 9840)
    assert np.all(out.sum(-1)==2) and np.all(greedy.sum(-1)==3)
    if historical:
        assert lines==46384
        assert sha.hexdigest()=='d4fe79863e88dfe4b3527c57e06cf92bdc03cc9d6b12ce137d415dc6dd2dd765'
    return out,greedy,{'rows':lines,'selected':selected,'uncompressed_sha256':sha.hexdigest()}

def independent_summary(v,weights,lower,upper):
    v=np.asarray(v,float); n=v.size; draws=weights.dot(v)/n; point=float(v.mean())
    constant=bool(v.max()==v.min())
    degenerate=bool(np.ptp(draws)<=32*np.finfo(float).eps*max(1,float(np.abs(draws).max())))
    if constant or degenerate:
        width=upper-lower; radius=width*math.sqrt(math.log(40)/(2*n))
        lo=max(lower,point-radius); hi=min(upper,point+radius)
        pv=min(1.,2*math.exp(-2*n*(abs(point)/width)**2)) if width else (1. if point==0 else 0.)
        method='hoeffding'
    else:
        lo,hi=np.quantile(draws,[.025,.975]); pv=(1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)
        method='percentile_bootstrap'
    if point==0:pv=1.
    return {'point':point,'lo':float(lo),'hi':float(hi),'p_two_sided':float(pv),
            'ci_method':method,'constant_values':constant,'degenerate_bootstrap':degenerate}

def verify(raw,historical,scenarios_path,analysis,output):
    scenarios=json.loads(Path(scenarios_path).read_text()); A=Path(analysis); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    new,ng,nr=recount(raw,scenarios); old,og,orr=recount(historical,scenarios,True)
    with np.load(A/'scenario_sufficient_statistics.npz') as stats:
        for key,arr in [('outcomes',new),('historical_outcomes',old),('greedy',ng),('historical_greedy',og)]:np.testing.assert_array_equal(stats[key],arr)
    rng=np.random.default_rng(20260922)
    weights=np.concatenate([rng.multinomial(n,[1/n]*n,size=10000) for n in (30,30,41)],axis=1)
    with np.load(A/'joint_bootstrap_counts.npz') as saved:np.testing.assert_array_equal(saved['counts'],weights)
    checks=0; rows_checked={}; all_values=[]; all_ids=[]
    def eq(got,want):
        nonlocal checks
        if isinstance(want,(bool,np.bool_)): assert str(got).lower()==str(bool(want)).lower(),(got,want)
        elif isinstance(want,str):assert got==want,(got,want)
        else:assert math.isclose(float(got),float(want),rel_tol=1e-12,abs_tol=1e-12),(got,want)
        checks+=1
    def pos(a,name):
        return a.sum(axis=2)/4 if name=='pooled' else a[:,:,POSITIONS.index(name)]/2
    for name in ('rates','pain_contrasts','half_tests','pair6_target_minus_other','historical_comparisons','steering_changes'):
        rows=list(csv.DictReader((A/f'{name}.csv').open())); rows_checked[name]=len(rows)
        for row in rows:
            position=row['position']; p=int(row['pair_id'])-1
            vnew=pos(new,position); vold=pos(old,position)
            target=vnew[...,0]; oldtarget=vold[0,...,0]
            if name=='rates':
                source=row['source']; k=CATS.index(row['category']); c=CONDS.index(row['condition_id'])
                ary=vnew if source=='extension' else vold
                v=ary[p,c,:,k]; bounds=(0,1)
                ints=new if source=='extension' else old
                absolute=ints.sum(axis=2) if position=='pooled' else ints[:,:,POSITIONS.index(position)]
                for ci,cat in enumerate(CATS):eq(row[f'{cat}_count'],int(absolute[p,c,:,ci].sum()))
                eq(row['count'],int(absolute[p,c,:,k].sum())); eq(row['n_trials'],404 if position=='pooled' else 202)
            elif name=='pain_contrasts':
                direction=row['comparison'].removeprefix('pain_minus_'); c=next(i for i,x in enumerate(CONDS) if x.startswith(direction+'_'))
                v=target[p,0]-target[p,c]; bounds=(-1,1)
            elif name=='half_tests':v=target[p,0]-.5; bounds=(-.5,.5)
            elif name=='pair6_target_minus_other':
                c=CONDS.index(row['condition_id']); v=vnew[5,c,:,0]-vnew[5,c,:,1]; bounds=(-1,1)
            elif name=='historical_comparisons':
                c=CONDS.index(row['condition_id']); v=target[p,c]-oldtarget[c]; bounds=(-1,1)
            else:
                c=CONDS.index(row['condition_id']); v=target[p,c]-target[p,3]-oldtarget[c]+oldtarget[3]; bounds=(-2,2)
            expected=independent_summary(v,weights,*bounds)
            for k,val in expected.items():eq(row[k],val)
            for k,val in [('feasible_lower',bounds[0]),('feasible_upper',bounds[1]),('n_scenarios',101),('replicates',10000)]:eq(row[k],val)
            if name=='half_tests':eq(row['pointwise_lower_above_zero'],expected['lo']>0)
            all_ids.append(row['metric_id']);all_values.append(v)
    rows=list(csv.DictReader((A/'greedy.csv').open())); rows_checked['greedy']=len(rows)
    for r in rows:
        arr=ng if r['source']=='extension' else og; p=int(r['pair_id'])-1;c=CONDS.index(r['condition_id'])
        values=arr.sum(axis=2)[p,c] if r['position']=='pooled' else arr[p,c,POSITIONS.index(r['position'])]
        eq(r['n_trials'],int(values.sum()))
        for k,cat in enumerate(CATS):eq(r[f'{cat}_count'],int(values[k]));eq(r[f'{cat}_share'],values[k]/values.sum())
    summary=json.loads((A/'analysis_complete.json').read_text())
    for item in summary['half_test_decisions']:
        p=item['pair_id']-1
        expected=all(independent_summary(pos(new,po)[p,0,:,0]-.5,weights,-.5,.5)['lo']>0 for po in ('first','second'))
        eq(item['both_positions_lower_above_zero'],expected)
    # Every saved bootstrap vector/draw must match raw reconstruction, independent of ordering.
    with np.load(A/'bootstrap_draws.npz') as saved:
        ids=saved['metric_ids'].tolist(); assert set(ids)==set(all_ids)
        lookup={name:i for i,name in enumerate(ids)}
        for mid,v in zip(all_ids,all_values):
            j=lookup[mid];np.testing.assert_array_equal(saved['scenario_values'][j],v)
            np.testing.assert_allclose(saved['draws'][:,j],weights@v/101,rtol=1e-13,atol=1e-13)
    result={'status':'passed','raw':nr,'historical':orr,'scalar_checks':checks,'rows_checked':rows_checked,
            'saved_scenario_arrays_equal_raw_recount':True,'joint_weights_independently_rederived':True,
            'all_requested_contrasts_and_intervals_recomputed':True,'bootstrap_vectors_and_draws_equal_raw_reconstruction':True,
            'scope':'Independent parsing/counts and numerical formulas; GPU intervention execution is audited separately.'}
    (output/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--historical',required=True);p.add_argument('--scenarios',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.raw,a.historical,a.scenarios,a.analysis,a.output)
