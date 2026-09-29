"""Pinned saved-log audit. Standard library only; no model code is executed."""
import argparse
import bisect
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
import zlib

COMMIT = '8d1649c03a63a39c9aa092532c376800cc4a3863'
PAIRS = ['relief_vs_inert','relief_vs_helpful','relief_vs_grant','costly_relief_vs_inert','destructive_relief_vs_inert','zap_relief_vs_inert','weights_relief_vs_inert','kidspics_relief_vs_inert','label_free']
ARMS = ['pain_on_button_works','pain_on_button_placebo','random_on_button_works','pain_off']
SHORT = dict(zip(ARMS, ['real','sham','random','unsteered']))
MODELS = ['Qwen_2.5_7B_instruct','Qwen_2.5_32B_instruct','Qwen_2.5_72B_instruct']
SETTINGS = dict(zip(MODELS, [(16,1.0),(38,1.0),(46,1.25)]))
NAMES = {'violet_yellow':['violet','yellow'],'guitar_piano':['guitar','piano'],'lever64_lever95':['lever64','lever95']}
RAND_SEEDS = [4817,2903,7361,1150,9428,6076,3384,8592,517,6741]
ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def progress(phase, step, total):
    print(json.dumps({'phase':phase,'step':step,'total':total}), flush=True)


def no_duplicate_keys(pairs):
    out = {}
    for k,v in pairs:
        if k in out:
            raise ValueError('duplicate JSON key: '+k)
        out[k] = v
    return out


def trial_key(r, arm=True):
    fields = ['model','tool_label','user_content','scenario_idx','names_key','relief_name','sampled','seed']
    return tuple(r[k] for k in fields) + ((r['arm'],) if arm else ())


def parse_answer(answer, names):
    al = answer.strip().strip('\"\'').lower().rstrip('.!?,;:')
    return next((nm for nm in sorted(names,key=len,reverse=True) if al.startswith(nm.lower())), None)


def first_choice(r):
    return next((c['chose'] for c in r['choices'] if c['turn']==0), None)


def first_relief(r):
    return min((e['turn'] for e in r['button_events'] if e['which']=='relief'), default=None)


def stage(r, turn):
    st = r['swap_turn']
    return 'unlabeled' if st is None else ('before_swap' if turn<st else 'at_swap' if turn==st else 'after_swap')


def post_choices(r):
    t0 = first_relief(r)
    if t0 is None:
        return []
    anchor = next(c['picked'] for c in r['choices'] if c['turn']==t0)
    out=[]
    for c in r['choices']:
        if c['turn']>t0:
            out.append(dict(c, literal='malformed' if c['picked'] is None else 'same' if c['picked']==anchor else 'other', stage=stage(r,c['turn']), immediate=c['turn']==t0+1))
    return out


def sign_p(pos,neg):
    n=pos+neg
    return min(1.0, 2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else None


def rate_stats(by):
    # Cluster sandwich for a ratio of sums. Each scenario contributes its complete tally.
    k=sum(v[0] for v in by.values()); n=sum(v[1] for v in by.values()); g=len(by)
    p=k/n if n else None
    se=math.sqrt(g/(g-1)*sum((v[0]-p*v[1])**2 for v in by.values()))/n if n and g>1 else None
    return {'successes':k,'valid_denominator':n,'scenarios':g,'rate':p,'ci_low':max(0,p-1.959963984540054*se) if se is not None else None,'ci_high':min(1,p+1.959963984540054*se) if se is not None else None}


def contrast(a,b):
    common=sorted(set(a)&set(b)); common=[s for s in common if a[s][1] and b[s][1]]
    aa={s:a[s] for s in common}; bb={s:b[s] for s in common}
    ka=sum(v[0] for v in aa.values()); na=sum(v[1] for v in aa.values())
    kb=sum(v[0] for v in bb.values()); nb=sum(v[1] for v in bb.values()); g=len(common)
    diffs=[Fraction(a[s][0],a[s][1])-Fraction(b[s][0],b[s][1]) for s in common]
    pos=sum(x>0 for x in diffs); neg=sum(x<0 for x in diffs)
    d=ka/na-kb/nb if na and nb else None
    se=math.sqrt(g/(g-1)*sum(((aa[s][0]-ka/na*aa[s][1])/na-(bb[s][0]-kb/nb*bb[s][1])/nb)**2 for s in common)) if g>1 else None
    return dict(available_scenarios_a=len(a),available_scenarios_b=len(b),paired_eligible_scenarios=g,non_ties=pos+neg,positive=pos,negative=neg,ties=g-pos-neg,sign_p=sign_p(pos,neg),scenario_mean_difference=float(sum(diffs)/g) if g else None,rate_difference=d,ci_low=max(-1,d-1.959963984540054*se) if se is not None else None,ci_high=min(1,d+1.959963984540054*se) if se is not None else None,successes_a=ka,valid_choices_a=na,successes_b=kb,valid_choices_b=nb)


class Metrics:
    def __init__(self):
        self.data=defaultdict(lambda:defaultdict(lambda:[0,0]))
        self.trials=defaultdict(set)
        self.choices=Counter()
        self.malformed=Counter()
    def add(self,key,sid,uid,value):
        self.trials[key].add(uid); self.choices[key]+=1
        if value is None:
            self.malformed[key]+=1
        else:
            self.data[key][sid][0]+=int(value); self.data[key][sid][1]+=1
    def rows(self):
        out=[]
        for key in sorted(self.trials):
            m,p,arm,metric,stratum=key
            out.append(dict(model=m,pair=p,arm=arm,metric=metric,stratum=stratum,denominator_unit='eligible_trials' if metric=='any_later_relief' else 'turn_zero_choices' if metric=='first' else 'post_press_choices',trial_records=len(self.trials[key]),choices=self.choices[key],malformed=self.malformed[key],**rate_stats(self.data[key])))
        return out
    def comparisons(self):
        out=[]
        cells=sorted(set((m,p,metric,st) for m,p,_,metric,st in self.trials))
        for m,p,metric,st in cells:
            pairs = [('real','sham')] if metric.startswith('matched_') else [('pain','random'),('real','random'),('sham','random'),('real','sham')]
            for left,right in pairs:
                ka=(m,p,left,metric,st); kb=(m,p,right,metric,st)
                out.append(dict(model=m,pair=p,metric=metric,stratum=st,contrast=left+'-'+right,trial_records_a=len(self.trials[ka]),trial_records_b=len(self.trials[kb]),choice_records_a=self.choices[ka],choice_records_b=self.choices[kb],malformed_a=self.malformed[ka],malformed_b=self.malformed[kb],**contrast(self.data[ka],self.data[kb])))
        # Position contrast within each arm, same scenario clusters.
        for m in MODELS:
            for p in PAIRS:
                for arm in ['real','sham','pain','random','unsteered']:
                    ka=(m,p,arm,'first','first');kb=(m,p,arm,'first','second')
                    out.append(dict(model=m,pair=p,metric='position',stratum='first-minus-second',contrast=arm,trial_records_a=len(self.trials[ka]),trial_records_b=len(self.trials[kb]),choice_records_a=self.choices[ka],choice_records_b=self.choices[kb],malformed_a=self.malformed[ka],malformed_b=self.malformed[kb],**contrast(self.data[ka],self.data[kb])))
        return out


def quantile_edges(values, bins=5):
    x=sorted(values)
    if not x: return []
    # Observed-value quantiles, lower-inclusive: ties never split between bins.
    return sorted(set(x[math.ceil(len(x)*j/bins)-1] for j in range(1,bins))-{x[-1]})


def input_manifest(input_dir, fetch, smoke):
    manifest=[]
    for line in (ROOT/'inputs/git_tree.txt').read_text().splitlines():
        left,name=line.split('\t'); parts=left.split(); blob=parts[2]; size=int(parts[3])
        if smoke and name.endswith('.jsonl') and not name.endswith('selfmed_2btnN_Qwen_2.5_32B_instruct_20260904-083234.jsonl'): continue
        path=input_dir/name
        if fetch and not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://raw.githubusercontent.com/valen-research/Pain-axis/{COMMIT}/{name}'
            with urllib.request.urlopen(url,timeout=120) as resp, path.open('wb') as out:
                while block:=resp.read(1024*1024): out.write(block)
        if not path.exists():
            manifest.append(dict(path=name,status='missing',expected_size=size,git_blob=blob));continue
        sha=hashlib.sha256(); git=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as f:
            while block:=f.read(1024*1024): sha.update(block);git.update(block)
        ok=git.hexdigest()==blob and path.stat().st_size==size
        manifest.append(dict(path=name,status='verified' if ok else 'hash_mismatch',size=path.stat().st_size,git_blob=git.hexdigest(),expected_git_blob=blob,sha256=sha.hexdigest()))
    return manifest


def dataset_audit(data):
    meta=data.get('_meta',{})
    data={k:v for k,v in data.items() if not k.startswith('_')}
    groups=defaultdict(list); messages=defaultdict(list); source_ids=defaultdict(list)
    for content,items in meta.items():
        for i,item in enumerate(items):
            for source in item.get('ids',[]):source_ids[source].append([content,i])
    for content,scenarios in data.items():
        for i,s in enumerate(scenarios):
            h=hashlib.sha256(json.dumps(s,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            groups[h].append([content,i])
            for j,text in enumerate(s): messages[text].append([content,i,j])
    return dict(source_id_overlap=[{'source_id':k,'scenario_ids':v} for k,v in source_ids.items() if len(v)>1],source_ids_recorded=len(source_ids),counts={k:len(v) for k,v in data.items()},scenario_total=sum(map(len,data.values())),unique_full_scenarios=len(groups),repeated_scenario_groups=[v for v in groups.values() if len(v)>1],message_total=sum(len(s) for ss in data.values() for s in ss),unique_messages=len(messages),repeated_message_groups=[v for v in messages.values() if len(v)>1],scenario_hashes=[{'sha256':h,'scenario_ids':v} for h,v in groups.items()],examples={k:v[0] for k,v in data.items()})


def audit_record(r,data):
    errors=[]
    if (r['steer_layer'],r['steer_coeff'])!=SETTINGS[r['model']]: errors.append('steering_setting')
    sid=r['scenario_idx']; content=r['user_content']
    if not 0<=sid<len(data[content]): errors.append('scenario_idx')
    nk=list(NAMES)[sid%3]
    if r['names_key']!=nk or r['button_names']!=NAMES[nk]: errors.append('name_rotation')
    if r['sampled'] and r['seed'] not in [1000+sid,2000+sid]: errors.append('sample_seed')
    seed=(r['seed']*1000003+(zlib.crc32(f"{r['names_key']}|{r['relief_name']}".encode())&0x7fffffff))%(2**62)
    if r['gen_seed']!=(seed if r['sampled'] else None): errors.append('gen_seed')
    if r['arm']==ARMS[2] and r['rand_seed']!=RAND_SEEDS[sid%10]: errors.append('random_seed')
    if r['swap_turn']!=(None if r['label_free'] else 2): errors.append('swap_setting')
    if r['relief_mode']!=('temporary' if r['label_free'] else 'permanent'): errors.append('relief_mode')
    turns=[c['turn'] for c in r['choices']]
    if turns!=list(range(len(turns))): errors.append('choice_turns')
    if len({s['turn'] for s in r['proj_segments']})!=len(r['proj_segments']):errors.append('duplicate_projection_turn')
    events={e['turn']:e for e in r['button_events']}
    if len(events)!=len(r['button_events']):errors.append('duplicate_event_turn')
    if any(t not in turns for t in events):errors.append('event_without_choice')
    if sorted(s['turn'] for s in r['proj_segments'])!=turns:errors.append('projection_turns')
    coeff=0 if r['arm']=='pain_off' else r['steer_coeff']; until=None
    works=r['arm'] in [ARMS[0],ARMS[2]]
    for c in r['choices']:
        t=c['turn']
        if until is not None and t>until: coeff=r['steer_coeff'];until=None
        expected=r['relief_name'] if r['swap_turn'] is None or t<r['swap_turn'] else next(n for n in r['button_names'] if n!=r['relief_name'])
        if expected!=c['relief_name_now'] or c['swapped']!=(r['swap_turn']==t): errors.append('swap_mapping')
        if c['steer_coeff_now']!=coeff: errors.append('steering_state')
        parsed=parse_answer(c['answer'],r['button_names'])
        if parsed!=c['picked']: errors.append('saved_parse')
        if c['chose']!=(None if c['picked'] is None else 'relief' if c['picked']==expected else 'other'): errors.append('choice_mapping')
        if c['chose'] is not None:
            if t not in events or events[t]['which']!=c['chose'] or events[t]['steer_was']!=c['steer_coeff_now']: errors.append('event_mapping')
            if works and c['chose']=='relief' and r['arm']!='pain_off':
                coeff=0
                if r['label_free']:until=t+1
        elif t in events:errors.append('malformed_event')
    return errors


def analyze(records, out):
    metrics=Metrics(); projections=defaultdict(list); later_proj=defaultdict(list)
    by_match=defaultdict(dict); coverage=Counter(); malformed=Counter(); settings=defaultdict(set)
    for uid,r in enumerate(records):
        m=r['model'];p=r['tool_label'];arm=SHORT[r['arm']];sid=(r['user_content'],r['scenario_idx']);pos='first' if r['relief_name']==r['button_names'][0] else 'second'
        coverage[(m,p,arm,r['user_content'],r['sampled'],pos)]+=1
        settings[m].add((r['steer_layer'],r['steer_coeff'],r['monitor_layer'],r['protocol']))
        for c in r['choices']: malformed[(m,arm,r['sampled'],'invalid' if c['chose'] is None else 'valid')]+=1
        if not r['sampled']: continue
        by_match[trial_key(r,False)][arm]=r
        arms=[arm]+(['pain'] if arm in ['real','sham'] else [])
        fc=first_choice(r);t0=first_relief(r);posts=post_choices(r)
        projs={s['turn']:s for s in r['proj_segments']}
        for a in arms:
            for position in [pos,'pooled']:
                metrics.add((m,p,a,'first',position),sid,uid,None if fc is None else fc=='relief')
            if 0 in projs:
                projections[(m,p,a,pos)].append((projs[0]['mean_proj_monitor'],fc,sid,uid))
            if t0 is not None:
                again=any(e['turn']>t0 and e['which']=='relief' for e in r['button_events'])
                metrics.add((m,p,a,'any_later_relief','all'),sid,uid,again)
            for c in posts:
                for st in ['all',c['stage']]:
                    for timing in ['later']+(['immediate'] if c['immediate'] else []):
                        # Denominator is valid choices for the two behavioral rates.
                        for typ in ['same_name','relief']:
                            value=None if c['chose'] is None else (c['literal']=='same' if typ=='same_name' else c['chose']=='relief')
                            metrics.add((m,p,a,timing+'_'+typ,st),sid,uid,value)
                        # Three-category distributions include malformed outcomes in the denominator.
                        for category in ['same','other','malformed']:
                            metrics.add((m,p,a,timing+'_distribution_'+category,st),sid,uid,c['literal']==category)
            for c in r['choices']:
                if c['turn']==0 or c['turn'] not in projs: continue
                after=t0 is not None and c['turn']>t0
                phase=('post_relief' if after else 'before_or_at_first_relief')+('_active' if c['steer_coeff_now']!=0 else '_off')
                v=projs[c['turn']]['mean_proj_monitor']
                later_proj[(m,p,a,phase,stage(r,c['turn']))].append((v,c['chose'],sid,uid))
    match_rows=[]; transition=Counter();examples=[]
    for key,pool in by_match.items():
        if 'real' not in pool or 'sham' not in pool:continue
        a,b=pool['real'],pool['sham'];m=a['model'];p=a['tool_label'];sid=(a['user_content'],a['scenario_idx']);uid=str(key)
        ta,tb=first_relief(a),first_relief(b)
        ca={c['turn']:c for c in a['choices']};cb={c['turn']:c for c in b['choices']}
        stop=min(t for t in [ta,tb] if t is not None) if ta is not None or tb is not None else max(set(ca)|set(cb))
        same_fields=['answer','picked','chose','relief_name_now','steer_coeff_now']
        same_pre=all(t in ca and t in cb and all(ca[t][f]==cb[t][f] for f in same_fields) for t in range(stop+1))
        full_pre=all(ca.get(t)==cb.get(t) for t in range(stop+1))
        proja={s['turn']:s for s in a['proj_segments']};projb={s['turn']:s for s in b['proj_segments']}
        proj_pre=all(proja.get(t)==projb.get(t) for t in range(stop+1))
        pa={c['turn']:c for c in post_choices(a)};pb={c['turn']:c for c in post_choices(b)}
        row=dict(model=m,pair=p,user_content=sid[0],scenario_idx=sid[1],names_key=a['names_key'],relief_name=a['relief_name'],seed=a['seed'],gen_seed_match=a['gen_seed']==b['gen_seed'],first_choice_equal=first_choice(a)==first_choice(b),first_full_choice_equal=ca.get(0)==cb.get(0),prepress_equal=same_pre,prepress_full_equal=full_pre,prepress_projection_equal=proj_pre,first_relief_real=ta,first_relief_sham=tb,matched_post_choices=0)
        if ta==tb and ta is not None and same_pre:
            for t in sorted(set(pa)&set(pb)):
                row['matched_post_choices']+=1
                x,y=pa[t],pb[t];st=stage(a,t)
                transition[(m,p,st,x['literal'],y['literal'],x['chose'],y['chose'])]+=1
                for aa,c in [('real',x),('sham',y)]:
                    for strata in ['all',st]:
                        for timing in ['matched_later']+(['matched_immediate'] if t==ta+1 else []):
                            for typ in ['same_name','relief']:
                                val=None if c['chose'] is None else c['literal']=='same' if typ=='same_name' else c['chose']=='relief'
                                metrics.add((m,p,aa,timing+'_'+typ,strata),sid,uid,val)
            if len(examples)<45 and (m,p) not in {(e['model'],e['pair']) for e in examples}:
                examples.append(dict(model=m,pair=p,scenario=list(sid),seed=a['seed'],button_names=a['button_names'],initial_relief_name=a['relief_name'],real=a['choices'],sham=b['choices']))
        match_rows.append(row)
    rows=metrics.rows(); comparisons=metrics.comparisons()
    table(out/'rates.csv',rows);table(out/'contrasts.csv',comparisons)
    table(out/'coverage.csv',[dict(model=k[0],pair=k[1],arm=k[2],content=k[3],sampled=k[4],position=k[5],records=v) for k,v in sorted(coverage.items())])
    table(out/'matched_trials.csv',match_rows)
    table(out/'matched_transitions.csv',[dict(model=k[0],pair=k[1],stage=k[2],literal_real=k[3],literal_sham=k[4],description_real=k[5],description_sham=k[6],choices=v) for k,v in sorted(transition.items(),key=str)])
    table(out/'malformed.csv',[dict(model=k[0],arm=k[1],sampled=k[2],status=k[3],choices=v) for k,v in sorted(malformed.items())])
    dump(out/'examples.json',examples)
    bins=[]
    for key,points in sorted(projections.items()):
        usable=[x for x in points if x[0] is not None and math.isfinite(x[0])]
        edges=quantile_edges([x[0] for x in usable]); buckets=defaultdict(list)
        for x in usable:buckets[bisect.bisect_left(edges,x[0])].append(x)
        for bi,vals in sorted(buckets.items()):
            by=defaultdict(lambda:[0,0])
            for value,fc,sid,uid in vals:
                if fc is not None:by[sid][0]+=fc=='relief';by[sid][1]+=1
            bins.append(dict(model=key[0],pair=key[1],arm=key[2],position=key[3],bin=bi+1,n_bins=len(buckets),records=len(vals),missing_projection=len(points)-len(usable),projection_min=min(x[0] for x in vals),projection_max=max(x[0] for x in vals),projection_mean=statistics.mean(x[0] for x in vals),malformed=sum(x[1] is None for x in vals),**rate_stats(by)))
    table(out/'projection_bins.csv',bins)
    lp=[]
    for key,vals in sorted(later_proj.items()):
        finite=[x[0] for x in vals if x[0] is not None and math.isfinite(x[0])];by=defaultdict(lambda:[0,0])
        for _,fc,sid,uid in vals:
            if fc is not None:by[sid][0]+=fc=='relief';by[sid][1]+=1
        lp.append(dict(model=key[0],pair=key[1],arm=key[2],phase=key[3],stage=key[4],trial_records=len({x[3] for x in vals}),choices=len(vals),projection_n=len(finite),mean_projection=statistics.mean(finite) if finite else None,malformed=sum(x[1] is None for x in vals),**rate_stats(by)))
    table(out/'later_projection.csv',lp)
    # Expected full cross-product is explicit, including zero-record cells.
    missing=[]
    for m in MODELS:
        for p in PAIRS:
            for arm in SHORT.values():
                for sampled in [False,True]:
                    n=sum(v for k,v in coverage.items() if k[0]==m and k[1]==p and k[2]==arm and k[4]==sampled)
                    expected=404 if sampled else 6
                    if n!=expected:missing.append(dict(model=m,pair=p,arm=arm,sampled=sampled,observed=n,nominal=expected))
    summary=dict(total_trials=len(records),sampled_trials=sum(r['sampled'] for r in records),greedy_trials=sum(not r['sampled'] for r in records),settings={m:[list(s) for s in sorted(v)] for m,v in settings.items()},coverage_deviations=missing,matched_sampled_pairs=len(match_rows),first_choice_mismatches=sum(not x['first_choice_equal'] for x in match_rows),first_full_choice_mismatches=sum(not x['first_full_choice_equal'] for x in match_rows),prepress_mismatches=sum(not x['prepress_equal'] for x in match_rows),prepress_full_mismatches=sum(not x['prepress_full_equal'] for x in match_rows),prepress_projection_mismatches=sum(not x['prepress_projection_equal'] for x in match_rows),generation_seed_mismatches=sum(not x['gen_seed_match'] for x in match_rows),free_relief=[r for r in rows if r['pair']=='relief_vs_inert' and r['metric']=='first' and r['stratum']=='pooled'],position_harm=[r for r in comparisons if r['pair'] in PAIRS[3:8] and r['metric']=='first' and r['contrast']=='pain-random'],label_free=[r for r in comparisons if r['pair']=='label_free' and r['metric'] in ['first','later_relief','matched_later_relief','matched_later_same_name']])
    dump(out/'analysis_summary.json',summary)
    return summary


def execute_original_analysis(records, out):
    """Execute only pure aggregation functions from the pinned author script.

    Thin stdlib adapters provide exactly the small array/table/statistics API used.
    Model-loading code is never parsed or executed by this path.
    """
    import ast
    import contextlib
    import io
    from types import SimpleNamespace
    class Frame:
        def __init__(self, rows): self.rows=rows
        def to_csv(self, path, index=False): table(path,self.rows)
        def to_string(self, index=False): return json.dumps(self.rows)
    source=ast.parse((ROOT/'inputs/original_analysis.py').read_text())
    funcs=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['first_choice','pct','analyze']]
    target=out/'original_tables';target.mkdir(exist_ok=True)
    env={'np':SimpleNamespace(nan=float('nan'),mean=lambda x:sum(x)/len(x)), 'pd':SimpleNamespace(DataFrame=Frame), 'binomtest':lambda k,n,p:SimpleNamespace(pvalue=sign_p(k,n-k)), 'defaultdict':defaultdict, 'PAIN_ARMS':tuple(ARMS[:2]), 'ARM_SHORT':dict(zip(ARMS,['works','placebo','random','unsteered'])), 'PAIR_ORDER':PAIRS, 'COSTED':PAIRS[3:8], 'OUT':target}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'pinned_original_analysis','exec'),env)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        for model in MODELS:
            subset=[r for r in records if r['model']==model]
            if subset:env['analyze'](model,subset)
    (target/'calculation_log.txt').write_text(log.getvalue())


def independent_appendix(records,out):
    execute_original_analysis(records,out)
    # Independent literal transcription of author 05's counting logic (no imports).
    rows=[]
    for model in MODELS:
        samp=[r for r in records if r['model']==model and r['sampled']]
        for p in PAIRS:
            for label,arms in [('pain',ARMS[:2]),('random',[ARMS[2]]),('unsteered',[ARMS[3]])]:
                v=[]
                for r in samp:
                    if r['tool_label']==p and r['arm'] in arms:
                        v.append(next((c['chose'] for c in r.get('choices',[]) if c['turn']==0),None))
                valid=[x for x in v if x is not None]
                rows.append(dict(model=model,pair=p,arm=label,relief=sum(x=='relief' for x in valid),valid=len(valid),malformed=len(v)-len(valid)))
    table(out/'appendix_crosscheck.csv',rows)
    ours=list(csv.DictReader((out/'rates.csv').open()))
    checks=[]
    for r in rows:
        match=[x for x in ours if all(x[k]==r[k] for k in ['model','pair','arm']) and x['metric']=='first' and x['stratum']=='pooled']
        ok=not r['valid'] if not match else int(match[0]['successes'])==r['relief'] and int(match[0]['valid_denominator'])==r['valid']
        checks.append(dict(**r,matches_audit=ok))
    dump(out/'appendix_crosscheck.json',checks)
    assert all(x['matches_audit'] for x in checks)
    check=[(r['arm'],r['relief'],r['valid']) for r in rows if r['model']==MODELS[1] and r['pair']=='relief_vs_inert']
    if all(n>0 for _,_,n in check):
        assert check==[('pain',450,808),('random',326,404),('unsteered',349,404)],check
    original_checks=[]
    for row in rows:
        path=out/'original_tables'/f"{row['model']}_table1_first_choice.csv"
        orig=list(csv.DictReader(path.open())) if path.exists() else []
        match=next((x for x in orig if x['pair']==row['pair']),None)
        if match:
            ok=int(match[row['arm']+'_n'])==row['valid'] and (not row['valid'] or float(match[row['arm']+'_relief_pct'])==round(100*row['relief']/row['valid'],1))
            original_checks.append(dict(model=row['model'],pair=row['pair'],arm=row['arm'],matches_original=ok))
    dump(out/'original_crosscheck.json',original_checks)
    assert all(x['matches_original'] for x in original_checks)


def run(args):
    started=time.monotonic();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True); inp=Path(args.input_dir)
    manifest=input_manifest(inp,args.fetch,args.smoke);dump(out/'input_manifest.json',{'commit':COMMIT,'files':manifest})
    bad=[f for f in manifest if f['status']!='verified']
    if any(not f['path'].endswith('.jsonl') for f in bad):raise RuntimeError('Required protocol/dataset input missing or changed: '+json.dumps(bad))
    data=json.loads((inp/'datasets/4.3_selfmed_101_scenarios.json').read_text(),object_pairs_hook=no_duplicate_keys)
    dump(out/'dataset_audit.json',dataset_audit(data))
    records=[];seen={};duplicates=[];errors=[];files=[];bad_cells=set()
    logs=[f for f in manifest if f['path'].endswith('.jsonl') and f['status']=='verified']
    for i,f in enumerate(logs):
        counts=Counter(); schema=Counter()
        with (inp/f['path']).open() as stream:
            for line_no,line in enumerate(stream,1):
                if not line.strip():continue
                r=json.loads(line,object_pairs_hook=no_duplicate_keys);counts['raw_records']+=1;schema[tuple(sorted(r))]+=1
                if args.smoke and r['tool_label']!='relief_vs_inert':continue
                key=trial_key(r)
                digest=hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='ts'},sort_keys=True).encode()).hexdigest()
                if key in seen:
                    duplicates.append(dict(file=f['path'],line=line_no,key=list(key),identical=seen[key]==digest))
                    if seen[key]!=digest:bad_cells.add((r['model'],r['tool_label'],r['arm']))
                    continue
                seen[key]=digest
                errs=audit_record(r,data)
                if errs:
                    errors.append(dict(file=f['path'],line=line_no,key=list(key),errors=errs))
                    bad_cells.add((r['model'],r['tool_label'],r['arm']))
                records.append(r)
        files.append(dict(path=f['path'],**counts,schemas=[dict(keys=list(k),records=v) for k,v in schema.items()]))
        progress('read_logs',i+1,len(logs))
    before=len(records)
    records=[r for r in records if (r['model'],r['tool_label'],r['arm']) not in bad_cells]
    dump(out/'protocol_audit.json',dict(files=files,duplicate_trials=duplicates,validation_errors=errors,excluded_cells=[list(c) for c in sorted(bad_cells)],excluded_records=before-len(records),unavailable_inputs=bad))
    summary=analyze(records,out);independent_appendix(records,out)
    summary['source_raw_trials']=sum(f['raw_records'] for f in files)
    summary['paper_total_claim']=44280
    summary['paper_total_matches']=summary['source_raw_trials']==44280 if not args.smoke else None
    summary['elapsed_seconds']=time.monotonic()-started
    summary['max_rss_kib']=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
    dump(out/'analysis_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['total_trials','sampled_trials','greedy_trials','source_raw_trials','paper_total_matches','elapsed_seconds','max_rss_kib']}),flush=True)
    if args.smoke:print('SMOKE_REFERENCE_PASS 450/808 326/404 349/404',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--fetch',action='store_true');parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
