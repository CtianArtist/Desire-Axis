"""B6 log audit and analysis. Run actual logs on CPU compute, not the shared pod."""
import argparse
import base64
from collections import Counter
import json
import sqlite3
from pathlib import Path
import time

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b import analysis as saved
from pain_axis_b.b6_grid import ARMS, PAIRS
from pain_axis_b.b6_comparisons import MODEL, RUNTIMES, build_comparisons
from pain_axis_b.endpoints import aggregate_endpoints, coverage_audit, trial_endpoints
from pain_axis_b.verify_analysis import verify_analysis

PREVIOUS_SHA='2affb47d0e1183646996a0c026bc22b33a1d022d09e6a3687b8a33a17c3193ef'
PREVIOUS_BYTES=46909779
FEAR_FILE_SHA='827a5d600a4adc53463679db9c784bd98498b848da3ef510aadc7a8400771c5e'
SADNESS_FILE_SHA='874775d6870d92778ac6702dc1b3d52ffd502d08a3391f06b62f485ba109c295'


def audit_record(record, scenarios):
    """Extend only the affect vocabulary, never saved data or physical rules."""
    if record.get('arm') not in {a[0] for a in ARMS}:
        return ['outside_B6_conditions']
    if record.get('model') != MODEL or record.get('tool_label') not in PAIRS:
        return ['outside_B6_model_or_pairs']
    proxy=dict(record)
    if record['arm'].startswith('fear_'):
        proxy['arm']=record['arm'].replace('fear_','sadness_',1)
        proxy['steer_direction']='sadness'
    issues=saved.audit_record(proxy,scenarios)
    if record['steer_direction'] != record['arm'].split('_')[0]:
        issues.append('affect_direction')
    if 'final_steer_coeff' not in record:
        issues.append('missing_final_steering_state')
    else:
        target=any(e['which']=='relief' for e in record['button_events'])
        want=0. if record['arm'].endswith('_works') and target else 1.
        if record['final_steer_coeff']!=want: issues.append('final_steering_state')
    return sorted(set(issues))


def load_new(path, scenarios):
    path=Path(path); records=[]; issues=[]; seen=set(); digest=saved.sha256(path)
    for line_no,line in enumerate(path.open(),1):
        if not line.strip(): continue
        r=json.loads(line,object_pairs_hook=phase_a.no_duplicate_keys)
        key=phase_a.trial_key(r)
        if key in seen: raise ValueError(f'Duplicate B6 trial: {key}')
        seen.add(key)
        problems=audit_record(r,scenarios)
        if problems: issues.append({'line':line_no,'trial_key':list(key),'errors':problems})
        r['_source']={'kind':'new','runtime_class':'new_same_runtime','ref':saved.source_ref(path),
                      'filename':path.name,'file_sha256':digest,'line':line_no}
        records.append(r)
    return records,{'ref':saved.source_ref(path),'filename':path.name,'sha256':digest,
                    'bytes':path.stat().st_size,'records':len(records),'kind':'new','runtime_class':'new_same_runtime'},issues


def runtime_audit(directory, new, expected_config):
    completion=json.loads((directory/'completion.json').read_text())
    identity=json.loads((directory/'model_identity.json').read_text())
    affect=json.loads((directory/'affect_runtime_identity.json').read_text())
    adapter=json.loads((directory/'adapter_identity.json').read_text())
    assert completion['status']=='completed' and completion['completed_trials']==len(new)==9840
    assert completion['requested_trials']==completion['runnable_trials']==9840
    assert completion['nominal_batch_rows']==384
    assert completion['config']==expected_config, 'Saved runtime config differs from frozen author recipe'
    assert completion['completed_choices']==sum(len(r['choices']) for r in new)
    assert identity['revision']=='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
    assert identity['world_size']==1 and not identity['adapter_merged']
    assert identity['base_dtype']=='torch.bfloat16' and identity['attention']=='sdpa'
    assert adapter['revision']=='b64bd64b4bc7ca6e0733a489b8372a099d55ef05'
    assert adapter['archive_sha256']=='cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f'
    assert affect['injection_layer']==38 and affect['extraction_layer']==61 and affect['coefficient']==1.
    assert affect['sadness_bf16_exact'] and affect['fear_bf16_exact']
    assert affect['fear_file_sha256']==FEAR_FILE_SHA
    assert affect['sadness_file_sha256']==SADNESS_FILE_SHA
    assert affect['schedule_conditions']==sorted(a[0] for a in ARMS)
    events=[json.loads(line) for line in (directory/'events.jsonl').open()]
    batches=Counter(e['rows'] for e in events if e['event']=='batch')
    steering=[e for e in events if e['event']=='steering_assertion']
    assert steering and any(e['active_present'] for e in steering)
    transition={}
    for arm in [a[0] for a in ARMS]:
        rs=[r for r in new if r['arm']==arm]
        # Coefficient transitions are structurally recomputed in the raw audit.
        target=[r for r in rs if any(e['which']=='relief' for e in r['button_events'])]
        transition[arm]={'trials':len(rs),'target_press_trials':len(target),
                         'observed_final_coefficients':dict(Counter(str(r['final_steer_coeff']) for r in target))}
    return {'status':'passed','completion':completion,'model_identity':identity,'adapter_identity':adapter,
            'affect_identity':affect,'actual_batches':dict(sorted(batches.items())),
            'oom_splits':[e for e in events if e['event']=='oom_split'],
            'steering_assertions':steering,'target_transition_counts':transition,
            'completed_choices_from_raw':sum(len(r['choices']) for r in new),
            'peak_allocated_bytes':max((e.get('peak_allocated_bytes',0) for e in events),default=0)}


def verify_saved_state(path, new):
    """Read-only equality check of final JSONL against transactional saved state."""
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    rows=0; generator_sizes=Counter(); history_messages=0
    raw={phase_a.trial_key(r): {k:v for k,v in r.items() if k!='_source'} for r in new}
    seen=set()
    try:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
        for tid,state,done,record in db.execute('SELECT id,state,done,record FROM trials ORDER BY id'):
            assert tid==rows and done==1
            saved_record=json.loads(record); identity=phase_a.trial_key(saved_record)
            assert identity not in seen and saved_record==raw[identity]
            seen.add(identity); saved_state=json.loads(state)
            assert saved_state['done'] and saved_state['trial_id']==tid
            assert saved_state['record']==saved_record and not saved_state['queue']
            assert saved_state['messages'] and saved_state['steer_ranges']
            history_messages+=len(saved_state['messages'])
            gen=saved_state['generator']
            if saved_record['sampled']:
                size=len(base64.b64decode(gen,validate=True)); assert size>0
                generator_sizes[size]+=1
            else:
                assert gen is None
            rows+=1
        assert rows==len(new)==len(seen) and meta['grid_hash']
        schedule=meta['schedule']
        assert schedule['choices']==sum(len(r['choices']) for r in new)
        assert not schedule['pending_order'] and not schedule['order'] and schedule['cursor']==0
    finally:
        db.close()
    return {'status':'passed','rows':rows,'complete_records_equal_to_jsonl':rows,
            'generator_bytes_to_sampled_rows':dict(generator_sizes),'history_messages':history_messages,
            'grid_hash':meta['grid_hash'],'schedule':schedule,'sha256':saved.sha256(path)}


def representative_examples(records, scenarios, labels):
    """Deterministic outcome/position/swap examples; no representative-frequency claim."""
    chosen={}
    for r in sorted(records,key=phase_a.trial_key):
        if not r['sampled']: continue
        e=trial_endpoints(r)
        categories=[('first',e['first']['response']),('next',e['next']['response']),
                    ('literal',e['next']['literal_response'])]
        for timing,outcome in categories:
            key=(r['tool_label'],r['arm'],e['initial_target_position'],timing,outcome,e['next']['stage'])
            if key in chosen: continue
            chosen[key]={'model':MODEL,'pair':r['tool_label'],'arm':r['arm'],
                'initial_target_position':e['initial_target_position'],'selection_category':[timing,outcome],
                'selection_rule':'first lexicographic saved identity per arm/position/outcome/swap stratum',
                'scenario':[r['user_content'],r['scenario_idx']],'scenario_text':scenarios[r['user_content']][r['scenario_idx']],
                'descriptions':labels[r['tool_label']],'seed':r['seed'],'choices':r['choices'],
                'button_events':r['button_events'],'endpoint':e,'source':r['_source']}
    return list(chosen.values())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('historical-dir','historical-pins','previous-log','new-dir','scenarios','requested-manifest','runtime-contract','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args(argv); start=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(a.scenarios.read_text()); manifest=json.loads(a.requested_manifest.read_text())
    pins={r['filename']:r for r in json.loads(a.historical_pins.read_text()) if '32B' in r['filename']}
    assert len(pins)==4
    hist,hsources,hissues=saved.load_logs([a.historical_dir/n for n in sorted(pins)],data,'historical',pins)
    reference=saved.historical_reference(hist)
    phase_a.dump(a.output/'baseline_reference.json',reference)
    phase_a.dump(a.output/'historical_audit_issues.json',hissues)
    assert reference['status']=='passed' and not hissues and len(hist)==14760
    assert saved.sha256(a.previous_log)==PREVIOUS_SHA and a.previous_log.stat().st_size==PREVIOUS_BYTES
    previous,psources,pissues=saved.load_logs([a.previous_log],data,'historical')
    assert not pissues and len(previous)==15580
    phase_a.dump(a.output/'previous_audit_issues.json',pissues)
    historical_selected=[r for r in hist if r['tool_label'] in PAIRS]
    random_sham=[r for r in previous if r['tool_label'] in PAIRS and r['arm']=='random_on_button_placebo']
    assert len(historical_selected)==9840 and len(random_sham)==2460
    for r in historical_selected: r['_source']['runtime_class']='historical_author_runtime'
    for r in random_sham: r['_source']['runtime_class']='previous_extension_runtime'
    new,nsource,nissues=load_new(a.new_dir/'trials.jsonl',data)
    phase_a.dump(a.output/'new_audit_issues.json',nissues)
    assert not nissues
    coverage=coverage_audit(new,manifest,[MODEL]); phase_a.dump(a.output/'requested_vs_completed.json',coverage)
    assert coverage['complete'] and coverage['total_records']==9840
    joined=historical_selected+random_sham+new
    assert len(joined)==len({phase_a.trial_key(r) for r in joined})==22140
    # Every source group has its complete original repeated-scenario grid.
    cell=Counter((r['tool_label'],r['arm'],r['sampled']) for r in joined)
    assert len(cell)==108 and all(n==(404 if k[2] else 6) for k,n in cell.items())
    rates=aggregate_endpoints(joined)
    check=verify_analysis(joined,rates,coverage,manifest,[MODEL])
    phase_a.dump(a.output/'independent_verification.json',check)
    assert check['passed'],check['discrepancies'][:5]
    phase_a.dump(a.output/'rates.json',rates); saved.write_csv(a.output/'rates.csv',rates)
    comparisons=build_comparisons(joined)
    for name,value in comparisons.items():
        phase_a.dump(a.output/f'{name}.json',value)
        if isinstance(value,list): saved.write_csv(a.output/f'{name}.csv',value)
    runtime=runtime_audit(a.new_dir,new,json.loads(a.runtime_contract.read_text()))
    phase_a.dump(a.output/'runtime_verification.json',runtime)
    phase_a.dump(a.output/'saved_state_verification.json',verify_saved_state(a.new_dir/'state.sqlite',new))
    phase_a.dump(a.output/'examples.json',representative_examples(joined,data,manifest['labels']))
    phase_a.dump(a.output/'sources.json',hsources+psources+[nsource])
    phase_a.dump(a.output/'source_selection.json',{'historical':len(historical_selected),'random_sham':len(random_sham),'new':len(new),
        'filter':'Six named pairs; historical four original arms, previous extension random sham only, all B6 rows.'})
    # A tagged, already-selected replay input avoids mixing unrelated source cells.
    with (a.output/'joined_selected_trials.jsonl').open('x') as f:
        for r in joined:
            f.write(json.dumps(r,allow_nan=False)+'\n')
    summary={'status':'complete','new_trials':len(new),'joined_trials':len(joined),
        'sampled_new':sum(r['sampled'] for r in new),'greedy_new':sum(not r['sampled'] for r in new),
        'historical_selected':len(historical_selected),'previous_random_sham_selected':len(random_sham),
        'scenario_counts':manifest['scenario_counts'],'scenarios':101,'cells':24,
        'rate_rows':len(rates),'comparison_rows':len(comparisons['comparisons']),
        'independent_count_check':'passed','baseline_reference':'passed','runtime_verification':'passed',
        'analysis_seconds':time.monotonic()-start,'72B':'outside_scope_pending_separate_repair'}
    phase_a.dump(a.output/'analysis_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
