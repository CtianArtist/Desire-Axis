"""Validate frozen tables or replay the ORIGINAL saved-log analyses on local assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=['b1_b3', 'b6', 'all'], default='all')
    p.add_argument('--data-dir', type=Path, help='Local assets in the README layout; no automatic download')
    p.add_argument('--output', type=Path, help='New output directory, required with --data-dir')
    a = p.parse_args(argv)
    studies = ['b1_b3', 'b6'] if a.study == 'all' else [a.study]
    checks = []
    for study in studies:
        root = ROOT / study
        provenance = json.loads((root / 'provenance.json').read_text())
        for item in provenance['files']:
            path = root / item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Frozen file changed: ' + item['path'])
        summary = json.loads((root/'tables/analysis_summary.json').read_text())
        expected = 31160 if study == 'b1_b3' else 9840
        assert summary['new_trials'] == expected
        rates = json.loads((root/'tables/rates.json').read_text())
        assert len(rates) == summary['rate_rows']
        assert {r['initial_target_position'] for r in rates} == {'first', 'second', 'pooled'}
        for row in rates:
            n, k = row['valid_denominator'], row['successes']
            assert 0 <= k <= n
            if n: assert abs(row['rate'] - k/n) < 1e-12
            else: assert row['rate'] is None
        checks.append({'study':study, 'source_files_checked':len(provenance['files']),
                       'new_trials':expected, 'rate_rows':len(rates), 'status':'passed'})
        if a.data_dir:
            if not a.output: p.error('--output is required with --data-dir')
            data = a.data_dir.resolve()
            common = ['--historical-dir', str(data/'historical'), '--historical-pins', str(root/'inputs/historical_input_hashes.json'),
                      '--scenarios', str(root/'inputs/scenarios.json'), '--requested-manifest', str(root/'inputs/requested_grid.json'),
                      '--output', str(a.output.resolve()/study)]
            pins = json.loads((root/'inputs/historical_input_hashes.json').read_text())
            required = [data/'historical'/x['filename'] for x in pins if study=='b1_b3' or '32B' in x['filename']]
            if study == 'b1_b3':
                logs = [data/'b1_32/trials.jsonl', data/'b1_72_final/trials.jsonl']
                required += logs
                args = ['-m', 'pain_axis_b.analysis', *common]
                for log in logs: args += ['--new-log', str(log)]
            else:
                required += [data/'b1_32/trials.jsonl'] + [data/'b6_32'/n for n in ['trials.jsonl','completion.json','model_identity.json','affect_runtime_identity.json','adapter_identity.json','events.jsonl','state.sqlite']]
                args = ['-m','pain_axis_b.b6_analysis',*common,'--previous-log',str(data/'b1_32/trials.jsonl'),
                        '--new-dir',str(data/'b6_32'),'--runtime-contract',str(root/'inputs/runtime_contract.json')]
            missing = [str(x.relative_to(data)) for x in required if not x.is_file()]
            if missing: p.error('Required local assets missing (public release pending): ' + ', '.join(missing))
            asset_pins = json.loads((root/'assets.json').read_text())['required_inputs']
            for path in required:
                pin = asset_pins[str(path.relative_to(data))]
                assert path.stat().st_size == pin['bytes'], 'Input size mismatch: '+path.name
                if pin['sha256']:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(1 << 20),b''): digest.update(block)
                    assert digest.hexdigest() == pin['sha256'], 'Input hash mismatch: '+path.name
            subprocess.run([sys.executable,*args],cwd=root,check=True)
            # Compare the exact original count/interval outputs, not rounded paper text.
            output = a.output.resolve()/study
            actual = json.loads((output/'rates.json').read_text())
            assert actual == rates, 'Original analysis rates differ from frozen release'
            names = [('b1','b1_display'),('b2','b2_display'),('b3','b3_display')] if study=='b1_b3' else []
            def display(x):
                # Original report display packaging omitted repeated scenario-ID lists only.
                if isinstance(x,dict): return {k:display(v) for k,v in x.items() if k != 'common_scenario_ids'}
                if isinstance(x,list): return [display(v) for v in x]
                return x
            for generated,frozen in names:
                assert display(json.loads((output/(generated+'.json')).read_text())) == json.loads((root/'tables'/(frozen+'.json')).read_text())
            if study == 'b6':
                # The original B6 display release removes these redundant detail arrays,
                # restores term_counts verbatim, and attaches the source row index.
                excluded = {'common_scenario_ids','terms','term_counts','per_pair_rates','per_pair_response_counts'}
                def compact(rows):
                    return [dict({k:v for k,v in row.items() if k not in excluded},
                                 source_row=i,term_counts=row['term_counts']) for i,row in enumerate(rows)]
                rows = compact(json.loads((output/'comparisons.json').read_text()))
                for metric in ['first_target','next_target','next_same_name','next_switch','any_later_target']:
                    assert [r for r in rows if r['metric']==metric] == json.loads((root/'tables'/(metric+'.json')).read_text())
                assert compact(json.loads((output/'paper_first_choice.json').read_text())) == json.loads((root/'tables/paper_first_choice_compact.json').read_text())
    print(json.dumps({'status':'passed','mode':'original_analysis_replay' if a.data_dir else 'frozen_table_integrity_only','checks':checks},indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
