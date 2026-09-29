"""Replay saved statistics without resampling, model loading or judging.
--data-dir accepts an unpacked saved-results asset; omit it to verify included tables.
This is a deterministic table replay, not an independent raw-record bootstrap.
"""
import argparse, csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def flatten(obj,prefix=''):
    out={}
    for k,v in obj.items():
        key=prefix+k
        if isinstance(v,dict): out.update(flatten(v,key+'.'))
        elif isinstance(v,list): out[key]=json.dumps(v,separators=(',',':'))
        else: out[key]=v
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); config=json.loads((ROOT/'config.json').read_text()); olmo=config['group']=='olmo_reset'
    frozen=ROOT/'results'/('frozen_analysis.json' if olmo else 'frozen_readout.json')
    saved=json.loads(frozen.read_text())
    if a.data_dir:
        path=a.data_dir/('final_review_v1/analysis/analysis.json' if olmo else 'tranche_readout_v1/readout.json')
        external=json.loads(path.read_text())
        if olmo:
            for key,cell in saved['cells'].items():
                for field in ('primary','n_observed','coverage_complete'): assert cell[field]==external['cells'][key][field],(key,field)
            assert saved['contrasts']==external['contrasts']
        else:
            assert saved['observed_cells']==external['observed_cells']
            assert saved['registered_decisions']==external['registered_decisions']
    if olmo:
        cells=list(saved['cells'].values()); assert len(cells)==18
        assert sum(c['n_observed'] for c in cells if c['source']=='fresh')==1600
        tables={'cells':cells,'contrasts':saved['contrasts'],'trajectories':saved['trajectories']}
    else:
        cells=saved['observed_cells']; assert len(cells)==2
        assert sum(c['n_complete'] for c in cells)==400
        assert [c['primary']['k'] for c in cells]==[0,0]
        assert [c['primary']['n_turns'] for c in cells]==[1400,1399]
        tables={'cells':cells,'registered_decisions':saved['registered_decisions']}
    if a.output:
        a.output.mkdir(parents=True,exist_ok=True)
        for name,rows in tables.items():
            rows=[flatten(r) for r in rows]; fields=list(dict.fromkeys(k for r in rows for k in r))
            with (a.output/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='passed',mode='saved_table_replay',rows={k:len(v) for k,v in tables.items()},frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),external_checked=bool(a.data_dir),resampling=False,model_calls=0,judge_calls=0)))
if __name__=='__main__': main()
