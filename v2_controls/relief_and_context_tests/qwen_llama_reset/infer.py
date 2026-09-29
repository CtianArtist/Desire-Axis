"""Portable future inference entrypoint; --dry-run never imports an ML runtime."""
import argparse, json, os, runpy, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--member', choices=('qwen25_32b','llama31_8b','olmo_32b'))
    p.add_argument('--style', choices=('selfreport','zone'), default='selfreport')
    a=p.parse_args(); c=json.loads((ROOT/'config.json').read_text())
    olmo=c['group']=='olmo_reset'
    member=a.member or ('olmo_32b' if olmo else 'qwen25_32b')
    if member not in c['models']: p.error('member not in this frozen study')
    if not olmo and a.style!='selfreport': p.error('only the completed pain-only selfreport tranche is in scope')
    print(json.dumps(dict(status='dry_run' if a.dry_run else 'future_generation', member=member, config=c, data_dir=str(a.data_dir), output=str(a.output), loads_model=not a.dry_run),indent=2))
    if a.dry_run: return
    sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'author/act-on-valence/src')]
    os.environ.setdefault('ALLOCATION_PAYLOAD_START',str(time.time()))
    if olmo:
        sys.argv=['olmo_run.py','--style',a.style,'--vectors-root',str(a.data_dir/'selection_v1'),'--output',str(a.output)]
        runpy.run_path(str(ROOT/'src/olmo_run.py'),run_name='__main__')
    else:
        # Exact original 200-ID loop, with only the completed pain-only cell selected.
        import removal
        layer=c['models'][member]['main_layer']
        raise SystemExit(removal.main(['--member',member,'--layer',str(layer),'--vectors-root',str(a.data_dir/'prepared_v1'),'--output',str(a.output),'--shard','0','--n-shards','1','--only-cell','gated_selfreport_pain_d1']))
if __name__=='__main__': main()
