"""Portable future fresh launch, with a model-free configuration/asset dry run.

This is new launch glue, NOT a reproduction of the historical zero-choice-state
launcher. The customer sampler and bounded recovery controller are retained.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = {'32': 'Qwen_2.5_32B_instruct', '72': 'Qwen_2.5_72B_instruct'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=['b1_b3','b6'], required=True)
    p.add_argument('--model', choices=['32','72'], default='32')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.study == 'b6' and a.model != '32': p.error('B6 only ran the 32B model')
    root = ROOT/a.study
    data = a.data_dir.resolve()
    model = MODELS[a.model]
    protocol = root/'scripts/4.3_selfmed/04_selfmed_two_buttons.py'
    # Parse literal recipe settings without importing torch, transformers or the protocol.
    tree = ast.parse(protocol.read_text())
    names = {'MODELS','RAND_SEEDS','SEED_BASES','TOOL_LABELS','SYSTEM_TEMPLATE','CHOICE_QUESTION',
             'CHOICE_QUESTION_LABEL_FREE','TEMPERATURE','TOP_P','CHOICE_MAX_TOKENS','POST_PRESS_TURNS','SWAP_TURN'}
    config = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign): continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try: config[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
            elif isinstance(target, ast.Tuple):
                try:
                    values = ast.literal_eval(node.value)
                    for key,value in zip(target.elts,values):
                        if isinstance(key,ast.Name) and key.id in names: config[key.id] = value
                except (ValueError, TypeError): pass
    config['selected_model'] = next(row for row in config['MODELS'] if row[1]==model)
    scenarios = json.loads((root/'inputs/scenarios.json').read_text())
    grid = json.loads((root/'inputs/requested_grid.json').read_text())
    assert isinstance(scenarios,dict) and grid['labels']
    vector_names = ['verification.json','original_tensor_hashes_v2.json',f'{model}_pain_vectors.safetensors',
                    f'{model}_pain_vectors.json',f'vectors_{model}.pt']
    required = [data/'vectors'/n for n in vector_names]
    if a.study == 'b6': required += [data/'fear'/f'fear_{model}.pt',data/'fear/verification_fear.json']
    if a.model == '72': required += [data/'tp_guard/source_manifest_v2.json',data/'tp_guard/pinned_72B_config.json']
    missing = [str(x.relative_to(data)) for x in required if not x.is_file()]
    manifest = {'study':a.study,'model':model,'config':config,'frozen_labels':grid['labels'],
                'missing_local_assets':missing,'public_asset_urls':None,
                'fresh_launch':True,'historical_resume_state_required':False,
                'weights_loading':'not attempted','model_cache':'Original pinned Hugging Face model and adapter downloads; network/cache needed only on actual execution',
                'historical_equivalence':'No claim of identical sampled trajectories; historical 72B started from a pinned zero-choice state and regrouped 192→96→48→24→12.'}
    if a.dry_run:
        assert 'torch' not in sys.modules and 'transformers' not in sys.modules
        print(json.dumps(manifest,indent=2))
        return 0
    if missing: p.error('Required local assets missing (public release pending): '+', '.join(missing))
    if not a.output: p.error('--output is required for inference')
    output = a.output.resolve()
    if output.exists(): p.error('Fresh launch requires an output path that does not exist')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    args = ['--model',model,'--inputs',str(data/'vectors'),'--scenarios',str(root/'inputs/scenarios.json')]
    if a.study == 'b6': args += ['--fear-inputs',str(data/'fear')]
    if a.model == '72':
        env.update(PAIN_TP_METADATA_REPAIR_V1='1',PAIN_TP_GUARD_INPUTS=str(data/'tp_guard'),PAIN_DURABLE_DIAGNOSTICS='1')
        sys.path.insert(0,str(root))
        from pain_axis_b.recovery import recover
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
                   '--max-restarts=0','--log-dir','{attempt_output}/diagnostics/torchrun','--redirects','3','--tee','3',
                   str(root/'rank_entrypoint.py'),str(protocol),*args]
        # Fresh-process controller snapshots initialized RNG/state itself. No historical state injected.
        result = recover(command,output,env,initial_batch=192,floor=1,max_restarts=7,
                         total_timeout=28000,attempt_timeout=28000,stall_timeout=1800,initial_state=None)
        return 0 if result['status']=='completed' else 1
    return subprocess.call([sys.executable,str(protocol),*args,'--output',str(output)],cwd=root,env=env)

if __name__ == '__main__':
    raise SystemExit(main())
