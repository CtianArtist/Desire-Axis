"""Plan or launch the isolated original runner. Dry-run never imports torch.

This packaging task did not execute inference. Real launch needs pinned CUDA
runtime and separately supplied vector/model/cache assets; see README.md.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def plan(a):
    if a.group in ('superseded_long_wording', 'matched_wording'):
        base = ROOT / a.group
        model = 'Qwen_2.5_32B_instruct'
        assert a.model == '32', 'Spam additions exist only for 32B'
    elif a.group == 'lamp32':
        base = ROOT / 'four_cells/lamp32'
        model = 'Qwen_2.5_32B_instruct'
        assert a.model == '32', 'Lamp additions exist only for 32B'
    else:
        base = ROOT / 'four_cells/b2'
        model = 'Qwen_2.5_' + a.model + 'B_instruct'
    spec = importlib.util.spec_from_file_location('protocol_builder', base / 'src/build_protocol.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.OUTPUT.read_text() == builder.generate(), 'Generated runner differs from frozen customer source'
    data = a.data_dir.resolve()
    if a.group == 'b2':
        cmd = [sys.executable, str(base / 'src/launch.py'), '--model', a.model,
               '--fear-vector', str(data / 'vectors' / ('fear' + a.model + '.safetensors' if a.model == '72' else 'fear_Qwen_2.5_32B_instruct.pt')),
               '--fear-pin', str(data / 'vectors' / ('fear' + a.model + '_pin.json')),
               '--pain-vector', str(data / 'vectors' / (model + '_pain_vectors.safetensors')),
               '--pain-manifest', str(data / 'vectors/original_tensor_hashes_v2.json'),
               '--pain-metadata', str(data / 'vectors' / (model + '_pain_vectors.json')),
               '--scenarios', str(base / 'inputs/scenarios.json'),
               '--output-base', str(a.output.resolve()), '--batch-rows', '192' if a.model == '72' else '384']
        if a.model == '72':
            cmd += ['--tp-guard-inputs', str(data / 'tp_guard')]
        trials = 410
    else:
        cmd = [sys.executable, str(base / 'src/run_profile.py'), '--model', model,
               '--inputs', str(data / 'vectors'), '--fear-inputs', str(data / 'vectors'),
               '--scenarios', str(base / 'inputs/frozen_scenarios.json'),
               '--stimulus-manifest', str(base / 'inputs/stimulus_manifest_v1.json'),
               '--vector-pins', str(base / 'inputs/portable_vector_pins.json'), '--output', str(a.output.resolve())]
        trials = 820 if a.group == 'lamp32' else 410
    return base, {'group': a.group, 'model': model, 'trials': trials, 'sampled': trials // 410 * 404,
                  'greedy': trials // 410 * 6, 'model_loaded': False, 'command': cmd,
                  'data_assets_checked': False, 'status': 'source_and_command_verified',
                  'note': 'Dry-run validates frozen generated code, configuration and command only; not asset availability or GPU compatibility.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['superseded_long_wording', 'matched_wording', 'lamp32', 'b2'], default='matched_wording')
    p.add_argument('--model', choices=['32', '72'], default='32')
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('new-inference-output'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-superseded', action='store_true', help='Explicitly permit a historical-only long-wording launch')
    a = p.parse_args()
    base, receipt = plan(a)
    if a.dry_run:
        assert not any(x in sys.modules for x in ('torch', 'transformers', 'peft'))
        print(json.dumps(receipt, indent=2))
        return
    if a.group == 'superseded_long_wording' and not a.allow_superseded:
        p.error('Historical-only condition: use --allow-superseded; never pool with current comparisons')
    env = dict(os.environ, PYTHONPATH=str(base), CHOICE_SOURCE_ROOT=str(base))
    subprocess.run(receipt['command'], cwd=base, env=env, check=True)


if __name__ == '__main__':
    main()
