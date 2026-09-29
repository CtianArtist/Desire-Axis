"""Process-local startup bindings. No generation or recovery implementation here."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'32': 'Qwen_2.5_32B_instruct', '72': 'Qwen_2.5_72B_instruct'}


def verify_source_identity():
    pins = json.loads((ROOT / 'inputs/source_identity.json').read_text())
    for relative, wanted in pins.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError(f'Immutable source changed: {relative}')
    import importlib.util
    spec = importlib.util.spec_from_file_location('b2_builder', ROOT / 'src/build_protocol.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.OUTPUT.read_text() != module.generate():
        raise ValueError('Generated runner changed; regenerate from pinned anchors')


def add_inputs(parser):
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--fear-vector', type=Path, required=True)
    parser.add_argument('--fear-pin', type=Path, required=True)
    parser.add_argument('--pain-vector', type=Path, required=True)
    parser.add_argument('--pain-manifest', type=Path, required=True)
    parser.add_argument('--pain-metadata', type=Path)
    parser.add_argument('--scenarios', type=Path, default=ROOT / 'inputs/scenarios.json')
    parser.add_argument('--tp-guard-inputs', type=Path)
    parser.add_argument('--batch-rows', type=int)


def nominal_batch(model, value):
    if model == '72' and value is None:
        raise ValueError('--batch-rows is mandatory for72; nominal192/384 decision is not assumed')
    value = 384 if value is None else value
    if type(value) is not int or value <= 0:
        raise ValueError('--batch-rows must be a positive integer')
    if model == '32' and value > 384:
        raise ValueError('32B batch cannot exceed original384')
    return value


def configure_guard(model, directory):
    if model == '72':
        if directory is None:
            raise ValueError('72B requires --tp-guard-inputs')
        for name in ('source_manifest_v2.json', 'pinned_72B_config.json'):
            if not (directory / name).is_file():
                raise ValueError(f'Missing native TP guard input: {name}')
        os.environ['PAIN_TP_METADATA_REPAIR_V1'] = '1'
        os.environ['PAIN_TP_GUARD_INPUTS'] = str(directory.resolve())
    else:
        os.environ.pop('PAIN_TP_METADATA_REPAIR_V1', None)
        os.environ.pop('PAIN_TP_GUARD_INPUTS', None)


def runner_main(namespace):
    p = argparse.ArgumentParser(description='Internal process entry; launch through src/launch.py')
    add_inputs(p)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--nominal-batch-rows', type=int, required=True)
    args = p.parse_args()
    verify_source_identity()
    assert os.environ.get('B2_CONTROLLED_LAUNCH') == '1', 'Use the fresh-output launcher'
    root = Path(os.environ['B2_RUN_ROOT']).resolve()
    assert args.output.resolve().is_relative_to(root)
    if args.resume:
        assert args.resume.resolve().is_relative_to(root), 'No historical/external SQLite input'
    assert not (args.output / 'state.sqlite').exists(), 'Refusing to attach an existing attempt state'
    batch = nominal_batch(args.model, args.batch_rows)
    nominal = nominal_batch(args.model, args.nominal_batch_rows)
    assert batch <= nominal
    if args.model == '72':
        assert int(os.environ.get('WORLD_SIZE', '0')) == 2
        assert int(os.environ.get('LOCAL_WORLD_SIZE', '0')) == 2
        assert os.environ.get('PAIN_RECOVERY_ATTEMPT') is not None
    else:
        assert int(os.environ.get('WORLD_SIZE', '1')) == 1
        assert 'PAIN_RECOVERY_ATTEMPT' not in os.environ
    configure_guard(args.model, args.tp_guard_inputs)
    from pain_axis_b.grid import read_scenarios
    from pain_axis_b.fear_inputs import load_inputs
    from pain_axis_b import runtime as rt
    scenarios = read_scenarios(args.scenarios)
    verified = load_inputs(MODELS[args.model], args.pain_vector, args.pain_manifest,
                           args.pain_metadata, args.fear_vector, args.fear_pin)
    # Optional UI shim changes telemetry only, never any numerical method.
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from progress_adapter import install
    install(rt)
    rt.init_runtime(2 if args.model == '72' else 1, args.output)
    adapter = rt.download_adapter(MODELS[args.model], args.output)
    spec = next(s for s in namespace['MODELS'] if s[1] == MODELS[args.model])
    spec = (*spec[:-1], batch)
    namespace['run_model'](*spec, list(namespace['TOOL_LABELS']), adapter, scenarios, 10**9,
                           args.pain_vector, args.fear_vector, args.output, args.resume,
                           verified_inputs=verified, nominal_batch_rows=nominal)
