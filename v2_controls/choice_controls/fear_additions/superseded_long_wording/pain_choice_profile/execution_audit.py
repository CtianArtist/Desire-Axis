"""Read-only execution identity and committed-state verification; stdlib only."""
import base64
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

from .design import CONDITIONS, RAND_SEEDS, TOOL_LABELS, build_grid, digest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_application_keys():
    return {(stage, name, 'rand'+str(seed) if c['direction'] == 'random' else c['dir_kind'])
            for stage in ('prefill', 'decode') for name, c in CONDITIONS.items()
            for seed in (RAND_SEEDS if c['direction'] == 'random' else [None])}


def verify_execution(production, experiment, output):
    production, experiment, output = map(Path, (production, experiment, output))
    output.mkdir(parents=True, exist_ok=True)
    config = load(production/'configuration.json')
    complete = load(production/'completion.json')
    model = load(production/'model_identity.json')
    adapter = load(production/'adapter_identity.json')
    launch = load(production/'launch_configuration.json')
    expected = {'model': 'Qwen_2.5_32B_instruct', 'revision': '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
        'adapter_revision': 'b64bd64b4bc7ca6e0733a489b8372a099d55ef05',
        'adapter_sha256': 'cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f',
        'layer': 38, 'steer_layer': 38, 'monitor_layer': 61, 'world_size': 1,
        'conditions': CONDITIONS, 'labels': TOOL_LABELS, 'temperature': .7, 'top_p': .95,
        'choice_max_tokens': 8, 'post_press_turns': 2, 'swap_turn': 2,
        'nominal_batch_rows': 384, 'batch_rows': 384, 'precision': 'bfloat16', 'attention': 'sdpa',
        'press_effect': 'descriptive_only'}
    for key, value in expected.items():
        require(config.get(key) == value, f'Execution config mismatch: {key}')
    for name in ('torch', 'transformers', 'peft'):
        require(config['versions'][name].split('+')[0] == {'torch':'2.11.0','transformers':'5.12.1','peft':'0.20.0'}[name], name)
    scenarios = load(experiment/'inputs/frozen_scenarios.json')
    pins = load(experiment/'inputs/vector_pins.json')
    require(config['scenario_digest'] == digest(scenarios), 'Scenario digest changed')
    require(config['vector_pins'] == pins, 'Vector pins changed')
    require(config['vector_fp32_sha256'] == {k: pins['fp32_tensor_sha256'][v] for k,v in
        [('pain','s2_pain_vector'),('sadness','sadness_vector_matched_fp32'),('fear','fear_vector_matched_fp32')]}, 'Raw vector hashes changed')
    for name, field in [('frozen_scenarios.json','scenarios_file_sha256'),('stimulus_manifest_v1.json','stimulus_manifest_file_sha256'),('vector_pins.json','vector_pins_file_sha256')]:
        require(config[field] == file_hash(experiment/'inputs'/name), f'Input file changed: {name}')
    root = experiment
    source_paths = {'entrypoint':experiment/'src/run_profile.py','builder':experiment/'src/build_protocol.py',
        'reference_protocol':experiment/'inputs/reference/b2_protocol.py','profile_runtime':root/'pain_choice_profile/runtime.py',
        'design':root/'pain_choice_profile/design.py','source_runtime':root/'pain_axis_b/runtime.py',
        'source_diagnostics':root/'pain_axis_b/diagnostics.py','source_vector_verifier':root/'pain_axis_b/verify_vectors.py'}
    require(config['source_sha256'] == {k:file_hash(v) for k,v in source_paths.items()}, 'Generation code differs from executed snapshot')
    require(complete['status'] == 'completed' and complete['requested_trials'] == complete['runnable_trials'] == complete['completed_trials'] == 410, 'Incomplete production')
    require(complete['config'] == config, 'Completion/config inconsistency')
    require(model['revision'] == config['revision'] and model['world_size'] == 1 and model['base_dtype'] == 'torch.bfloat16', 'Model identity changed')
    require(model['attention'] == 'sdpa' and model['adapter_merged'] is False and model['adapter_dtypes'] == ['torch.float32'] and model['lora_layers'] == 448, 'Adapter execution changed')
    require(adapter['revision'] == config['adapter_revision'] and adapter['archive_sha256'] == config['adapter_sha256'], 'Adapter archive mismatch')
    for key in ('source_sha256','scenario_digest','vector_pins','batch_rows','conditions','adapter_sha256'):
        require(launch[key] == config[key], f'Launch/config mismatch: {key}')
    early = load(production/'early_validation_all_conditions.json')
    require(early['status'] == 'passed' and set(early['conditions']) == set(CONDITIONS), 'Missing early condition snapshots')
    for name, receipt in early['conditions'].items():
        require(receipt['condition_id'] == name and receipt['dose'] == CONDITIONS[name]['dose'] and receipt['literal_turn_zero_count'] > 0, 'Bad early capture identity')
        require(receipt['finite_prefill'] and receipt['applied_dose_checked'], 'Early capture failed')
    seen = set()
    application_rows, progress_rows = [], []
    events = {}
    with (production/'events.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            kind = row['event']
            events[kind] = events.get(kind, 0)+1
            if kind == 'profile_steering_assertion':
                name = row['condition_id']
                require(row['applied_dose'] == CONDITIONS[name]['dose'] and row['mask_preserved'] is True, 'Injection/mask mismatch')
                if CONDITIONS[name]['dose']:
                    require(row['actual_bf16_delta_norm'] > 0 and row['intended_delta_norm'] > 0, 'Steering did not reach model state')
                else:
                    require(row['actual_bf16_delta_norm'] == row['intended_delta_norm'] == 0, 'None condition steered')
                seen.add((row['stage'], name, row['direction']))
                application_rows.append(row)
            if kind == 'oom_split':
                require(row['rng_restored'] is True, 'OOM split lost RNG state')
            if kind == 'progress':
                progress_rows.append(row)
    require(seen == expected_application_keys(), f'Missing/extra condition-direction-stage checks: {expected_application_keys()-seen}')
    require(progress_rows and progress_rows[-1]['completed_choices'] == complete['completed_choices'], 'Final progress count mismatch')
    # Exact raw export must match every committed completed record, preserving order.
    db = sqlite3.connect(f'file:{production / "state.sqlite"}?mode=ro', uri=True)
    require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'SQLite integrity failed')
    require(db.execute('SELECT COUNT(*),SUM(done) FROM trials').fetchone() == (410, 410), 'Committed state incomplete')
    grid_hash = json.loads(db.execute('SELECT value FROM meta WHERE key="grid_hash"').fetchone()[0])
    expected_hash = hashlib.sha256(json.dumps({'grid':build_grid(scenarios),'config':config}, sort_keys=True).encode()).hexdigest()
    require(grid_hash == expected_hash, 'Committed grid/config identity mismatch')
    sampled_rng, n_choices = 0, 0
    with (production/'trials.jsonl').open() as f:
        for ordinal, (row, line) in enumerate(itertools.zip_longest(db.execute('SELECT id,state,record FROM trials ORDER BY id'), f)):
            require(row is not None and line is not None, 'Raw/SQLite row count mismatch')
            tid, state_text, record_text = row
            require(tid == ordinal and record_text == line.rstrip('\n'), 'Raw export not byte-identical to committed row')
            state, record = json.loads(state_text), json.loads(record_text)
            require(state['record'] == record and state['done'] is True and bool(state['messages']), 'Incomplete preserved history')
            if record['sampled']:
                require(len(base64.b64decode(state['generator'], validate=True)) > 0, 'Missing sampled RNG state')
                sampled_rng += 1
            else:
                require(state['generator'] is None, 'Unexpected greedy RNG state')
            n_choices += len(record['choices'])
    db.close()
    require(sampled_rng == 404 and n_choices == complete['completed_choices'], 'Final RNG/choice coverage mismatch')
    receipt = {'status':'passed','model':model,'grid_rows':410,'sampled_rng_states':sampled_rng,
        'choices':n_choices,'raw_equals_committed_sqlite':True,'early_conditions':len(early['conditions']),
        'application_checks':len(seen),'events':events,'first_progress':progress_rows[0],'last_progress':progress_rows[-1],
        'production_source_sha256':config['source_sha256'],'payload_accounting':load(production/'payload_accounting.json'),
        'limitation':'Payload clock excludes provider startup. Source batch/RNG behavior is retained, not batch-invariant.'}
    (output/'execution_verification.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    (output/'application_checks.json').write_text(json.dumps(application_rows,indent=2,allow_nan=False)+'\n')
    return receipt
