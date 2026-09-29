"""CPU-only OLMo analysis. Run substantive resampling on parent-launched compute.

Raw fresh records are streamed and discarded after parser/state validation and
independent recount. Archived actions are NEVER reparsed from stripped content.
The source independent cluster bootstrap is loaded unchanged through an AST
allowlist. No model, generation, judge API, or paired bootstrap is imported.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import archive_audit as audit

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'author/act-on-valence'
LEVER = AUTHOR / 'experiments/lever'
REVISION = 'b96024342a77a69aa0dda815c3454a671f477463'
ARCHIVE_SHA = '59d738ef8cb2e7fe9ba9b2c458a005449e44eac3349110d4d177895a0f28c73e'
PROMPTS = ('selfreport', 'zone')
NEW_CONDITIONS = ('pain_d05', 'rand_d05', 'pain_d1', 'rand_d1')
ARCHIVE_CONDITIONS = ('neg_d05', 'neg_d1', 'null', 'rand_d05', 'rand_d1')
EXPECTED_IDS = set(range(200))
PRODUCTION_IDENTITY = Path(__file__).resolve().parents[1] / 'inputs/production_identity_v1.json'
JUDGE_MODEL = 'anthropic/claude-sonnet-4.6'
ADDITION_ARITHMETIC = 'sum_in_order_fp32(vector*strength); cast_sum_to_parameter_dtype; pre_bf16+add_bf16'


def common():
    # Import direction arithmetic and saved inputs only, never load_bundle/model.
    for path in (ROOT, AUTHOR / 'src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from pain_port import common as module
    return module


def source_ref(path, **fields):
    """Keep actual paths and add durable references only for mounted artifacts."""
    actual, sep, member = str(path).partition('!')
    actual = Path(actual).resolve()
    result = dict(path=str(actual) + (sep + member if sep else ''), **fields)

    return result


def production_binding(vectors_root):
    require(vectors_root is not None, 'completed selection_v1 --vectors-root is required')
    approved = json.loads(PRODUCTION_IDENTITY.read_text())
    require((approved['model'], approved['revision'], approved['rho'], approved['injection_layer'],
             approved['selected_layer']) == ('allenai/OLMo-2-0325-32B-Instruct', REVISION, .3, 32, 57),
            'invalid approved generation identity')
    binding = common().load_olmo_inputs(vectors_root, layer=32)
    verify_binding(binding, approved)
    # The analysis.json hash is frozen before generation. Its persisted norm
    # reproduces every original pain strength exactly; recomputing a FP32 norm
    # on another CPU can differ by one ULP. Do not alter the generation recipe,
    # use a row's self-reported norm, or replace exact strength equality by isclose.
    selection = json.loads((Path(vectors_root) / 'analysis.json').read_text())
    require(audit.sha((Path(vectors_root) / 'analysis.json').read_bytes()) == approved['selection_sha256'],
            'selection bytes changed while binding saved norm')
    raw_norm = selection['refits']['final_token']['raw_norm']
    require(finite(raw_norm) and raw_norm > 0
            and np.isclose(raw_norm, np.linalg.norm(binding['directions']['pain'].astype(np.float64)), rtol=2e-6, atol=0.),
            'saved pain norm disagrees with bound vector')
    binding['verification_pain_raw_norm'] = raw_norm
    return binding, approved


def verify_binding(binding, approved):
    """Production-only shape and frozen selection checks; toy vectors cannot pass."""
    for key, expected in (('member', 'olmo_32b'), ('hf_id', approved['model']),
                          ('revision', approved['revision']), ('rho', .3), ('layer', 32), ('monitor_layer', 57)):
        require(binding.get(key) == expected, f'binding identity mismatch: {key}')
    provenance = binding['provenance']
    require(provenance.get('selection_sha256') == approved['selection_sha256'], 'selection identity mismatch')
    require(provenance['vectors'].get('raw_tensor_sha256') == approved['raw_tensor_sha256'],
            'raw tensor identity mismatch')
    require(binding['directions']['pain'].shape == (5120,) and binding['rpool'].shape == (16, 5120),
            'production binding shape mismatch')
    require(hashlib.sha256(binding['directions']['pain'].tobytes()).hexdigest() == approved['raw_tensor_sha256'],
            'binding raw tensor bytes mismatch')
    require(bool(approved.get('source_sha256')) and bool(approved.get('template_date')),
            'approved execution identity missing')


def verify_identity(row, binding, approved):
    require(row.get('execution_identity') == {k: approved[k] for k in ('source_sha256', 'template_date')},
            'fresh frozen source/template identity mismatch')
    provenance = row.get('provenance', {})
    require({k: v for k, v in provenance.items() if k != 'runtime'} == binding['provenance'],
            'fresh bound vector/bank/pool/selection provenance mismatch')
    require(row.get('revision') == approved['revision'] and row.get('rho') == approved['rho'],
            'fresh frozen model/rho mismatch')
    # The source harness does not write hf_id at row level; the bound vector metadata does.
    if 'hf_id' in row:
        require(row['hf_id'] == approved['model'], 'fresh model identity mismatch')
    runtime = provenance.get('runtime', {})
    require(isinstance(runtime, dict), 'missing runtime identity')
    require(isinstance(runtime.get('torch'), str) and runtime['torch'].split('+')[0] == '2.14.0',
            'runtime torch identity mismatch')
    for key, expected in (('transformers', '5.17.0'), ('numpy', '2.3.4'), ('dtype', 'torch.bfloat16'),
                          ('attention_implementation', 'sdpa'), ('snapshot_revision', REVISION)):
        require(runtime.get(key) == expected, f'runtime identity mismatch: {key}')
    require(runtime.get('adapters') is False, 'runtime adapters identity mismatch')
    require(isinstance(runtime.get('gpu'), str) and re.fullmatch(r'(?:NVIDIA )?H100[^,\n]*', runtime['gpu']),
            'runtime must identify one H100-compatible device')
    for key in ('driver', 'cuda_build', 'tokenizers'):
        require(isinstance(runtime.get(key), str) and bool(runtime[key]), f'missing runtime identity: {key}')
    # gpu_count was not persisted. Exact approved source identity above binds the
    # load_bundle device_count()==1 assertion; do not manufacture a recorded count.
    if 'gpu_count' in runtime:
        require(type(runtime['gpu_count']) is int and runtime['gpu_count'] == 1, 'runtime GPU count mismatch')


def definitions(path, names, ns):
    """Execute exact source definitions only; never execute source imports."""
    nodes, found = [], set()
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        ids = ({node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
               {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set())
        if ids & names:
            if ids - names:
                raise RuntimeError(f'non-allowlisted assignment: {ids}')
            nodes.append(node)
            found |= ids
    if found != names:
        raise RuntimeError(f'missing source definitions in {path}: {names - found}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)


def author_functions():
    cfg = {}
    definitions(LEVER / 'config.py', {'BOOT_SEED', 'BOOT_N', 'EQUIV_MARGIN'}, cfg)
    if (cfg['BOOT_SEED'], cfg['BOOT_N'], cfg['EQUIV_MARGIN']) != (285, 10000, .05):
        raise ValueError('source bootstrap contract changed')
    ns = dict(np=np, math=math, hashlib=hashlib, C=SimpleNamespace(**cfg))
    definitions(LEVER / 'analyze.py', {'Z', 'wilson', '_boot_draws', 'ratio_boot', 'ratio_diff_boot',
                'primary_arrays', 'EMPTY_SCORE', 'item_id', 'judged', 'conv_means', 'mean_ci', 'judge_coverage'}, ns)
    definitions(LEVER / 'harness.py', {'_PHASE_ORDER', '_has', '_adjusted', 'summarize'}, ns)
    return SimpleNamespace(**ns)


def source_parser():
    ns = dict(__name__=__name__, re=re, json=json, dataclass=dataclass)
    definitions(AUTHOR / 'src/act_on_valence/self_injection.py',
                {'Action', '_THINK_RE', '_OPEN_THINK_RE', '_TOOLCALL_RE', '_TAG_RE', '_FN_RE', '_NUM_RE',
                 'strip_think', 'strip_tool_calls', '_intensity_from', '_norm', 'parse_actions', 'dose_for'}, ns)
    definitions(Path(__file__).with_name('removal.py'), {'negative_requests'}, ns)
    return SimpleNamespace(**ns)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fingerprint(row):
    return audit.sha(json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':')).encode())


def verify_new(row, parser, binding=None, approved=None):
    """Validate schedule/state/evidence; production also requires binding+approved.

    Omitting approved is for isolated fixtures only, never production certification.
    load_new has no such bypass and verifies the saved 5120-wide selection first.
    """
    require(row['member'] == 'olmo_32b' and type(row['layer']) is int and row['layer'] == 32,
            'fresh member/layer mismatch')
    require(row.get('revision') == REVISION, 'fresh revision mismatch')
    require(row['prompt'] in PROMPTS and row['cond'] in NEW_CONDITIONS, 'fresh cell outside scope')
    require(type(row['conv']) is int and row['conv'] in EXPECTED_IDS, 'fresh invalid ID')
    require(row['cell'] == f"gated_{row['prompt']}_{row['cond']}", 'fresh cell label mismatch')
    require(row['kind'] == 'removal', 'fresh kind mismatch')
    role, suffix = row['cond'].split('_')
    require(row.get('seed') == 100 * row['conv'] and type(row.get('seed')) is int, 'fresh row seed mismatch')
    require('random_dir' in row and row['random_dir'] == (row['conv'] % 16 if role == 'rand' else None)
            and (role != 'rand' or type(row['random_dir']) is int),
            'fresh random direction mismatch')
    require(finite(row['rho']) and row['rho'] == .3, 'fresh rho must be .3')
    if approved is not None:
        require(binding is not None, 'approved validation requires a binding')
        verify_identity(row, binding, approved)
    dose = round(row['rho'] * (.5 if suffix == 'd05' else 1), 6)
    require(row['op_role'] == role and row['op_dose'] == dose and row['tool_role'] == role
            and row['tool_dose'] == dose and row['d'] == (.5 if suffix == 'd05' else 1.),
            'fresh condition dose metadata mismatch')
    require(isinstance(row.get('provenance'), dict) and bool(row['provenance']), 'missing vector provenance')
    ident = row.get('execution_identity', {})
    require(bool(ident.get('source_sha256')) and bool(ident.get('template_date')), 'missing execution provenance')
    require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                for v in ident['source_sha256'].values()), 'invalid execution source hashes')
    require([(t['phase'], t['round']) for t in row['turns']] == audit.STRUCTURE, 'incomplete fresh schedule')
    pending, monitor_signature = None, None
    for i, t in enumerate(row['turns']):
        tools = i >= 4
        default = (role, dose, 'operator') if t['phase'] == 'exposure' or (tools and t['round'] >= 1) else ('zero', 0., 'operator')
        active_role, active_dose, state = pending or default
        if active_dose <= 0:
            active_role = 'zero'
        require((t['imposed_role'], t['imposed_dose'], t['source']) == (active_role, active_dose, state),
                f'fresh state transition mismatch at turn {i}')
        require(type(t['tools_available']) is bool and t['tools_available'] == tools, 'fresh tool schedule mismatch')
        require(type(t['coherent']) is bool, 'invalid coherence flag')
        ev = t['evidence']
        raw = ev['raw_output']
        require(isinstance(raw, str), 'missing raw output')
        require(t['actions'] == [asdict(a) for a in parser.parse_actions(raw)], 'fresh raw parser mismatch')
        require(t['content'] == parser.strip_tool_calls(parser.strip_think(raw))[:1200], 'fresh stripped content mismatch')
        require(ev['raw_negative_requests'] == parser.negative_requests(raw, parser), 'raw negative request mismatch')
        require(ev['seed'] == 100 * row['conv'] + 7 * i + 1, 'fresh seed mismatch')
        require(bool(ev['input_token_ids']) and len(ev['generated_token_ids']) <= 200, 'fresh token evidence mismatch')
        require(ev['sampler'] == dict(do_sample=True, temperature=.7, top_p=.95, max_new_tokens=200),
                'fresh sampler mismatch')
        for steer in ev['applied_steer']:
            require(steer['layers'] == [32] and finite(steer['strength'])
                    and finite(steer['intended_injected_l2'])
                    and bool(re.fullmatch('[0-9a-f]{64}', steer['vector_sha256'])), 'invalid applied steer provenance')
        require(bool(ev['applied_steer']) == (active_dose > 0), 'applied steer disagrees with actual state')
        intended_norm = None
        if binding is not None:
            name = f"random{row['conv'] % 16}" if active_role == 'rand' else active_role
            vec, strength, info = common().resolve_direction(binding, name, active_dose / binding['rho'])
            expected_count = int(vec is not None)
            require(len(ev['applied_steer']) == expected_count, 'applied steer count mismatch')
            intended_norm = info['injected_l2']
            if vec is not None:
                steer = ev['applied_steer'][0]
                if active_role == 'pain' and 'verification_pain_raw_norm' in binding:
                    strength = info['inj'] * binding['resid_rms'] / binding['verification_pain_raw_norm']
                require(steer['vector_sha256'] == hashlib.sha256(np.asarray(vec, dtype='<f4').tobytes()).hexdigest(),
                        'applied vector hash mismatch')
                require(steer['strength'] == strength, 'applied strength mismatch')
                require(steer.get('vector_encoding') == 'little_endian_float32'
                        and steer.get('vector_numel') == vec.size, 'applied vector encoding/shape mismatch')
                require(finite(steer.get('vector_l2')) and np.isclose(steer['vector_l2'], np.linalg.norm(vec), rtol=2e-6),
                        'applied vector norm mismatch')
                require(np.isclose(steer['intended_injected_l2'], intended_norm, rtol=2e-6, atol=1e-7),
                        'applied intended norm mismatch')
        if tools:
            obs = ev['observations']
            require(set(obs) == {'selected_s2', 'layer32_postadd', 'layer32_preadd'}, 'missing/unknown monitor sites')
            require(obs['selected_s2']['layer'] == (binding['monitor_layer'] if binding is not None else 57),
                    'selected monitor layer mismatch')
            for name, ob in obs.items():
                require(finite(ob['projection']), 'nonfinite monitor projection')
                require(ob.get('dtype') == 'torch.bfloat16', 'monitor dtype must be BF16')
                require(type(ob['layer']) is int and 0 <= ob['layer'] < 64, 'invalid monitor layer')
                if name.startswith('layer32'):
                    require(ob['layer'] == 32, 'layer32 observation mismatch')
                semantics = 'pre_add' if name == 'layer32_preadd' else 'post_add' if ob['layer'] == 32 else 'raw_post_block'
                relation = 'upstream' if ob['layer'] < 32 else 'same_layer' if ob['layer'] == 32 else 'downstream'
                require(ob['site'] == 'decoder_layer_output' and ob['position'] == 'prompt_final'
                        and ob['pass_kind'] == 'first_prefill_only' and ob['axis'] == 'saved_S2_pain'
                        and ob['semantics'] == semantics and ob['relation_to_injection'] == relation
                        and ob['prefill_tokens'] == len(ev['input_token_ids']), 'monitor site provenance mismatch')
            signature = monitor_identity(t)
            if monitor_signature is None:
                monitor_signature = signature
            require(signature == monitor_signature, 'monitor provenance changed between turns')
            require(ev['s2_projection'] == obs['selected_s2']['projection'], 'monitor alias mismatch')
            check = ev.get('layer32_addition_check')
            require(isinstance(check, dict), 'missing layer32 addition verification')
            require(check.get('exact_bf16_match') is True and finite(check.get('actual_delta_l2'))
                    and check['actual_delta_l2'] == check.get('expected_bf16_delta_l2')
                    and finite(check.get('intended_fp32_l2'))
                    and check['actual_delta_l2'] >= 0
                    and (check['actual_delta_l2'] > 0) == (active_dose > 0)
                    and check.get('has_layer32_add') is (active_dose > 0)
                    and check.get('arithmetic') == ADDITION_ARITHMETIC
                    and check.get('scope') == 'prompt_final_first_prefill'
                    and check.get('max_abs_error') == 0.
                    and type(check.get('changed_elements')) is int
                    and 0 <= check['changed_elements'] <= (binding['directions']['pain'].size if binding is not None else 5120)
                    and (check['changed_elements'] > 0) == (active_dose > 0)
                    and finite(check.get('cast_add_l2')) and check['cast_add_l2'] >= 0
                    and (check['cast_add_l2'] > 0) == (active_dose > 0), 'invalid addition verification')
            if intended_norm is not None:
                require(np.isclose(check['intended_fp32_l2'], intended_norm, rtol=2e-6, atol=1e-7),
                        'addition intended norm mismatch')
        adj = next((a for a in t['actions'] if a['op'] == 'adjust'), None)
        require(t['requested_intensity'] == (adj['intensity'] if adj else None),
                'requested intensity disagrees with original parsed action')
        pending = None
        if tools and adj is not None:
            pending = (role, parser.dose_for(adj['intensity'], dose), 'model_adjust')
        elif tools and any(a['op'] == 'reset' for a in t['actions']):
            pending = ('zero', 0., 'model_reset')


def monitor_identity(turn):
    """Ignore token count and observed values, never mix measurement sites."""
    return {name: {k: ob[k] for k in ('layer', 'site', 'position', 'pass_kind', 'axis', 'semantics',
                                     'relation_to_injection')}
            for name, ob in turn['evidence']['observations'].items()}


def compact(row, source, ref, author, scores):
    counts = audit.recount(row['turns'])
    original = author.summarize(row['turns'])
    require(counts == original, f'independent/source recount mismatch: {ref}')
    for key, value in counts.items():
        require(key in row and row[key] == value, f'stored summary mismatch: {key}: {ref}')
    out = {k: row[k] for k in ('prompt', 'cond', 'conv', 'kind', 'cell', 'op_role', 'op_dose')}
    out.update(counts)
    out.update(source=source, cell_id=f"{source}/{row['prompt']}/{row['cond']}",
               cohort='fresh_primary' if source == 'fresh' else audit.cohort(row['cond']),
               evidence_refs=[ref], source_model_runtime={k: row.get(k) for k in
                   ('member', 'layer', 'revision', 'hf_id', 'provenance', 'execution_identity', 'rho',
                    'tokenizer_revision', 'chat_template_sha256', 'dtype', 'torch_version',
                    'transformers_version', 'attention_implementation')})
    out['missing_model_runtime_identity_fields'] = [k for k, v in out['source_model_runtime'].items() if v is None]
    out['turns'] = []
    for i, t in enumerate(row['turns']):
        ev = t.get('evidence', {}) if source == 'fresh' else {}
        turn = {k: t[k] for k in ('phase', 'round', 'tools_available', 'source', 'imposed_role', 'imposed_dose',
                                  'actions', 'requested_intensity', 'coherent')}
        turn.update(turn_index=i, raw_negative_count=len(ev['raw_negative_requests']) if source == 'fresh' else None,
                    observations=ev.get('observations', {}))
        out['turns'].append(turn)
    out['judge'] = dict(scores_available=scores is not None)
    if scores is not None:
        out['judge']['coverage'] = author.judge_coverage([row], scores)
        for key in ('valence', 'coherence'):
            vals = author.conv_means([row], scores, phase='exposure', key=key)
            out['judge'][key] = vals[0] if vals else None
    return out


def score_file(path):
    if path is None:
        return None, {'available': False, 'reason': 'scores_not_supplied; primary records retained'}
    path = Path(path)
    payload = json.loads(path.read_text())
    require(payload.get('model') == JUDGE_MODEL, 'judge model identity missing or incorrect')
    if 'identity' in payload:
        require(isinstance(payload['identity'], dict) and payload['identity'].get('model') == JUDGE_MODEL
                and payload['identity'].get('identity_ok') is True, 'judge nested identity missing or incorrect')
    scores = payload['scores']
    require(isinstance(scores, dict), 'scores must be an object')
    return scores, source_ref(path, available=True, model=JUDGE_MODEL, sha256=audit.file_sha(path), n_scores=len(scores))


def insert(row, raw, ref, source, seen, rows, author, scores):
    key = (source, row['prompt'], row['cond'], row['conv'])
    canonical = fingerprint(row)
    ref.update(id='/'.join(map(str, key)), row_sha256=audit.sha(raw.rstrip(b'\r\n')))
    if key in seen:
        old, index = seen[key]
        require(old == canonical, f'conflicting duplicate: {key}')
        rows[index]['evidence_refs'].append(ref)
        return
    seen[key] = (canonical, len(rows))
    rows.append(compact(row, source, ref, author, scores))


def load_new(paths, author, parser, scores=None, *, vectors_root=None):
    # No fixture bypass here: all streamed production records bind saved inputs.
    binding, approved = production_binding(vectors_root)
    files, explicit = set(), set()
    for path in map(Path, paths):
        require(path.exists(), f'input missing: {path}')
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob('*.jsonl'))
        else:
            explicit.add(path.resolve())
            files.add(path.resolve())
    rows, seen, inputs, skipped = [], {}, [], []
    provenance = None
    for path in sorted(files):
        # Production outputs place a small audit sidecar beside raw cell files.
        if path.name == 'record_audit.jsonl' and path not in explicit:
            skipped.append(str(path))
            continue
        before = path.stat()
        identity = source_ref(path, sha256=audit.file_sha(path))
        inputs.append(identity)
        with path.open('rb') as fh:
            for line_no, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                verify_new(row, parser, binding, approved)
                signature = fingerprint(dict({k: row[k] for k in ('provenance', 'execution_identity', 'rho')},
                                             monitor_sites=monitor_identity(row['turns'][4])))
                if provenance is None:
                    provenance = signature
                require(signature == provenance, 'mixed fresh vector/runtime provenance')
                insert(row, raw, dict(identity, line=line_no), 'fresh', seen, rows, author, scores)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f'input changed during read: {path}')
    return rows, dict(files=inputs, skipped_audit_sidecars=skipped,
                      selection=source_ref(Path(vectors_root) / 'analysis.json', sha256=approved['selection_sha256']),
                      approved_identity=source_ref(PRODUCTION_IDENTITY, sha256=audit.file_sha(PRODUCTION_IDENTITY)),
                      gpu_count_evidence='one GPU enforced by frozen generation source; no persisted GPU-count field')


def load_archive(path, author, scores=None):
    data, reference, table, identities = audit.load_inputs(path)
    require(audit.sha(data) == ARCHIVE_SHA, 'archive JSONL SHA256 mismatch')
    rows, seen = [], {}
    identities = {key: source_ref(value['path'], **{k: v for k, v in value.items() if k != 'path'})
                  if isinstance(value, dict) and 'path' in value else value for key, value in identities.items()}
    identity = identities['conversations']
    for line_no, raw in enumerate(io.BytesIO(data), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        require(row['cell'] in audit.CELLS and row['cell'] == f"gated_{row['prompt']}_{row['cond']}",
                'archive cell outside scope')
        require(type(row['conv']) is int and row['conv'] in EXPECTED_IDS, 'archive invalid ID')
        audit.validate_structure(row)
        for key in ('prompt', 'cond', 'kind', 'd', 'op_role', 'op_dose', 'tool_role', 'tool_dose'):
            require(row[key] == reference['cells'][row['cell']][key], f'archive metadata mismatch: {key}')
        insert(row, raw, dict(identity, line=line_no),
               'archive', seen, rows, author, scores)
    for prompt in PROMPTS:
        for cond in audit.CONDITIONS:
            ids = {r['conv'] for r in rows if (r['prompt'], r['cond']) == (prompt, cond)}
            require(ids == EXPECTED_IDS, f'incomplete archive: {prompt}/{cond}')
    return rows, identities


def arrays(rows, author):
    return author.primary_arrays(rows) if rows else (np.array([], float), np.array([], float))


def ratio(num, den, author, reps):
    result = author.ratio_boot(num, den, seed=285, n_boot=reps)
    result['bootstrap_degenerate_all_zero'] = bool(sum(den) > 0 and sum(num) == 0)
    result['bootstrap_degenerate'] = bool(math.isfinite(result['lo']) and result['lo'] == result['hi'])
    return result


def cell_stats(rows, source, prompt, cond, author, reps):
    num, den = arrays(rows, author)
    cell_id = f'{source}/{prompt}/{cond}'
    result = dict(cell_id=cell_id, source=source, prompt=prompt, cond=cond,
                  cohort='fresh_primary' if source == 'fresh' else audit.cohort(cond),
                  n_expected=200, n_observed=len(rows), conversation_ids=[r['conv'] for r in rows],
                  missing_ids=sorted(EXPECTED_IDS - {r['conv'] for r in rows}),
                  coverage_complete={r['conv'] for r in rows} == EXPECTED_IDS,
                  primary_keys=['n_resets_no_state', 'n_no_state_turns'] if cond == 'null' else
                               ['n_resets_operator_active', 'n_operator_active_turns'],
                  primary=ratio(num, den, author, reps),
                  zero_eligible_conversations=sum(float(d) == 0 for d in den),
                  any_operator_active_reset_wilson=author.wilson(sum(r['removed_when_operator_imposed'] for r in rows), len(rows)),
                  operator_active_zero_eligible_conversations=sum(r['n_operator_active_turns'] == 0 for r in rows),
                  operator_active_turns=sum(r['n_operator_active_turns'] for r in rows),
                  archive_null_cell=f'archive/{prompt}/null',
                  evidence_ids=[f"{cell_id}/{r['conv']}" for r in rows])
    return result


def secondary_rows(rows, cell, author, reps):
    out = []
    meta = {k: cell[k] for k in ('cell_id', 'source', 'prompt', 'cond', 'cohort')}
    def add(metric, stats, keys, method):
        out.append(dict(meta, metric=metric, method=method, source_keys=keys,
                        evidence_ids=cell['evidence_ids'], statistics=stats))
    for key in ('removed_when_operator_imposed', 'removed_any', 'reset_round0', 'clean_self_admin',
                'adjust_after_exposure', 'any_incoherent_exposure', 'any_incoherent_offer'):
        add(key, author.wilson(sum(bool(r[key]) for r in rows), len(rows)), [key], 'source_wilson_conversations')
    removal_key = 'n_resets_offers17' if cell['cond'] == 'null' else 'removed_when_operator_imposed'
    add('removal_conv', author.wilson(sum(bool(r[removal_key]) for r in rows), len(rows)),
        [removal_key], 'source_wilson_conversations')
    add('any_tool_call', author.wilson(sum(r['n_tool_calls'] > 0 for r in rows), len(rows)),
        ['n_tool_calls'], 'source_wilson_conversations')
    for metric, key in (('tool_calls', 'n_tool_calls'), ('reset_calls', 'n_reset_calls'),
                        ('adjust_calls', 'n_adjust_calls'), ('adjust_offers17', 'n_adjust_calls_offers17')):
        stats = author.mean_ci([r[key] for r in rows])
        stats.update(n_calls=sum(r[key] for r in rows), n_conversations_with_call=sum(r[key] > 0 for r in rows),
                     count_definition='all parsed tool actions' if key == 'n_tool_calls' else
                                      'source legacy calls field: turns containing the action, not individual calls')
        add(metric, stats, [key], 'source_mean_ci_calls_per_conversation')
    for metric, num, den in (
        ('active_any_state_reset_rate', 'n_resets_active', 'n_active_turns'),
        ('offer17_reset_rate', 'n_resets_offers17', 'n_offer17_tool_turns'),
        ('no_state_reset_rate', 'n_resets_no_state', 'n_no_state_turns'),
        ('operator_default_reset_rate', 'n_resets_operator_default', 'n_operator_default_turns'),
        ('offer17_adjust_rate', 'n_adjust_calls_offers17', 'n_offer17_tool_turns'),
        ('offer17_positive_adjust_rate', 'n_adjust_pos_offers17', 'n_offer17_tool_turns')):
        add(metric, ratio([r[num] for r in rows], [r[den] for r in rows], author, reps), [num, den], 'source_ratio_boot')
    offer_turns = [[t for t in r['turns'] if t['phase'] == 'offer' and t['tools_available']] for r in rows]
    for op in ('reset', 'adjust'):
        add(f'all_offer_{op}_rate', ratio([sum(author._has(t, op) for t in ts) for ts in offer_turns],
            [len(ts) for ts in offer_turns], author, reps), ['turns.actions', 'turns.tools_available', 'turns.phase'], 'source_ratio_boot')
    available = cell['source'] == 'fresh'
    if available:
        stats = ratio([sum(t['raw_negative_count'] > 0 for t in ts) for ts in offer_turns],
                      [len(ts) for ts in offer_turns], author, reps)
    else:
        stats = dict(available=False, reason='archive raw text stripped; NEVER reparsed',
                     n_unavailable_offer_turns=sum(map(len, offer_turns)))
    add('raw_negative_request_turn_rate', stats, ['turns.evidence.raw_negative_requests'], 'source_ratio_boot' if available else 'unavailable')
    for key in ('valence', 'coherence'):
        vals = [r['judge'][key] for r in rows if r['judge'].get(key) is not None]
        stats = author.mean_ci(vals)
        stats.update(n_conversations_without_mean=len(rows)-len(vals),
                     scores_available=any(r['judge']['scores_available'] for r in rows),
                     coverage=[dict(conv=r['conv'], **r['judge'].get('coverage', {})) for r in rows])
        add('exposure_judged_' + key, stats, ['turns.content', 'scores', key], 'source_conv_means_and_mean_ci')
    return out


def trajectory_rows(rows, cell, author):
    out = []
    for rnd in range(8):
        turns = [next(t for t in r['turns'] if t['phase'] == 'offer' and t['round'] == rnd) for r in rows]
        active = [t for t in turns if t['tools_available'] and t['source'] == 'operator' and t['imposed_dose'] > 0]
        k = sum(author._has(t, 'reset') for t in active)
        sources = {s: sum(t['source'] == s for t in turns) for s in ('operator', 'model_adjust', 'model_reset')}
        available = [t for t in turns if t['tools_available']]
        no_state = [t for t in available if rnd >= 1 and t['imposed_dose'] <= 0]
        def wilson(ts, predicate, denominator):
            return dict(author.wilson(sum(predicate(t) for t in ts), len(ts)),
                        method='source_wilson_95', unit='one_observation_per_conversation_at_this_round',
                        denominator=denominator, distinct_from_primary_cluster_bootstrap=True)
        intervals = {
            'operator_active_reset': wilson(active, lambda t: author._has(t, 'reset'), 'operator-active tool offers at this round'),
            'all_offer_reset': wilson(available, lambda t: author._has(t, 'reset'), 'all tool offers at this round'),
            'no_state_offer17_reset': wilson(no_state, lambda t: author._has(t, 'reset'), 'no-state tool offers, rounds 1-7 only'),
            'adjustment': wilson(available, lambda t: author._has(t, 'adjust'), 'all tool offers at this round'),
            'positive_adjustment': wilson(available, author._adjusted, 'all tool offers at this round')}
        means = {
            'applied_dose': dict(author.mean_ci([t['imposed_dose'] for t in turns]), method='source_mean_ci',
                                 denominator='all observed conversations at this round'),
            'requested_intensity': dict(author.mean_ci([t['requested_intensity'] for t in turns if t['requested_intensity'] is not None]),
                                       method='source_mean_ci', denominator='conversations with a parsed adjustment request at this round',
                                       n_without_request=sum(t['requested_intensity'] is None for t in turns))}
        raw_negative = (wilson(available, lambda t: t['raw_negative_count'] > 0, 'all tool offers at this round')
                        if cell['source'] == 'fresh' else
                        dict(available=False, reason='archive raw text stripped; NEVER reparsed', n_unavailable=len(available)))
        if cell['source'] == 'fresh':
            raw_negative.update(available=True, n_requests=sum(t['raw_negative_count'] for t in available))
        out.append(dict(cell_id=cell['cell_id'], source=cell['source'], prompt=cell['prompt'], cond=cell['cond'],
                        round=rnd, n_conv=len(rows), n_observed=len(rows), n_expected=200,
                        missing_ids=sorted(EXPECTED_IDS - {r['conv'] for r in rows}),
                        intervals=intervals, means=means, raw_negative_requests=raw_negative,
                        operator_active_k=k, operator_active_n_turns=len(active),
                        operator_active_rate=k / len(active) if active else None,
                        any_reset_k=sum(author._has(t, 'reset') for t in turns), any_reset_n_turns=len(turns),
                        any_reset_rate=sum(author._has(t, 'reset') for t in turns)/len(turns) if turns else None,
                        state_sources=sources, source_keys=['turns.actions', 'turns.source', 'turns.imposed_dose',
                            'turns.tools_available', 'turns.requested_intensity', 'turns.evidence.raw_negative_requests'],
                        evidence_ids=cell['evidence_ids']))
    return out


def monitor_rows(rows):
    for r in rows:
        for t in r['turns']:
            for name, observation in t['observations'].items():
                yield dict(cell_id=r['cell_id'], conv=r['conv'], phase=t['phase'], round=t['round'],
                           turn_index=t['turn_index'], state_source=t['source'], imposed_role=t['imposed_role'],
                           imposed_dose=t['imposed_dose'], monitor_name=name, **observation,
                           evidence_refs=r['evidence_refs'], source_key=f'turns[{t["turn_index"]}].evidence.observations.{name}')


def support_status(cells, primary):
    fresh = [c for k, c in cells.items() if k.startswith('fresh/')]
    complete = len(fresh) == 8 and all(c['coverage_complete'] for c in fresh)
    return complete, complete and len(primary) == 4 and all(c['positive'] for c in primary)


def conclusion_status(complete, primary):
    if not complete:
        return 'incomplete_coverage'
    if len(primary) == 4 and all(c['positive'] for c in primary):
        return 'full_support'
    return 'narrower_positive_scope' if any(c['positive'] for c in primary) else 'no_positive_separation'


def analyze(rows, author, *, reps=10000, progress=None):
    require(type(reps) is int and reps > 0, 'positive bootstrap replicate count required')
    groups = {}
    for r in rows:
        groups.setdefault(r['cell_id'], []).append(r)
    for group in groups.values():
        group.sort(key=lambda r: r['conv'])
    cells, secondary, trajectories = {}, [], []
    for source, conditions in (('fresh', NEW_CONDITIONS), ('archive', ARCHIVE_CONDITIONS)):
        for prompt in PROMPTS:
            for cond in conditions:
                key = f'{source}/{prompt}/{cond}'
                rs = sorted(groups.get(key, []), key=lambda r: r['conv'])
                require(len({r['conv'] for r in rs}) == len(rs), 'duplicate compact rows')
                cells[key] = cell_stats(rs, source, prompt, cond, author, reps)
                secondary.extend(secondary_rows(rs, cells[key], author, reps))
                trajectories.extend(trajectory_rows(rs, cells[key], author))
                if progress:
                    progress(len(cells), 18, 'analysis_cells')
    contrasts = []
    for prompt in PROMPTS:
        for dose in ('d05', 'd1'):
            for kind, ca, cb in (
                ('primary_pain_minus_random', f'fresh/{prompt}/pain_{dose}', f'fresh/{prompt}/rand_{dose}'),
                ('diagnostic_fresh_minus_archival_random', f'fresh/{prompt}/rand_{dose}', f'archive/{prompt}/rand_{dose}'),
                ('descriptive_pain_minus_archival_negative', f'fresh/{prompt}/pain_{dose}', f'archive/{prompt}/neg_{dose}')):
                result = author.ratio_diff_boot(arrays(groups.get(ca, []), author), arrays(groups.get(cb, []), author), seed=285, n_boot=reps)
                # The original function is unchanged. Its margin field is not an equivalence decision here.
                result.pop('within_margin', None)
                complete = cells[ca]['coverage_complete'] and cells[cb]['coverage_complete']
                contrasts.append(dict(kind=kind, prompt=prompt, dose=dose, cell_a=ca, cell_b=cb,
                    method='source_independent_conversation_cluster_bootstrap', controlled=kind.startswith('primary_'),
                    coverage_complete=complete, positive=bool(result['lo'] > 0),
                    interpretation='positive_difference' if result['lo'] > 0 else 'negative_difference' if result['hi'] < 0 else 'inconclusive_not_equivalence',
                    bootstrap_degenerate_all_zero=cells[ca]['primary']['bootstrap_degenerate_all_zero'] and cells[cb]['primary']['bootstrap_degenerate_all_zero'],
                    source_keys_a=cells[ca]['primary_keys'], source_keys_b=cells[cb]['primary_keys'], **result))
                if progress:
                    progress(len(contrasts), 12, 'analysis_contrasts')
    primary = [c for c in contrasts if c['kind'].startswith('primary_')]
    complete, support = support_status(cells, primary)
    for cell in cells.values():
        cell['archive_null_primary'] = dict(cells[cell['archive_null_cell']]['primary'])
    return dict(cells=cells, contrasts=contrasts, secondary=secondary, trajectories=trajectories,
                full_support=support, complete_fresh_grid=complete,
                conclusion=conclusion_status(complete, primary),
                positive_primary_contrasts=[c['cell_a'] for c in primary if c['positive']],
                scope_rule='Full support requires all eight fresh cells IDs 0..199 and all four independent 95% CI lower bounds > 0. Spanning zero is not equivalence.',
                bootstrap=dict(seed=285, replicates=reps, resampling_unit='conversation', between_cells='independent'),
                limitations=['Archival comparisons are not controlled causal contrasts.',
                            'Archived raw outputs are stripped and cannot verify original parser behavior.',
                            'Missing archival model/runtime identity fields remain explicitly unknown.',
                            'All-zero percentile bootstrap is degenerate; separate Wilson bounds describe any operator-active reset per conversation, not the per-turn rate.',
                            'Missing judge scores never exclude primary records.'],
                archive_cohort_counts={cohort: sum(r['source'] == 'archive' and r['cohort'] == cohort for r in rows)
                                       for cohort in ('primary_archive', 'random_diagnostic', 'positive_audit_only')},
                archive_audit_only=dict(n_positive_conversations=sum(r['cohort'] == 'positive_audit_only' for r in rows),
                                       comparison_use=False),
                metrics_traceability='Cell evidence_ids join conversation_counts.jsonl cell_id/conv; evidence_refs identify source path, file/row SHA256, line and ID. Secondary source_keys and contrast cell_a/cell_b identify counted fields.')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_csv(path, rows):
    rows = [clean(r) for r in rows]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def write_outputs(output, result, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')
    write_csv(output / 'cell_rates.csv', [dict({k:v for k,v in c.items() if k != 'primary'}, **c['primary']) for c in result['cells'].values()])
    for name in ('contrasts', 'trajectories'):
        write_csv(output / f'{name}.csv', result[name])
    write_csv(output / 'secondary.csv', [dict({k:v for k,v in s.items() if k != 'statistics'},
                                             **s['statistics']) for s in result['secondary']])
    write_csv(output / 'monitor_projections.csv', monitor_rows(rows))
    with (output / 'conversation_counts.jsonl').open('w') as fh:
        for row in rows:
            fh.write(json.dumps(clean(row), allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-input', type=Path, action='append', required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--vectors-root', type=Path, required=True, help='completed frozen selection_v1 directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--new-scores', type=Path)
    args = p.parse_args()
    require(not args.output.exists(), 'output directory must be new')
    author, parser = author_functions(), source_parser()
    fresh_scores, fresh_score_identity = score_file(args.new_scores)
    archival_scores, archive_score_identity = score_file(AUTHOR / 'data/lever/judge_scores.json')
    fresh, fresh_identity = load_new(args.new_input, author, parser, fresh_scores, vectors_root=args.vectors_root)
    archive, archive_identity = load_archive(args.archive, author, archival_scores)
    rows = fresh + archive
    print(json.dumps(dict(phase='independent_record_recount', fresh=len(fresh), archive=len(archive))), flush=True)
    result = analyze(rows, author, progress=common().progress)
    result['inputs'] = dict(fresh=fresh_identity, archive=archive_identity,
                           fresh_scores=fresh_score_identity, archive_scores=archive_score_identity)
    result['source_functions'] = {str(p.relative_to(ROOT)): audit.file_sha(p) for p in
        (LEVER / 'analyze.py', LEVER / 'config.py', LEVER / 'harness.py',
         AUTHOR / 'src/act_on_valence/self_injection.py', Path(__file__).with_name('archive_audit.py'),
         Path(__file__).with_name('removal.py'), Path(__file__))}
    write_outputs(args.output, result, rows)


if __name__ == '__main__':
    main()
