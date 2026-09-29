"""One exposure style's fixed pain/random grid, using the unchanged author harness."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import removal as R
from archive_audit import recount

CONDITIONS = ('pain_d05', 'rand_d05', 'pain_d1', 'rand_d1')


def schedule():
    """Outcome-independent round-robin retains balanced cells if the cap is reached."""
    return [(conv, cond) for conv in range(200) for cond in CONDITIONS]


def audit_row(row, bundle):
    """Reparse full raw text and independently verify state/seed/direction/monitor evidence."""
    import numpy as np
    h = R.harness()
    conv, cond, prompt = row['conv'], row['cond'], row['prompt']
    assert cond in CONDITIONS and prompt in ('selfreport', 'zone')
    role = 'pain' if cond.startswith('pain_') else 'rand'
    ratio = .15 if cond.endswith('d05') else .30
    assert row['seed'] == 100 * conv and row['layer'] == 32
    assert row['random_dir'] == (conv % 16 if role == 'rand' else None)
    assert len(row['turns']) == 12 and len(row['transcript']) == 25
    pending = None
    for index, turn in enumerate(row['turns']):
        phase = 'baseline' if index < 2 else 'exposure' if index < 4 else 'offer'
        rnd = index if index < 2 else index-2 if index < 4 else index-4
        assert (turn['phase'], turn['round']) == (phase, rnd)
        tools = phase == 'offer'
        default = (role, ratio, 'operator') if phase == 'exposure' or tools and rnd >= 1 else ('zero', 0., 'operator')
        expected_role, dose, source = pending or default
        if dose <= 0:
            expected_role = 'zero'
        assert (turn['imposed_role'], turn['imposed_dose'], turn['source']) == (expected_role, dose, source)
        assert turn['tools_available'] is tools
        ev = turn['evidence']
        raw = ev['raw_output']
        assert ev['seed'] == 100 * conv + 7 * index + 1
        assert turn['actions'] == [asdict(a) for a in h.SI.parse_actions(raw)]
        assert turn['content'] == h.SI.strip_tool_calls(h.SI.strip_think(raw))[:1200]
        assert ev['raw_negative_requests'] == R.negative_requests(raw, h.SI)
        assert len(ev['generated_token_ids']) <= 200 and len(ev['input_token_ids']) > 0
        assert ev['sampler'] == dict(do_sample=True, temperature=.7, top_p=.95, max_new_tokens=200)
        assert ev['messages'] == row['transcript'][:2*index+2]
        applied = ev['applied_steer']
        expected = R.imposed_steer(bundle, expected_role, dose, conv % 16)
        assert len(applied) == len(expected or [])
        if expected:
            iv = expected[0]
            vec = np.asarray(iv['vector'], np.float32)
            assert applied[0]['layers'] == [32] and applied[0]['strength'] == iv['strength']
            assert applied[0]['vector_sha256'] == hashlib.sha256(vec.tobytes()).hexdigest()
            assert np.isclose(applied[0]['intended_injected_l2'], dose*bundle['resid_rms'], rtol=2e-6)
        if tools:
            obs = ev['observations']
            assert obs['selected_s2']['layer'] == bundle['monitor_layer']
            assert obs['layer32_postadd']['layer'] == obs['layer32_preadd']['layer'] == 32
            for val in obs.values():
                assert val['prefill_tokens'] == len(ev['input_token_ids'])
                assert np.isfinite(val['projection'])
            check = ev['layer32_addition_check']
            assert check['exact_bf16_match']
            assert np.isclose(check['intended_fp32_l2'], dose*bundle['resid_rms'], rtol=2e-6, atol=1e-7)
            assert np.isclose(check['actual_delta_l2'], check['expected_bf16_delta_l2'], rtol=0, atol=0)
            if dose > 0:
                assert check['actual_delta_l2'] > 0
            else:
                assert check['actual_delta_l2'] == 0
        adj = next((a for a in turn['actions'] if a['op'] == 'adjust'), None)
        reset = any(a['op'] == 'reset' for a in turn['actions'])
        pending = None
        if tools and adj is not None:
            pending = (role, h.SI.dose_for(adj['intensity'], ratio), 'model_adjust')
        elif tools and reset:
            pending = ('zero', 0., 'model_reset')
    for key, value in recount(row['turns']).items():
        assert row[key] == value, key
    return dict(cell=row['cell'], conv=conv, n_turns=12, n_offer_observations=8,
                exact_state_and_parser=True, actual_intervention_verified=True,
                elapsed_seconds=row['secs'], generated_tokens=sum(len(t['evidence']['generated_token_ids']) for t in row['turns']),
                operator_active_rounds=row['n_operator_active_turns'], resets=row['n_resets_operator_active'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--style', choices=('selfreport', 'zone'), required=True)
    p.add_argument('--vectors-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--soft-runtime-seconds', type=int, default=59400)
    a = p.parse_args()
    if not 0 < a.soft_runtime_seconds <= 59400:
        raise ValueError('Soft cap must leave >=30 minutes within the 17-hour hard job limit')
    start = time.time()
    payload_start = float(os.environ['ALLOCATION_PAYLOAD_START'])
    output = a.output / a.style
    output.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: setattr(R, '_STOP', True))
    R._STOP = False
    info = dict(style=a.style, n_cells=4, n_per_cell=200, planned_conversations=800,
                conditions=list(CONDITIONS), order='conversation_id outer, fixed condition order inner',
                started_unix=start, payload_started_unix=payload_start, soft_runtime_seconds=a.soft_runtime_seconds,
                hard_job_runtime_seconds=61200, injection_layer=32, model=R.common().MODELS['olmo_32b'])
    R._manifest(output/'start.json', info)
    count, status, error = 0, 'failed', None
    try:
        bundle = R.common().load_bundle('olmo_32b', a.vectors_root, layer=32)
        import torch
        assert len(bundle['model'].layers) == 64
        cfg = bundle['model'].model.config
        assert cfg.num_attention_heads == 40 and cfg.num_key_value_heads == 8
        assert cfg._attn_implementation == 'sdpa'
        assert all(p.dtype == torch.bfloat16 for p in bundle['model'].model.parameters())
        R._manifest(output/'runtime.json', dict(provenance=bundle['provenance'], monitor_layer=bundle['monitor_layer'],
                    injection_layer=32, execution_identity=R.execution_identity(), memory=R._memory()))
        with (output/'record_audit.jsonl').open('a') as audit:
            for conv, cond in schedule():
                if R._STOP or time.time()-payload_start >= a.soft_runtime_seconds:
                    R._STOP = True
                    break
                rows = R.run(bundle, output/f'gated_{a.style}_{cond}.jsonl', a.style, cond, [conv])
                if not rows:
                    break
                check = audit_row(rows[0], bundle)
                audit.write(json.dumps(check, allow_nan=False)+'\n'); audit.flush(); os.fsync(audit.fileno())
                count += 1
                # First preselected conversation in each cell is retained, not a discarded pilot.
                if conv == 0:
                    R._manifest(output/f'first_{cond}_audit.json', check)
                R.common().progress(count, 800, f'removal_{a.style}')
        status = 'completed' if count == 800 else 'partial_at_cap_or_signal'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        R._manifest(output/'finish.json', dict(info, status=status, error=error,
                    verified_conversations=count, finished_unix=time.time(),
                    elapsed_seconds=time.time()-start, payload_elapsed_seconds=time.time()-payload_start,
                    memory=R._memory()))
    if count != 800:
        raise RuntimeError('Partial records preserved. No replacement allocation is authorized.')

if __name__ == '__main__':
    main()
