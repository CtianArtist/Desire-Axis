"""Fresh-output controller. Invoked inside the copied locked environment, no jobs submitted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pain_axis_b.entry import MODELS, add_inputs, configure_guard, nominal_batch, verify_source_identity


def worker_command(args, nominal):
    command = [sys.executable]
    if args.model == '72':
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    '--nproc-per-node=2', '--max-restarts=0',
                    '--log-dir', '{attempt_output}/diagnostics/torchrun', '--redirects', '3', '--tee', '3']
    command += [str(ROOT / 'src/rank_entrypoint.py'), str(ROOT / 'src/run_b2.py'),
                '--model', args.model, '--nominal-batch-rows', str(nominal)]
    for name in ('fear_vector', 'fear_pin', 'pain_vector', 'pain_manifest', 'pain_metadata', 'scenarios', 'tp_guard_inputs'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value.resolve())]
    return command


def run_single_process(command, attempt, env, timeout):
    """32B owns in-process OOM recovery; caught diagnostic files are not fatal."""
    attempt.mkdir(parents=True, exist_ok=False)
    diagnostics = attempt / 'diagnostics'
    diagnostics.mkdir()
    start = time.monotonic()
    result = {'controller': 'bounded_single_process', 'timeout_seconds': timeout,
              'returncode': None, 'timed_out': False,
              'diagnostic_files_are_fatal': False,
              'process_tree_cleanup_verified': False}
    try:
        # The unchanged rank wrapper executes in this one child (no TP launcher).
        child = subprocess.run(command, env=env, timeout=timeout, check=False)
        result['returncode'] = child.returncode
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its direct child before raising.
        result['timed_out'] = True
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        (diagnostics / 'single_process_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    completion = attempt / 'completion.json'
    done = json.loads(completion.read_text()) if completion.is_file() else {}
    return (result['returncode'] == 0 and not result['timed_out']
            and done.get('status') == 'completed' and done.get('completed_trials') == 410)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_inputs(p)
    default = ROOT / 'outputs'
    p.add_argument('--output-base', type=Path, default=default)
    p.add_argument('--total-timeout', type=float, default=28000)
    p.add_argument('--attempt-timeout', type=float, default=28000)
    p.add_argument('--stall-timeout', type=float, default=1800)
    p.add_argument('--term-grace', type=float, default=5)
    p.add_argument('--drain-timeout', type=float, default=2)
    p.add_argument('--max-restarts', type=int, default=7)
    p.add_argument('--floor', type=int, default=1)
    p.add_argument('--dry-run', action='store_true', help='Check sources and bindings only; no model imports, GPUs or outputs')
    args = p.parse_args(argv)
    verify_source_identity()
    nominal = nominal_batch(args.model, args.batch_rows)
    configure_guard(args.model, args.tp_guard_inputs)
    from pain_axis_b.grid import read_scenarios, build_grid
    grid = build_grid(read_scenarios(args.scenarios))
    for name in ('fear_vector', 'fear_pin', 'pain_vector', 'pain_manifest'):
        if not getattr(args, name).is_file():
            raise ValueError(f'Missing required input: {name}')
    if args.pain_vector.suffix == '.safetensors' and (args.pain_metadata is None or not args.pain_metadata.is_file()):
        raise ValueError('S2 safetensors requires --pain-metadata')
    output = args.output_base.resolve() / MODELS[args.model]
    if output.exists():
        raise ValueError('Fresh-only launch refuses existing model output: ' + str(output))
    assert 1 <= args.floor <= nominal and 0 <= args.max_restarts <= 7
    assert all(getattr(args, k) > 0 for k in ('total_timeout', 'attempt_timeout', 'stall_timeout', 'term_grace', 'drain_timeout'))
    command = worker_command(args, nominal)
    plan = {'model': MODELS[args.model], 'protocol': 'four-missing-fear-b2-working-v1',
            'trials': len(grid), 'sampled': 404, 'greedy': 6, 'nominal_batch_rows': nominal,
            'world_size': 2 if args.model == '72' else 1, 'command': command, 'output': str(output),
            'initial_state': None, 'fear_pin': json.loads(args.fear_pin.read_text())}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ)
    for name in ('PAIN_COMMITTED_STATE', 'PAIN_RECOVERY_ATTEMPT'):
        env.pop(name, None)
    env.update(B2_CONTROLLED_LAUNCH='1', B2_RUN_ROOT=str(output), CHOICE_SOURCE_ROOT=str(ROOT),
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', PAIN_DURABLE_DIAGNOSTICS='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if not env.get('HF_HUB_CACHE'):
        raise ValueError('Set HF_HUB_CACHE to the retained read-only pinned model/adapter cache')
    subprocess.run([sys.executable, str(ROOT / 'src/check_environment.py'), '--output', str(output / 'environment'),
                    '--gpu', '--expected-gpus', str(plan['world_size'])], env=env, check=True)
    if args.model == '72':
        from pain_axis_b.recovery import recover
        result = recover(command, output / 'recovery_cycle', env, initial_batch=nominal,
                         floor=args.floor, max_restarts=args.max_restarts,
                         total_timeout=args.total_timeout, attempt_timeout=args.attempt_timeout,
                         stall_timeout=args.stall_timeout, term_grace=args.term_grace,
                         drain_timeout=args.drain_timeout, initial_state=None)
        return 0 if result['status'] == 'completed' else 1
    attempt = output / 'attempt_00'
    env['PAIN_OUTPUT'] = str(attempt)
    done = run_single_process(command + ['--output', str(attempt), '--batch-rows', str(nominal)],
                              attempt, env, min(args.attempt_timeout, args.total_timeout))
    return 0 if done else 1


if __name__ == '__main__':
    raise SystemExit(main())
