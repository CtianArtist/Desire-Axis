"""Fresh-process recovery of committed trial state; no model imports or execution here.

Only an explicit generate_segment OOM receipt permits halving. A timeout, native
abort, unknown failure, logging failure, or unrelated exception stops the cycle.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from pain_axis_b.diagnostics import atomic_write


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def state_summary(path):
    """Read only after the complete child tree is dead. SQLite rolls back hot journals."""
    path = Path(path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [('ok',)]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'trials', 'meta'} <= tables, "Incomplete authoritative state schema"
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        rows = list(db.execute("SELECT id,state,done,record FROM trials ORDER BY id"))
    assert len({r[0] for r in rows}) == len(rows), "Duplicate committed trial IDs"
    for tid, state, done, record in rows:
        payload = json.loads(state)
        assert payload['trial_id'] == tid and bool(payload['done']) == bool(done)
        assert payload['record'] == json.loads(record)
        assert 'generator' in payload
    return {'meta': meta, 'rows': rows,
            'logical_sha256': hashlib.sha256(json.dumps([meta, rows], sort_keys=True).encode()).hexdigest(),
            'completed_ids': [r[0] for r in rows if r[2]], 'row_count': len(rows)}


def snapshot_state(source, destination):
    """Backup a stopped worker's committed DB, not a raw live-file copy."""
    source, destination = Path(source), Path(destination)
    before = state_summary(source)
    if before is None:
        return None
    assert not destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    with destination.open('rb') as f:
        os.fsync(f.fileno())
    after = state_summary(destination)
    assert before == after, 'Backup changed committed state'
    return after


def assert_preserved(previous, current):
    if previous is None:
        return
    assert current is not None, 'Previously committed state is missing'
    assert previous['meta']['grid_hash'] == current['meta']['grid_hash'], 'Grid identity changed'
    assert current['meta']['schedule']['choices'] >= previous['meta']['schedule']['choices'], 'Committed cursor regressed'
    now = {r[0]: r for r in current['rows']}
    for row in previous['rows']:
        assert row[0] in now, 'Committed trial disappeared'
        if row[2]:
            assert now[row[0]] == row, 'Completed trial was changed or repeated'


def retry_reason(attempt, receipt):
    """A recognized OOM is necessary; missing evidence never becomes permission."""
    if (not receipt.get('cleanup_verified') or receipt.get('logging_errors')
            or receipt.get('cause') not in {'failure_signal', 'child_nonzero'}):
        return None
    files = list((attempt / 'diagnostics').glob('rank_*/exception.json'))
    # The scoped original exception receipt is atomic. The secondary marker is
    # supplementary, so killing between the two writes cannot lose eligibility.
    errors = [json.loads(p.read_text()) for p in files]
    if not files:  # Legacy marker-only CPU fixtures; no untyped process error accepted.
        files = list((attempt / 'diagnostics').glob('rank_*/recovery_failure.json'))
        errors = [json.loads(p.read_text()) for p in files]
    if not errors or any(e.get('scope') != 'generate_segment' or e.get('kind') != 'oom' for e in errors):
        return None
    # Other Python exceptions are fatal, except peer transport consequences.
    # Do not guess whether an arbitrary secondary exception is a transport consequence:
    # fail closed when a rank has an exception but no recognized original OOM marker.
    marked = {p.parent for p in files}
    if any(p.parent not in marked for p in (attempt/'diagnostics').glob('rank_*/exception.json')):
        return None
    rows = [e.get('rows') for e in errors]
    if not all(type(n) is int and n > 0 for n in rows) or len(set(rows)) != 1:
        return None
    return rows[0]


def recover(command, output, env, *, initial_batch, floor=1, max_restarts=7,
            total_timeout=28800, attempt_timeout=3600, stall_timeout=1800,
            term_grace=5, drain_timeout=2, initial_state=None):
    """Execute a finite cycle. The caller must separately authorize actual GPU use.

    Each attempt receives its own output tree and fresh processes. A copied SQLite
    snapshot is the ONLY continuation input. Existing attempt trees are never reused.
    Batch cap stays halved after a retry; this changes later RNG consumption paths.
    """
    from pain_axis_b.process_watchdog import supervise_attempt
    assert type(initial_batch) is int and 1 <= floor <= initial_batch
    assert type(max_restarts) is int and 0 <= max_restarts <= 7
    assert all(x > 0 for x in (total_timeout, attempt_timeout, stall_timeout, term_grace, drain_timeout))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    batch = initial_batch
    previous = None
    resume = None
    if initial_state is not None:
        resume = output/'snapshots'/'initial.sqlite'
        previous = snapshot_state(initial_state, resume)
        assert previous is not None, 'Requested initial state absent'
    result = {'status': 'stopped', 'attempts': [], 'initial_batch': initial_batch,
              'floor': floor, 'max_restarts': max_restarts,
              'total_timeout': total_timeout, 'batch_regrouping_changes_sampled_paths': True}
    try:
        for index in range(max_restarts + 1):
            remaining = total_timeout - (time.monotonic() - start)
            if remaining <= term_grace + drain_timeout + 2:
                result['reason'] = 'total_deadline'; break
            attempt = output / f'attempt_{index:02d}'
            args = [*[arg.replace('{attempt_output}', str(attempt)) for arg in command],
                    '--output', str(attempt), '--batch-rows', str(batch)]
            if resume is not None:
                args += ['--resume', str(resume)]
            attempt_env = {**env, 'PAIN_OUTPUT': str(attempt), 'PAIN_RECOVERY_ATTEMPT': str(index),
                           'PAIN_DURABLE_DIAGNOSTICS': '1'}
            receipt = supervise_attempt(args, attempt, attempt_env,
                attempt_timeout=min(attempt_timeout, remaining - term_grace - drain_timeout - 2),
                stall_timeout=stall_timeout, term_grace=term_grace, drain_timeout=drain_timeout)
            row = {'index': index, 'batch_rows': batch, 'output': str(attempt),
                   'resume': str(resume) if resume is not None else None, 'process': receipt}
            result['attempts'].append(row)
            if not receipt.get('cleanup_verified'):
                result['reason'] = 'cleanup_unverified'; break
            current = state_summary(attempt/'state.sqlite')
            assert_preserved(previous, current)
            row['committed_state'] = ({k: v for k, v in current.items() if k != 'rows'}
                                      if current is not None else None)
            completion = attempt/'completion.json'
            if receipt.get('returncode') == 0 and not receipt.get('cause') and completion.exists():
                done = json.loads(completion.read_text())
                assert done['status'] == 'completed'
                assert current is not None and len(current['completed_ids']) == done['completed_trials']
                assert current['row_count'] == done['completed_trials'], 'Not all initialized rows completed'
                result.update(status='completed', reason='completed', final_output=str(attempt)); break
            failed_rows = retry_reason(attempt, receipt)
            if failed_rows is None:
                result['reason'] = 'failure_not_retryable'; break
            if current is None:
                result['reason'] = 'missing_committed_state'; break
            if min(batch, failed_rows) <= floor or index == max_restarts:
                result['reason'] = 'retry_or_floor_limit'; break
            next_batch = max(floor, min(batch // 2, failed_rows // 2))
            assert next_batch < batch
            resume = output/'snapshots'/f'after_{index:02d}.sqlite'
            previous = snapshot_state(attempt/'state.sqlite', resume)
            row['next_batch_rows'] = next_batch
            row['snapshot'] = str(resume)
            batch = next_batch
            write_json(output/'recovery_receipt.json', result)
        else:
            result['reason'] = 'retry_limit'
    except BaseException as exc:
        result['reason'] = 'controller_error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        write_json(output/'recovery_receipt.json', result)
    return result
