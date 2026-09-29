"""Linux-only, stdlib subprocess supervision with bounded tree cleanup.

``supervise_attempt`` requires a fresh OUTPUT/diagnostics directory and runs in
its caller's main thread (temporary SIGINT/SIGTERM handlers). ``env`` is the
complete child environment, not an overlay. Binary launcher logs are exclusive.
Tee destinations must expose file descriptors (sockets use MSG_DONTWAIT;
other descriptors are reopened through /proc/self/fd). Tee failure fails the
attempt, never silently ignored.

A private subreaper guardian keeps double-fork/setsid children in the owned
ancestry. No caller-wide subreaper setting or process-group-wide kill is used.
PID/starttime checks precede every signal. Deadlines bound userspace polling,
pipe drain, and reap; Linux uninterruptible I/O can defeat wall-time bounds.
Only cleanup_verified=True establishes that no observed owned process is live
and that ownership was not lost. This module makes no retry/OOM decision.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time

_POLL = 0.025
_TEE_LIMIT = 1024 * 1024


def _pidfd_syscall(name, *args):
    """glibc <2.36 has no wrappers; use only the installed-header verified ABI.

    This is the same Linux interface, not a permission fallback. Kernel denial
    or lack of support is propagated and prevents any workload launch.
    """
    import platform
    if platform.machine() != 'x86_64':
        raise RuntimeError('Unverified pidfd syscall architecture')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    expected = {'pidfd_open': 434, 'pidfd_send_signal': 424}[name]
    lines = [s.split() for s in header.splitlines() if s.startswith('#define __NR_' + name + ' ')]
    if len(lines) != 1 or int(lines[0][-1]) != expected:
        raise RuntimeError('Unverified pidfd syscall definition')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(expected), *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_open'):
        return _pidfd_syscall('pidfd_open', ctypes.c_int(pid), ctypes.c_uint(0))
    fd = libc.pidfd_open(ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def _pidfd_signal(fd, signum):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, signum)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_send_signal'):
        return _pidfd_syscall('pidfd_send_signal', ctypes.c_int(fd), ctypes.c_int(signum),
                              ctypes.c_void_p(), ctypes.c_uint(0))
    if libc.pidfd_send_signal(ctypes.c_int(fd), ctypes.c_int(signum),
                              ctypes.c_void_p(), ctypes.c_uint(0)) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _proc(pid):
    """Return process identity, including Linux boot-relative start ticks."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text[text.rfind(")") + 2:].split()
    return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
            "pgid": int(fields[2]), "sid": int(fields[3]),
            "starttime": int(fields[19])}


def _snapshot():
    result = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                entry = _proc(int(name))
            except PermissionError:
                continue  # Other users' processes are not our descendants.
            if entry:
                result[entry["pid"]] = entry
    return result


def _same(a, b):
    return b is not None and a["pid"] == b["pid"] and a["starttime"] == b["starttime"]


def _live(entry):
    return entry is not None and entry["state"] not in ("Z", "X")


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".attempt_receipt.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _open_tee(stream):
    # Reopen rather than dup: O_NONBLOCK must not change the caller's shared
    # open-file description. O_APPEND preserves existing redirected log bytes.
    fd = stream.fileno()
    if stat.S_ISSOCK(os.fstat(fd).st_mode):
        return os.dup(fd)
    return os.open(f"/proc/self/fd/{fd}",
                   os.O_WRONLY | os.O_NONBLOCK | os.O_APPEND | os.O_CLOEXEC)


def _tee_write(fd, data):
    if stat.S_ISSOCK(os.fstat(fd).st_mode):
        stream = socket.socket(fileno=fd)
        try:
            return stream.send(data, socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL)
        finally:
            stream.detach()
    return os.write(fd, data)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("zero-byte log write")
        view = view[count:]


def _guardian(fd, command):
    """Private executable entrypoint. Do not call in the supervising process."""
    def emit(value):
        _write_all(fd, (json.dumps(value) + "\n").encode())

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
        # Caught handlers revert to default on exec; SIG_IGN would be inherited.
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        child = subprocess.Popen(command, close_fds=True)
        emit({"event": "started", "pid": child.pid, "process": _proc(child.pid)})
        rc = None
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == child.pid:
                rc = os.waitstatus_to_exitcode(status)
                child.returncode = rc
                emit({"event": "exited", "returncode": rc})
            if pid == 0:
                time.sleep(0.01)
        emit({"event": "empty", "returncode": rc})
        os.close(fd)
        return 0
    except BaseException as exc:
        try:
            emit({"event": "error", "error": repr(exc)})
        except BaseException:
            pass
        # Remain an ancestry anchor if failure happened after spawning.
        while True:
            time.sleep(1)


def supervise_attempt(command: list[str], output: Path, env: dict, *,
                      attempt_timeout: float, stall_timeout: float,
                      term_grace: float = 2, drain_timeout: float = 2) -> dict:
    """Run one fresh tree; return an atomic-receipt-shaped dictionary.

    Causes: None (clean success), failure_signal, child_nonzero, attempt_timeout,
    stall_timeout, logging_failure, interrupt, lingering_descendants,
    supervisor_error. original_returncode is the command's observed exit code
    before cleanup; final_returncode also includes exits observed during cleanup.
    kill_times_seconds and signal_events are relative to the monotonic start.
    failure_signals contains paths only; their contents remain authoritative.
    receipt_written=False means the returned receipt could not be persisted.
    """
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("watchdog requires Linux and the main thread")
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a nonempty list of strings")
    for name, value in (("attempt_timeout", attempt_timeout), ("stall_timeout", stall_timeout),
                        ("term_grace", term_grace), ("drain_timeout", drain_timeout)):
        if not math.isfinite(value) or value < 0 or (name.endswith("timeout") and value == 0):
            raise ValueError(f"invalid {name}")
    start = time.monotonic()
    deadline = start + attempt_timeout
    diagnostics = Path(output) / "diagnostics"
    receipt = {
        "schema_version": 1, "command": list(command), "cause": None,
        "original_returncode": None, "final_returncode": None,
        "attempt_timeout": attempt_timeout, "stall_timeout": stall_timeout,
        "term_grace": term_grace, "drain_timeout": drain_timeout,
        "deadline_basis": "monotonic supervise_attempt entry; never reset",
        "attempt_deadline_reached": False, "stall_deadline_reached": False,
        "elapsed_seconds": None, "cleanup_verified": False,
        "logging_errors": [], "cleanup_errors": [], "supervisor_errors": [],
        "failure_signals": [], "process_inventory": [], "signal_events": [],
        "kill_times_seconds": {"term": None, "kill": None},
        "receipt_written": False, "interrupt_signal": None,
    }
    child = None
    owned = {}
    logs, tees, queues, sources = {}, {}, {}, {}
    selector = selectors.DefaultSelector()
    control = b""
    control_read = control_write = None
    anchor = None
    guardian_empty = False
    ownership_lost = False
    anchor_killed_safely = False
    logging_ok = True
    failed_logs = set()
    handlers = {}
    interrupted = None
    progress = {}
    last_progress = start
    exit_seen = None
    created = False

    def fail(cause):
        if receipt["cause"] is None:
            receipt["cause"] = cause
            receipt["original_returncode"] = receipt["final_returncode"]

    def log_error(exc):
        nonlocal logging_ok
        receipt["logging_errors"].append(repr(exc))
        logging_ok = False
        fail("logging_failure")

    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum  # No exception can interrupt tree teardown.

    def discover():
        nonlocal ownership_lost
        snap = _snapshot()
        if anchor is None:
            return []
        roots = {pid for pid, entry in owned.items() if _same(entry, snap.get(pid))}
        changed = True
        while changed:
            changed = False
            for pid, entry in snap.items():
                if pid not in roots and entry["ppid"] in roots:
                    roots.add(pid)
                    changed = True
        now = time.monotonic() - start
        for pid in roots:
            entry = snap[pid]
            if pid not in owned:
                owned[pid] = dict(entry, first_seen_seconds=now)
            elif not _same(owned[pid], entry):
                ownership_lost = True
                continue
            owned[pid].update(state=entry["state"], ppid=entry["ppid"],
                              pgid=entry["pgid"], sid=entry["sid"], last_seen_seconds=now)
        return [entry for pid, entry in snap.items()
                if pid in owned and _same(owned[pid], entry) and _live(entry)]

    def send(entry, signum):
        try:
            current = _proc(entry["pid"])
            if _same(entry, current) and _live(current):
                # pidfd closes the stat-check/kill PID-reuse race on modern Linux.
                fd = _pidfd_open(entry["pid"])
                try:
                    if _same(entry, _proc(entry["pid"])):
                        _pidfd_signal(fd, signum)
                        receipt["signal_events"].append({"pid": entry["pid"],
                            "starttime": entry["starttime"], "signal": signum,
                            "seconds": time.monotonic() - start})
                finally:
                    os.close(fd)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            receipt["cleanup_errors"].append(repr(exc))

    def control_events(data):
        nonlocal control, guardian_empty, exit_seen
        control += data
        while b"\n" in control:
            line, control = control.split(b"\n", 1)
            event = json.loads(line)
            if event["event"] == "started":
                receipt["launcher_pid"] = event["pid"]
                entry = event.get("process")
                if entry and entry["pid"] not in owned:
                    owned[entry["pid"]] = dict(entry, first_seen_seconds=time.monotonic() - start)
            elif event["event"] == "exited":
                receipt["final_returncode"] = event["returncode"]
                exit_seen = time.monotonic()
            elif event["event"] == "empty":
                guardian_empty = True
            elif event["event"] == "error":
                receipt["supervisor_errors"].append(event["error"])
                fail("supervisor_error")

    def pump(wait=0):
        # One bounded read per ready pipe; a noisy child cannot starve deadlines.
        for key, _ in selector.select(wait):
            kind = key.data
            try:
                data = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            except OSError as exc:
                if kind == "control":
                    raise
                selector.unregister(key.fd)
                log_error(exc)
                continue
            if not data:
                selector.unregister(key.fd)
                continue
            if kind == "control":
                control_events(data)
            else:
                # A broken tee must not discard otherwise writable diagnostics.
                if kind not in failed_logs:
                    try:
                        _write_all(logs[kind], data)
                    except BaseException as exc:
                        failed_logs.add(kind)
                        log_error(exc)
                if logging_ok:
                    queues[kind].extend(data)
                    if len(queues[kind]) > _TEE_LIMIT:
                        log_error(OSError("tee backlog exceeded bounded buffer"))
        if logging_ok:
            for kind, queue in queues.items():
                if queue:
                    try:
                        count = _tee_write(tees[kind], queue[:65536])
                        if count == 0:
                            raise OSError("zero-byte tee write")
                        del queue[:count]
                    except BlockingIOError:
                        pass
                    except BaseException as exc:
                        log_error(exc)

    def diagnostic_poll():
        nonlocal last_progress, progress
        failures = list(diagnostics.glob("rank_*/exception.json"))
        failures += list(diagnostics.glob("rank_*/recovery_failure.json"))
        if failures:
            receipt["failure_signals"] = sorted(str(p) for p in failures)
            fail("failure_signal")
        current = {}
        for path in diagnostics.glob("rank_*/latest_stage.json"):
            try:
                st = path.stat()
            except FileNotFoundError:
                continue
            current[str(path)] = (st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)
        if current != progress:
            last_progress = time.monotonic()
            progress = current

    def teardown():
        nonlocal ownership_lost, anchor_killed_safely
        if child is None:
            return
        receipt["kill_times_seconds"]["term"] = time.monotonic() - start
        term_end = time.monotonic() + term_grace
        termed = set()
        while True:
            pump()
            live = discover()
            for entry in live:
                identity = (entry["pid"], entry["starttime"])
                if identity not in termed:
                    send(entry, signal.SIGTERM)
                    termed.add(identity)
            if not live or time.monotonic() >= term_end:
                break
            time.sleep(_POLL)
        if live:
            receipt["kill_times_seconds"]["kill"] = time.monotonic() - start
            # Freeze known parents, then discover children born during the scan.
            # Repeat under a finite bound before killing; no killpg can hit peers.
            freeze_end = time.monotonic() + min(0.5, drain_timeout / 2)
            while True:
                live = discover()
                running = [e for e in live if e["state"] not in ("T", "t")]
                if not running:
                    anchor_killed_safely = any(_same(anchor, e) for e in live)
                    break
                for entry in running:
                    send(entry, signal.SIGSTOP)
                if time.monotonic() >= freeze_end:
                    ownership_lost = True
                    break
                time.sleep(0.005)
            # Keep anchor until every possible descendant has received KILL.
            for entry in sorted(live, key=lambda e: e["pid"] == anchor["pid"]):
                send(entry, signal.SIGKILL)
        end = time.monotonic() + drain_timeout
        while time.monotonic() < end:
            pump()
            live = discover()
            for entry in live:
                send(entry, signal.SIGKILL)
            child.poll()  # WNOHANG, never an unbounded wait.
            if not live and not selector.get_map() and (not logging_ok or not any(queues.values())):
                break
            time.sleep(_POLL)

    try:
        # Refuse stale outputs without replacing any prior logs or receipt.
        diagnostics.mkdir(parents=True, exist_ok=False)
        created = True
        for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
            try:
                logs[name] = os.open(diagnostics / f"launcher.{name}.log",
                                     os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                tees[name] = _open_tee(stream)
                queues[name] = bytearray()
            except BaseException as exc:
                log_error(exc)
                raise
        if int(os.readlink("/proc/self")) != os.getpid():
            raise RuntimeError("/proc is not mounted for this PID namespace; refusing to spawn")
        probe_fd = _pidfd_open(os.getpid())
        try:
            _pidfd_signal(probe_fd, 0)
        finally:
            os.close(probe_fd)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupt)
        control_read, control_write = os.pipe2(os.O_CLOEXEC)
        os.set_blocking(control_read, False)
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  "--guardian", str(control_write), *command],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True, pass_fds=(control_write,), bufsize=0)
        os.close(control_write)
        control_write = None
        anchor = _proc(child.pid)
        if anchor is None:
            ownership_lost = True
            raise RuntimeError("guardian exited before identity capture")
        owned[child.pid] = dict(anchor, first_seen_seconds=time.monotonic() - start)
        for name, source in (("stdout", child.stdout), ("stderr", child.stderr)):
            sources[name] = source
            os.set_blocking(source.fileno(), False)
            selector.register(source, selectors.EVENT_READ, name)
        selector.register(control_read, selectors.EVENT_READ, "control")
        while receipt["cause"] is None:
            pump(_POLL)
            diagnostic_poll()
            live = discover()
            now = time.monotonic()
            if interrupted is not None:
                receipt["interrupt_signal"] = interrupted
                fail("interrupt")
            if now >= deadline:
                receipt["attempt_deadline_reached"] = True
                fail("attempt_timeout")
            if now - last_progress >= stall_timeout:
                receipt["stall_deadline_reached"] = True
                fail("stall_timeout")
            if receipt["final_returncode"] not in (None, 0):
                fail("child_nonzero")
            if child.poll() is not None and not guardian_empty:
                # Read final control bytes before declaring ownership lost.
                pump()
                if not guardian_empty:
                    fail("supervisor_error")
            if guardian_empty and not live and not selector.get_map() and not any(queues.values()):
                if receipt["final_returncode"] == 0:
                    fail("success")
                else:
                    fail("supervisor_error")
            elif exit_seen is not None and now - exit_seen >= drain_timeout:
                if any(queues.values()):
                    log_error(TimeoutError("tee did not drain after launcher exit"))
                else:
                    fail("lingering_descendants")
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            receipt["interrupt_signal"] = signal.SIGINT
            fail("interrupt")
        else:
            receipt["supervisor_errors"].append(repr(exc))
            fail("supervisor_error")
    finally:
        if child is not None and receipt["cause"] != "success":
            try:
                teardown()
            except BaseException as exc:
                # Logging/diagnostic failure must never suppress best-effort KILL.
                receipt["cleanup_errors"].append(repr(exc))
                ownership_lost = True
                try:
                    discover()
                except BaseException as inner:
                    receipt["cleanup_errors"].append(repr(inner))
                for entry in owned.values():
                    send(entry, signal.SIGKILL)
        try:
            if child is not None:
                child.poll()
                # Once explicitly killed, loss of the anchor is expected. It is
                # safe only if no unknown children could be created after freeze.
                live = []
                for entry in owned.values():
                    current = _proc(entry["pid"])
                    entry["alive_at_end"] = _same(entry, current) and _live(current)
                    if entry["alive_at_end"]:
                        live.append(entry)
                ownership_lost = ownership_lost or not (guardian_empty or anchor_killed_safely)
                receipt["cleanup_verified"] = not live and not ownership_lost
            else:
                receipt["cleanup_verified"] = True
        except BaseException as exc:
            receipt["cleanup_errors"].append(repr(exc))
        if logging_ok and any(queues.values()):
            log_error(TimeoutError("tee bytes remained at bounded drain deadline"))
        if any(key.data != "control" for key in selector.get_map().values()):
            log_error(TimeoutError("launcher pipes did not drain within cleanup bound"))
        for fd in logs.values():
            try:
                os.fsync(fd)
            except BaseException as exc:
                log_error(exc)
        for fd in list(logs.values()) + list(tees.values()) + [control_read, control_write]:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as exc:
                    receipt["logging_errors"].append(repr(exc))
        for source in sources.values():
            try:
                source.close()
            except OSError as exc:
                receipt["logging_errors"].append(repr(exc))
        selector.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if receipt["cause"] == "success" and receipt["logging_errors"]:
            receipt["cause"] = "logging_failure"
        if receipt["cause"] == "success" and not receipt["cleanup_verified"]:
            receipt["cause"] = "supervisor_error"
            receipt["supervisor_errors"].append("owned process cleanup could not be verified")
        receipt["guardian_returncode"] = child.returncode if child is not None else None
        receipt["process_inventory"] = list(owned.values())
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt["last_progress_seconds"] = last_progress - start
        receipt["ownership_lost"] = ownership_lost
        receipt["returncode"] = receipt["final_returncode"]
        receipt["outcome"] = "succeeded" if receipt["cause"] == "success" else "failed"
        if receipt["cause"] == "success":
            receipt["cause"] = None
        if created:
            try:
                receipt["receipt_written"] = True
                _atomic_json(diagnostics / "attempt_receipt.json", receipt)
            except BaseException as exc:
                receipt["receipt_written"] = False
                log_error(exc)
                receipt["outcome"] = "failed"
    return receipt


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--guardian":
        raise SystemExit(_guardian(int(sys.argv[2]), sys.argv[3:]))
    raise SystemExit("private watchdog guardian entrypoint")
