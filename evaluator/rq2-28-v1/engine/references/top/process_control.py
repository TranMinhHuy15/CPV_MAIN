"""Own a process group, enforce a wall-clock deadline, and clean up descendants."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

GROUP_ENV = 'TOP_SUPERVISED_GROUP'
DEADLINE_ENV = 'TOP_JOB_DEADLINE'


class DeadlineExceeded(RuntimeError):
    pass


def stop_tree(process, owns_group):
    if process.pid <= 1 or process.pid == os.getpid():
        raise RuntimeError('Refusing unsafe process target.')
    if os.name == 'posix' and owns_group:
        if process.pid == os.getpgrp():
            raise RuntimeError('Refusing to signal our own process group.')
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The group may outlive its leader; kill remaining descendants as well.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == 'nt':
        # Local regression only. Production Vast execution is Linux/POSIX.
        if process.poll() is None:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    elif process.poll() is None:
        # Nested worker: the outer supervisor owns/cleans the whole group.
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_checked(command, log_path=None, env=None, deadline=None):
    child_env = dict(os.environ if env is None else env)
    owns_group = os.name == 'posix' and child_env.get(GROUP_ENV) != '1'
    inherited = child_env.get(DEADLINE_ENV)
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError('Explicit deadline must be finite.')
    if inherited:
        inherited_deadline = float(inherited)
        if not math.isfinite(inherited_deadline):
            raise ValueError('Inherited deadline must be finite.')
        deadline = min(deadline, inherited_deadline) if deadline is not None else inherited_deadline
    if deadline is not None and time.time() >= deadline:
        raise DeadlineExceeded('Total job budget exhausted before launching a child.')
    child_env[GROUP_ENV] = '1'
    if deadline is not None:
        child_env[DEADLINE_ENV] = str(deadline)
    log = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True,exist_ok=True)
        log = log_path.open('a',encoding='utf-8')
    options = dict(env=child_env,start_new_session=owns_group)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    if log:
        options.update(stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                       encoding='utf-8',errors='replace',bufsize=1)
    try:
        process = subprocess.Popen(command,**options)
    except BaseException:
        if log:
            log.close()
        raise
    reader, reader_errors = None, []
    if log:
        def forward():
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                    log.flush()
            except Exception as error:
                reader_errors.append(error)
        reader = threading.Thread(target=forward,daemon=True)
        reader.start()
    old_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Stop requested by SIGTERM.')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        while True:
            remaining = deadline-time.time() if deadline is not None else 1.
            if remaining <= 0:
                raise DeadlineExceeded('Total job wall-clock budget reached; stopping process tree.')
            try:
                code = process.wait(timeout=min(1.,remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        # Includes normal exit: no orphan workers left after a launcher failure.
        try:
            stop_tree(process,owns_group)
        finally:
            signal.signal(signal.SIGTERM,old_term)
            if reader:
                reader.join(timeout=5)
            if process.stdout is not None and (reader is None or not reader.is_alive()):
                process.stdout.close()
            if log:
                log.close()
    if code:
        if deadline is not None and time.time() >= deadline:
            raise DeadlineExceeded('Budget reached while child exited.')
        raise subprocess.CalledProcessError(code,command)
    if reader_errors:
        raise RuntimeError('Log forwarding failed: %s'%reader_errors)
    return subprocess.CompletedProcess(command,code)


@contextmanager
def exclusive_job(path):
    """Advisory lock releases automatically on exit/crash; do not unlink locks."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as file:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another job already owns this output directory.')
            try:
                yield
            finally:
                fcntl.flock(file,fcntl.LOCK_UN)
        else:
            import msvcrt
            if path.stat().st_size == 0:
                file.write(b'0')
                file.flush()
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                raise RuntimeError('Another job already owns this output directory.')
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
