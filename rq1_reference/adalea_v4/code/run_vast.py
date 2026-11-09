"""Stdlib supervisor. Does not connect to or rent a Vast instance."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import math
import os
import platform
import ctypes
import shutil

from process_control import DeadlineExceeded, exclusive_job, run_checked, GROUP_ENV, DEADLINE_ENV

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='AdaLEA V4: 3-seed full training/evaluation, or evaluation of completed V3 checkpoints.')
    parser.add_argument('mode', choices=['setup', 'preflight', 'train', 'all', 'evaluate'])
    parser.add_argument('--data-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-python', help='Existing pinned V6 python. Omit only after setup.')
    parser.add_argument('--runtime-root', default='/workspace/adalea_v4_runtime')
    parser.add_argument('--legacy-output', help='Existing V3 output directory containing seed_42 etc. Read only.')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42,43,44], default=[42,43,44])
    parser.add_argument('--trust-v3-checkpoints', action='store_true', help='Only load your own audited V3 pickle checkpoints.')
    parser.add_argument('--hours', type=float, required=True, help='Total wall-clock cap for this invocation, all seeds combined.')
    parser.add_argument('--ack-approximation', action='store_true', help='Acknowledge PROTOCOL_VI deviations, not exact Table1 replication.')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0 or args.hours > 168:
        parser.error('--hours must be in (0,168].')
    if args.mode in ('train','all','evaluate') and not args.ack_approximation:
        parser.error('Read PROTOCOL_VI.md; full training requires --ack-approximation.')
    if args.mode != 'setup' and (not args.data_root or not Path(args.data_root).is_dir()):
        parser.error('--data-root must be an existing directory containing all Nexar MP4s exactly once.')
    if args.legacy_output and (args.mode != 'evaluate' or not args.trust_v3_checkpoints or not Path(args.legacy_output).is_dir()):
        parser.error('--legacy-output requires evaluate, an existing output directory, and --trust-v3-checkpoints.')
    if args.legacy_output:
        previous = Path(args.legacy_output).resolve()
        proposed = Path(args.output).resolve()
        if previous == proposed or previous in proposed.parents or proposed in previous.parents:
            parser.error('New output must be separate from, not nested in, legacy output.')
        required = [previous/'feature_weights.json'] + [previous/('seed_%d'%s)/name
            for s in args.seeds for name in ('completion.json','resume_latest.pth','test_predictions.json')]
        if any(not path.is_file() for path in required):
            parser.error('Missing legacy completed-seed files; unpack the FULL output and select only completed seeds.')
        for seed in args.seeds:
            finished = json.loads((previous/('seed_%d'%seed)/'completion.json').read_text())
            if (finished.get('seed') != seed or finished.get('complete') is not True
                    or finished.get('completed_epochs') != 50 or finished.get('all_test_ids') != 1344):
                parser.error('Seed%d has not completed50epochs/full1344test; do not include partial seeds.'%seed)
    if len(args.seeds) != len(set(args.seeds)) or (args.mode in ('train','all') and args.seeds != [42,43,44]):
        parser.error('Training requires seeds42,43,44; evaluate may select distinct completed seeds.')
    if args.mode != 'setup' and shutil.which('ffprobe') is None:
        parser.error('ffprobe missing; install ffmpeg before running paid extraction.')
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Production launcher requires Linux x86_64 Vast container.')
    for library in ('libGL.so.1', 'libglib-2.0.so.0'):
        try:
            ctypes.CDLL(library)
        except OSError:
            raise RuntimeError('Missing %s; install libgl1 and libglib2.0-0 before setup, not after a paid long run.' % library)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                         check=True, text=True, capture_output=True, timeout=30)
    rows = [line for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or '4080' not in rows[0] or float(rows[0].rsplit(',', 1)[-1]) < 15*1024:
        raise RuntimeError('Requires exactly one RTX4080 with >=15GiB VRAM. Detected: %s' % rows)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time()+args.hours*3600
    job_env = dict(os.environ)
    for key in (GROUP_ENV, DEADLINE_ENV, 'PYTHONHOME', 'PYTHONPATH', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        job_env.pop(key, None)
    # Verify packaged bytes before any paid work; new edits require repackaging/new contract.
    import hashlib
    hashes = json.loads((ROOT/'PACKAGE_SHA256.json').read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Package checksum changed: %s' % name)
    with exclusive_job(output/'.job.lock'):
        if args.mode == 'setup':
            run_checked([sys.executable, str(ROOT/'setup_environment.py'), '--launcher', str(ROOT),
                         '--runtime-root', args.runtime_root, '--world-size', '1'],
                        log_path=output/'setup.log', deadline=deadline, env=job_env)
        else:
            python = args.runtime_python
            if python is None:
                info = json.loads((ROOT/'runtime_bootstrap.json').read_text())
                python = info['python_executable']
            if not Path(python).is_file():
                raise FileNotFoundError('Runtime python missing: %s' % python)
            run_checked([python, '-u', str(ROOT/'test_process_safety.py')],
                        log_path=output/'process_safety.log', deadline=deadline, env=job_env)
            run_checked([python, '-u', str(ROOT/'test_adalea.py')],
                        log_path=output/'cpu_checks.log', deadline=deadline, env=job_env)
            run_checked([python, '-u', str(ROOT/'test_full_eval.py')],
                        log_path=output/'full_eval_checks.log', deadline=deadline, env=job_env)
            run_checked([python, '-u', str(ROOT/'test_gpu_predictor.py'), '--report', str(output/'gpu_predictor_check.json')],
                        log_path=output/'gpu_predictor_check.log', deadline=deadline, env=job_env)
            phases = ['preflight', 'train'] if args.mode == 'all' else [args.mode]
            for phase in phases:
                command = [python, '-u', str(ROOT/'worker.py'), phase, '--data-root', args.data_root,
                    '--output', str(output), '--deadline', str(deadline), '--seeds'] + [str(s) for s in args.seeds]
                if args.legacy_output:
                    command += ['--legacy-output', args.legacy_output, '--trust-v3-checkpoints']
                run_checked(command, log_path=output/(phase+'.log'), deadline=deadline, env=job_env)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, DeadlineExceeded):
        print('Stopped process tree. Previous completed-epoch checkpoint remains. Vast instance is NOT stopped.', flush=True)
        sys.exit(75)
    except subprocess.CalledProcessError as error:
        print('Worker stopped with exit code %d. Read log; no automatic protocol changes. Vast instance remains billed.' % error.returncode,
              flush=True)
        sys.exit(error.returncode if error.returncode > 0 else 1)
