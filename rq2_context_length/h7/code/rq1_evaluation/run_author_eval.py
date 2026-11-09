"""Evaluate all six frozen checkpoints; unchanged GPU engine, author-convention test report."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
sys.path.insert(0, str(ENGINE))
import run_all as engine


def verify():
    for name, expected in json.loads((ROOT / 'PACKAGE_SHA256.json').read_text()).items():
        if engine.digest(ROOT / name) != expected:
            raise ValueError('Package file changed: ' + name)
    engine.verify_package()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--riskprop-backup')
    parser.add_argument('--top-backup')
    parser.add_argument('--data-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-python')
    parser.add_argument('--hours', type=float, default=10.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours must be finite and positive.')
    if not args.report_only and not all((args.riskprop_backup, args.top_backup, args.data_root)):
        parser.error('Require both backups and data-root, except in --report-only mode.')
    if args.report_only and not args.runtime_python:
        parser.error('--report-only requires --runtime-python with NumPy1.24.4/scikit-learn1.3.2; no GPU setup needed.')
    output = Path(args.output).resolve()
    inputs = [ROOT] + [Path(v).resolve() for v in (args.riskprop_backup, args.top_backup, args.data_root) if v]
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        parser.error('Output must be separate from code/data/backups.')
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    with engine.exclusive_job(output / '.job.lock'):
        verify()
        contract = dict(package_manifest_sha256=engine.digest(ENGINE / 'PACKAGE_SHA256.json'),
                        expected_jobs_sha256=engine.digest(ENGINE / 'expected_jobs.json'),
                        protocol='frozen_v6_top_observed_clip_eval_v1')
        path = output / 'job_contract.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('Output belongs to a different capture engine. Use a separate output.')
        if not path.exists() and any(p.name != '.job.lock' for p in output.iterdir()):
            raise ValueError('Nonempty unowned output folder; use an empty directory or matching capture output.')
        if args.report_only and not path.exists():
            raise ValueError('Report-only requires an existing complete capture output.')
        engine.save(path, contract)
        if not args.report_only:
            jobs = json.loads((ENGINE / 'expected_jobs.json').read_text())
            for model, backup in [('riskprop', args.riskprop_backup), ('top', args.top_backup)]:
                print('VERIFY_BACKUP', model, flush=True)
                engine.import_runs(backup, model, output, jobs)
            if not shutil.which('ffprobe'):
                raise RuntimeError('ffprobe missing. Install ffmpeg before starting.')
            engine.prepare_data(args.data_root, output, deadline)
            if args.prepare_only:
                print('CPU_PREPARATION_COMPLETE. No GPU used.', flush=True)
                return
        runtime = args.runtime_python
        if runtime is None:
            engine.run_checked([sys.executable, str(ENGINE / 'references/top/setup_environment.py'),
                '--launcher', str(ENGINE / 'references/top'), '--runtime-root', str(output / 'runtime'),
                '--world-size', '1'], log_path=output / 'setup.log', deadline=deadline)
            runtime = str(output / 'runtime/venv/bin/python')
        runtime = str(Path(runtime).resolve())
        env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        if not args.report_only:
            base = [runtime, str(ENGINE / 'capture_worker.py'), '--output', str(output), '--deadline', str(deadline)]
            for stage in ('preflight', 'capture'):
                for model in ('riskprop', 'top'):
                    for seed in (42, 43, 44):
                        print('STAGE', stage, model, seed, flush=True)
                        engine.run_checked(base + ['--model', model, '--seed', str(seed), '--stage', stage],
                            log_path=output / 'logs' / ('%s_%d_%s.log' % (model, seed, stage)), env=env, deadline=deadline)
                if args.preflight_only:
                    print('ALL_SIX_PREFLIGHTS_PASSED. No full capture performed.', flush=True)
                    return
        engine.run_checked([runtime, str(ROOT / 'author_report.py'), '--output', str(output)],
                           log_path=output / 'logs/author_report.log', env=env, deadline=deadline)


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt('Stop requested; keep output to resume.')
    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except (KeyboardInterrupt, engine.DeadlineExceeded) as error:
        print('STOPPED safely; re-run same command/output to resume:', error, flush=True)
        sys.exit(75)
