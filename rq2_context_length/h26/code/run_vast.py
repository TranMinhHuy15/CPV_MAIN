"""Linux/Vast launcher; scientific model/loss remain in the locked author source."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from process_control import run_checked, DEADLINE_ENV
from checkpoint_safety import restore_snapshot

HERE = Path(__file__).resolve().parent
from rq2_profile import PROTOCOL, scientific_profile


def validate_request(args):
    if not math.isfinite(args.hours) or args.hours < 1:
        raise ValueError('--hours must be finite and at least 1.')
    data = Path(args.data_root).resolve()
    workspace = Path(args.workspace).resolve()
    runtime = Path(args.runtime_root).resolve()
    if not data.is_dir():
        raise FileNotFoundError('Extract/upload Nexar_HF first; --data-root must contain all MP4 videos.')
    if workspace == data or data in workspace.parents or runtime == data or data in runtime.parents:
        raise ValueError('Workspace/runtime must be outside the dataset folder to avoid duplicate video scanning.')
    videos = list(data.rglob('*.mp4'))
    if len(videos) != 2844 or len({path.stem for path in videos}) != 2844:
        raise ValueError('Expected exactly 2844 distinct Nexar MP4 IDs, found %d. Extract archives before running.' % len(videos))
    existing = workspace / 'run/run_manifest.json'
    recovery = Path(args.resume_root).resolve() if args.resume_root else (workspace/"run" if args.evaluate_only else None)
    if recovery is not None:
        restore_snapshot(recovery, PROTOCOL, args.seed)
    previous = Path(args.resume_root).resolve() if args.resume_root else None
    if args.evaluate_only:
        if not existing.is_file() or not (workspace / 'run/training_status.json').is_file():
            raise RuntimeError('--evaluate-only needs a completed Vast run inside this workspace.')
        status = json.loads((workspace / 'run/training_status.json').read_text())
        if not status.get('complete') or status.get('completed_epochs') != 50:
            raise RuntimeError('Test stays locked until all 50 epochs are complete.')
        previous = workspace / 'run'
    elif existing.exists() and not previous:
        raise RuntimeError('A run already exists. Use --resume-root pointing at its run folder; no automatic fresh restart.')
    if previous:
        manifest = previous / 'run_manifest.json'
        if not manifest.is_file():
            raise RuntimeError('--resume-root must be a saved run directory, not an ipynb or best checkpoint file.')
        contract = json.loads(manifest.read_text())['contract']
        if contract.get('seed') != args.seed:
            raise RuntimeError('Checkpoint seed does not match --seed; refusing mixed runs.')
        if contract.get('protocol') != PROTOCOL or contract.get('world_size') != 1:
            raise RuntimeError('Only this Vast single-GPU protocol can resume. Kaggle/v3/v4/v5/v6 world2 outputs are rejected.')
        status = json.loads((previous / 'training_status.json').read_text())
        if status.get('complete') and not args.evaluate_only:
            raise RuntimeError('Training already completed; use --evaluate-only, not another training run.')
        epoch, best = status['completed_epochs'], status['best_checkpoint']
        if type(epoch) is not int or not 0 < epoch <= 50 or status.get('target_epochs') != 50:
            raise RuntimeError('Invalid saved training schedule.')
        if not isinstance(best, str) or Path(best).name != best or best in ('.', '..'):
            raise RuntimeError('Invalid checkpoint filename.')
        required = ['resume_latest.pth', best, 'rng_epoch%d_rank0.pth' % epoch, 'validation_history.jsonl']
        missing = [name for name in required if not (previous / name).is_file() or not (previous / name).stat().st_size]
        if missing:
            raise RuntimeError('Incomplete resume output: %s' % missing)
        print('RESUME %s: completed %d/50; strict full contract checked after runtime setup.' % (previous, epoch))
    return data, workspace, runtime, previous


def check_machine(workspace):
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Use a Linux x86_64 CUDA-enabled Vast container, not Windows/ARM.')
    if not shutil.which('ffprobe'):
        raise RuntimeError('Install ffmpeg before starting: apt-get update && apt-get install -y ffmpeg')
    for library in ('libGL.so.1', 'libglib-2.0.so.0'):
        try:
            ctypes.CDLL(library)
        except OSError:
            raise RuntimeError('Missing OpenCV system library %s. In the container terminal run: '
                               'apt-get update && apt-get install -y libgl1 libglib2.0-0' % library)
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                             '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or '4080' not in rows[0]:
        raise RuntimeError('This package targets exactly one RTX 4080. Detected: %s' % rows)
    print('GPU / total MiB / free MiB / driver:', rows[0], flush=True)
    workspace.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workspace).free
    if free < 15 * 1024**3:
        raise RuntimeError('Need at least 15 GiB FREE beyond uploaded videos for runtime, pretraining and checkpoints.')
    host = dict(platform=platform.platform(), gpu_query=rows[0], cpu_count=os.cpu_count(),
                workspace_free_bytes=free, target_gpu_full224_not_yet_verified=True)
    (workspace / 'host_preflight.json').write_text(json.dumps(host, indent=2), encoding='utf-8')


def execute(command, log_path, env=None):
    print('Running:', ' '.join(str(arg) for arg in command), flush=True)
    return run_checked(command, log_path=log_path, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--workspace', default='/workspace/rq2_internal_run')
    parser.add_argument('--runtime-root', default='/workspace/rq2_runtime')
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--job-deadline', type=float)
    parser.add_argument('--resume-root')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true', help='After mandatory smoke, train toward 50 total epochs.')
    mode.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--validation-only', action='store_true', help='Complete training/validation but do not unlock held-out test.')
    args = parser.parse_args(argv)
    job_deadline = args.job_deadline if args.job_deadline is not None else time.time()+args.hours*3600
    if not math.isfinite(job_deadline) or job_deadline <= time.time():
        raise ValueError("Total job deadline must be finite and in the future.")
    data, workspace, runtime_root, previous = validate_request(args)
    check_machine(workspace)
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS='2')
    env[DEADLINE_ENV] = str(job_deadline)
    logs = workspace / 'logs'
    execute([sys.executable, str(HERE / 'setup_environment.py'), '--launcher', str(HERE),
             '--runtime-root', str(runtime_root), '--world-size', '1'], logs / 'setup.log', env)
    runtime = json.loads((HERE / 'runtime_bootstrap.json').read_text())
    python = runtime['python_executable']
    settings = dict(workspace=str(workspace), input_root=str(HERE), data_root=str(data),
                    resume_root=str(previous) if previous else None,
                    seed=args.seed, epochs=50, batch_per_gpu=2, effective_batch_size=16, world_size=1,
                    amp=True, workers_per_gpu=2, verify_video_hashes=True,
                    deadline=min(time.time()+args.hours*3600, job_deadline), runtime_contract=runtime['contract'])
    settings_file = workspace / 'settings.json'
    settings_file.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (workspace / 'runtime_environment.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    shutil.copyfile(HERE / 'runtime_pip_freeze.txt', workspace / 'runtime_pip_freeze.txt')
    source = workspace / 'author_source'
    env['PYTHONPATH'] = os.pathsep.join([str(source), str(HERE)])

    def stage(name):
        execute([python, str(HERE / 'repro.py'), name, '--settings', str(settings_file)], logs / (name+'.log'), env)

    stage('prepare')
    execute([python, str(HERE / 'test_repro.py'), '--source', str(source), '--report',
             str(workspace / 'cpu_checks.json')], logs / 'cpu_checks.log', env)
    execute([python, str(HERE / 'test_amp_precision.py'), '--source', str(source)], logs / 'amp_precision.log', env)
    execute([python, str(HERE / 'test_rq2_model.py')], logs / 'rq2_tests.log', env)
    execute([python, str(HERE / 'test_rq2_integration.py'), '--source', str(source)], logs / 'rq2_integration.log', env)
    if args.evaluate_only:
        if args.validation_only:
            print('VALIDATION_ONLY: training already complete; held-out test remains unopened.',flush=True)
            return
        stage('evaluate')
        return
    stage('smoke')
    smoke = json.loads((workspace / 'smoke_report.json').read_text())
    print(json.dumps(smoke, indent=2), flush=True)
    if not smoke.get('passed') or not smoke.get('mini_validation_passed'):
        raise RuntimeError('Smoke/mini-validation did not pass; no training.')
    if not args.train:
        print('PREFLIGHT PASSED. To train, run the same command with --train. No real training epoch started.', flush=True)
        return
    stage('train')
    status = json.loads((workspace / 'run/training_status.json').read_text())
    print(json.dumps(status, indent=2), flush=True)
    if not status['complete']:
        print('PARTIAL RUN saved. Back up ALL outputs, then rerun with --train --resume-root %s' % (workspace / 'run'), flush=True)
        return
    if args.validation_only:
        print('VALIDATION_ONLY_COMPLETE: all50 epochs; no test curves/metrics computed. Later use queue without --validation-only.',flush=True)
        return
    stage('evaluate')
    print('COMPLETED RQ2: 50 epochs + selected-checkpoint validation/full observed-clip test. Outputs: %s' % workspace, flush=True)


if __name__ == '__main__':
    main()
