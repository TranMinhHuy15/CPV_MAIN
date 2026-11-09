"""Sequential RTX4080 RQ2 queue. Default: technical preflight; never paid training implicitly."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from checkpoint_safety import restore_snapshot
from process_control import run_checked, exclusive_job, DeadlineExceeded, GROUP_ENV, DEADLINE_ENV
from rq2_profile import VARIANTS, protocol, scientific_profile, METRIC_SCHEMA, validate_metrics

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def package_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_package():
    for name, expected in load(HERE / 'PACKAGE_SHA256.json').items():
        if package_digest(HERE / name) != expected:
            raise ValueError('Package changed: ' + name)


def validate_arguments(args):
    if args.total_hours is None or not math.isfinite(args.total_hours) or args.total_hours <= 0:
        raise ValueError('Specify a finite positive --total-hours, including setup/smoke/train/test.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate seeds/variants are not allowed.')
    if not args.seeds or any(s not in SEEDS for s in args.seeds):
        raise ValueError('Allowed seeds:42,43,44.')
    if not args.variants or any(v not in VARIANTS for v in args.variants):
        raise ValueError('This ZIP runs only its medium arm.')


def state(root, variant, seed):
    run = root / variant / ('seed_%d' % seed) / 'run'
    manifest, status = run/'run_manifest.json', run/'training_status.json'
    if not manifest.exists() and not status.exists():
        return run, None
    if not manifest.exists() or not status.exists():
        raise ValueError('Incomplete metadata, do not overwrite: ' + str(run))
    contract, value = load(manifest)['contract'], load(status)
    if (contract.get('protocol') != protocol(variant) or contract.get('seed') != seed
            or contract.get('rq2') != scientific_profile(variant)):
        raise ValueError('Wrong protocol/variant/seed: ' + str(run))
    hashes = contract.get('launcher_sha256',{})
    if not hashes or any(Path(name).name != name or package_digest(HERE/name) != value for name,value in hashes.items()):
        raise ValueError('Saved run launcher differs from this package; use the exact original package.')
    if contract.get('rq1_evaluator_sha256') != package_digest(HERE/'rq1_evaluation/PACKAGE_SHA256.json'):
        raise ValueError('Saved run evaluator differs.')
    epoch = value.get('completed_epochs')
    if type(epoch) is not int or not 0 < epoch <= 50 or value.get('target_epochs') != 50 or value.get('complete') != (epoch == 50):
        raise ValueError('Invalid completed-epoch state.')
    best = value.get('best_checkpoint')
    if not isinstance(best, str) or Path(best).name != best or '\\' in best:
        raise ValueError('Unsafe checkpoint path.')
    return run, value


EVALUATION_FILES = ('rq2_test_report.json','rq2_test_metrics.csv','test_predictions.json',
                    'test_capture/capture_manifest.json','test_capture/capture_contract.json')


def seal_evaluation(run, checkpoint_sha):
    """The LAST atomic write commits the entire evaluation, never just its first JSON."""
    report=load(run/'rq2_test_report.json')
    validate_metrics(report['summary_metrics'])
    with (run/'rq2_test_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if (len(rows)!=len(report['summary_metrics']) or
            {row['metric']:float(row['value']) for row in rows}!=report['summary_metrics']):
        raise ValueError('CSV is incomplete or disagrees with the validated JSON metrics.')
    save(run/'evaluation_complete.json',dict(complete=True,test_count=1344,metric_schema=METRIC_SCHEMA,
        checkpoint_sha256=checkpoint_sha,files={name:package_digest(run/name) for name in EVALUATION_FILES}))


def evaluated(run, status):
    path = run/'rq2_test_report.json'
    done_path=run/'evaluation_complete.json'
    if not status or not status['complete'] or not done_path.is_file():
        return False
    done=load(done_path)
    if (done.get('complete') is not True or done.get('test_count')!=1344
            or done.get('metric_schema')!=METRIC_SCHEMA or set(done.get('files',{}))!=set(EVALUATION_FILES)):
        return False
    for name in EVALUATION_FILES:
        if not (run/name).is_file() or package_digest(run/name)!=done['files'][name]:
            return False  # Interrupted finalization can be regenerated from unchanged curves.
    report = load(path)
    if (report.get('complete') is not True or report.get('test_count') != 1344
            or report.get('selected_checkpoint') != status['best_checkpoint']):
        return False
    validate_metrics(report['summary_metrics'])
    if report.get('checkpoint_sha256') != package_digest(run/status['best_checkpoint']):
        raise ValueError('Evaluated checkpoint changed.')
    if report.get('capture_fingerprint',{}).get('training_contract') != load(run/'run_manifest.json')['contract']:
        raise ValueError('Test report belongs to another training contract.')
    if done.get('checkpoint_sha256')!=report['checkpoint_sha256']:
        raise ValueError('Completion marker belongs to another checkpoint.')
    capture=load(run/'test_capture/capture_manifest.json')
    if (capture.get('complete') is not True or capture.get('test_count')!=1344
            or capture.get('checkpoint_sha256')!=report['checkpoint_sha256']):
        return False
    with (HERE/'metadata_snapshot/solution.csv').open(encoding='utf-8-sig',newline='') as stream:
        expected={row['id'] for row in csv.DictReader(stream)}
    hashes=report.get('curve_file_sha256',{})
    if set(hashes)!=expected or len(expected)!=1344:
        return False
    for video_id,expected_sha in hashes.items():
        curve=run/'test_capture/curves'/(video_id+'.json')
        if not curve.is_file():
            return False
        if package_digest(curve)!=expected_sha:
            raise ValueError('Saved risk curve changed: '+video_id+'; refusing silent reuse.')
    return True


def command_for(args, run, status, seed, deadline):
    if status and status['complete'] and args.validation_only:
        return None
    command = [sys.executable,str(HERE/'run_vast.py'),'--data-root',str(Path(args.data_root).resolve()),
        '--workspace',str(run.parent),'--runtime-root',str(Path(args.runtime_root).resolve()),'--seed',str(seed),
        '--hours',str(max(1.,args.total_hours)),'--job-deadline',str(deadline)]
    if args.train:
        command += ['--evaluate-only'] if status and status['complete'] else ['--train']
    if args.validation_only:
        command += ['--validation-only']
    if status and not status['complete']:
        command += ['--resume-root',str(run)]
    return command


def summarize(root):
    jobs, aggregate, common_contracts = [], {}, []
    for variant in VARIANTS:
        arm, contracts = [], []
        for seed in SEEDS:
            run, status = state(root, variant, seed)
            row = dict(variant=variant, seed=seed, state='not_started', completed_epochs=0)
            if status:
                row.update(completed_epochs=status['completed_epochs'], selected_checkpoint=status['best_checkpoint'],
                           state='awaiting_test' if status['complete'] else 'partial')
            if evaluated(run, status):
                row.update(state='complete', metrics=load(run/'rq2_test_report.json')['summary_metrics'])
                arm.append(row['metrics'])
                contract = load(run/'run_manifest.json')['contract']
                contracts.append({k:v for k,v in contract.items() if k!='seed'})
                common_contracts.append({k:v for k,v in contract.items() if k not in ('seed','protocol','rq2')})
            jobs.append(row)
        if len(arm) == 3:
            if any(c != contracts[0] for c in contracts[1:]):
                raise ValueError('Seed contracts differ; refusing mixed mean/SD.')
            keys = set(arm[0])
            if any(set(v) != keys for v in arm):
                raise ValueError('Metric schemas differ across seeds.')
            aggregate[variant] = {}
            for key in sorted(keys):
                values = [v[key] for v in arm]
                if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                    raise ValueError('Invalid metric: ' + key)
                aggregate[variant][key] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n=3)
    if common_contracts and any(c!=common_contracts[0] for c in common_contracts[1:]):
        raise ValueError('Arms differ beyond the declared temporal profile; refusing an unmatched comparison.')
    result = dict(complete_study=len(aggregate) == len(VARIANTS), expected_runs=3*len(VARIANTS), training_complete_runs=sum(j['completed_epochs']==50 for j in jobs), evaluation_complete_runs=sum(j['state']=='complete' for j in jobs), jobs=jobs, mean_sample_sd=aggregate,
                  paper_equivalent=False, note='Each arm aggregates only after all three fixed seeds complete. No best-seed selection.')
    save(root/'summary.json', result)
    with (root/'mean_sd.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant','metric','mean','sample_sd','n'])
        for variant, metrics in aggregate.items():
            for key, values in metrics.items():
                writer.writerow([variant,key,values['mean'],values['sample_sd'],3])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--output-root', default='/workspace/rq2_medium_h13')
    parser.add_argument('--runtime-root', default='/workspace/rq2_runtime_medium_h13')
    parser.add_argument('--total-hours', type=float, help='Required explicit total cap. Does NOT stop Vast billing.')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validation-only', action='store_true',help='Pilot: finish50epochs but leave test unopened.')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    if args.summary_only:
        print(json.dumps(summarize(root), indent=2))
        return
    validate_arguments(args)
    if not args.data_root or not Path(args.data_root).is_dir():
        parser.error('Require extracted Nexar_HF --data-root.')
    data, runtime = Path(args.data_root).resolve(), Path(args.runtime_root).resolve()
    for a, b in ((root,data),(root,HERE),(runtime,data),(runtime,HERE),(runtime,root)):
        if a == b or a in b.parents or b in a.parents:
            parser.error('Code/data/output/runtime must be separate directories, not nested.')
    deadline = time.time() + args.total_hours*3600
    with exclusive_job(root/'job.lock'):
        verify_package()
        job_env = dict(os.environ)
        for key in (GROUP_ENV, DEADLINE_ENV):
            job_env.pop(key, None)
        print('RQ2 Medium H13: sequential selected seeds. Train=', args.train, '; billing is NOT stopped automatically.', flush=True)
        try:
            run_checked([sys.executable, str(HERE/'test_rq2_queue.py')], env=job_env, deadline=deadline)
            run_checked([sys.executable, str(HERE/'test_rq2_completion.py')], env=job_env, deadline=deadline)
            run_checked([sys.executable, str(HERE/'test_safety.py')], env=job_env, deadline=deadline)
            run_checked([sys.executable, str(HERE/'test_medium_ops.py')], env=job_env, deadline=deadline)
            for seed in args.seeds:
                for variant in args.variants:
                    restore_snapshot(root/variant/('seed_%d'%seed)/'run', protocol(variant), seed)
                    run, status = state(root,variant,seed)
                    if evaluated(run,status):
                        print('SKIP complete', variant,seed,flush=True)
                        continue
                    if status and status['complete'] and args.validation_only:
                        print('SKIP training complete; test remains unopened.',variant,seed,flush=True)
                        continue
                    if status and status['complete'] and not args.train:
                        print('Awaiting final test; use --train to unlock evaluation.',flush=True)
                        continue
                    command = command_for(args,run,status,seed,deadline)
                    env = dict(job_env,RQ2_VARIANT=variant)
                    print('START',variant,seed,'original Kinetics pretraining' if status is None else 'strict resume',flush=True)
                    run_checked(command,env=env,deadline=deadline)
                    summarize(root)
                    finished = state(root,variant,seed)[1]
                    if args.train and not (finished and finished['complete'] and (args.validation_only or evaluated(run,finished))):
                        print('PARTIAL: later jobs blocked. Back up all outputs and rerun the SAME command.',flush=True)
                        return 75
        except (KeyboardInterrupt,DeadlineExceeded,subprocess.CalledProcessError) as error:
            save(root/'job_stop.json',dict(reason=type(error).__name__,detail=str(error),deadline=deadline,
                billing_still_active=True,action='Back up outputs; rerun same command. Stop rented instance yourself.'))
            raise
        print('OUTPUT',root,'; three-seed study complete=',summarize(root)['complete_study'],flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
