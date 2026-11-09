"""Run three independent seeds sequentially; resume partial runs, never cherry-pick."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
import statistics
import subprocess
import sys

from run_vast import PROTOCOL
from process_control import run_checked, exclusive_job, DeadlineExceeded, GROUP_ENV, DEADLINE_ENV
from checkpoint_safety import restore_snapshot

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def saved_state(root, seed):
    run = root / ('seed_%d' % seed) / 'run'
    manifest, status = run / 'run_manifest.json', run / 'training_status.json'
    if not manifest.exists() and not status.exists():
        return run, None, None
    if not manifest.is_file() or not status.is_file():
        raise RuntimeError('Incomplete run metadata at %s. Inspect the failed run; do not overwrite it.' % run)
    contract = read_json(manifest)['contract']
    if contract.get('protocol') != PROTOCOL or contract.get('seed') != seed or contract.get('world_size') != 1:
        raise RuntimeError('Wrong protocol/seed at %s. Kaggle and old single-seed outputs cannot be mixed.' % run)
    value = read_json(status)
    epoch = value.get('completed_epochs')
    if type(epoch) is not int or not 0 < epoch <= 50 or value.get('target_epochs') != 50:
        raise RuntimeError('Invalid epoch metadata at %s' % run)
    if value.get('complete') != (epoch == 50):
        raise RuntimeError('Completion flag contradicts epoch count at %s' % run)
    return run, value, contract


def evaluated(run, status):
    if not status or not status['complete']:
        return False
    required = ['validation_metrics.json', 'official_test_metrics.json', 'evaluation_notes.json']
    if not all((run / name).is_file() for name in required):
        return False
    notes = read_json(run / 'evaluation_notes.json')
    return (notes.get('selected_checkpoint') == status.get('best_checkpoint')
            and notes.get('completed_epochs') == 50 and notes.get('test_was_not_used_for_selection') is True)


def collect_summary(root):
    rows, contracts = [], []
    for seed in SEEDS:
        run, status, contract = saved_state(root, seed)
        row = dict(seed=seed, state='not_started', completed_epochs=0, best_checkpoint=None)
        if status:
            row.update(state='awaiting_evaluation' if status['complete'] else 'partial',
                       completed_epochs=status['completed_epochs'], best_checkpoint=status['best_checkpoint'])
        if evaluated(run, status):
            row['state'] = 'complete'
            val = read_json(run / 'validation_metrics.json')
            for key in ('mAP@', 'mAUC@', 'mAUC@0.1', 'mtta@0.1'):
                row['validation/' + key] = float(val[key])
            test = read_json(run / 'official_test_metrics.json')
            for usage in ('Public', 'Private'):
                for key in ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1'):
                    row['test_%s/%s' % (usage.lower(), key)] = float(test[usage][key])
                for group in test[usage]['groups']:
                    row['test_%s/AP@%.1fs' % (usage.lower(), group['lead_time_s'])] = float(group['AP'])
            if not all(math.isfinite(value) for key, value in row.items() if '/' in key):
                raise RuntimeError('Non-finite evaluated metric for seed %d' % seed)
            contracts.append({key:value for key,value in contract.items() if key != 'seed'})
        rows.append(row)
    if contracts and any(contract != contracts[0] for contract in contracts[1:]):
        raise RuntimeError('Seed runs have different experiment contracts; refusing a mixed mean/SD.')
    complete = sum(row['state'] == 'complete' for row in rows)
    aggregate = {}
    if complete == len(SEEDS):
        metric_names = {key for key in rows[0] if '/' in key}
        if any({key for key in row if '/' in key} != metric_names for row in rows):
            raise RuntimeError('Metric schemas differ between seeds.')
        for key in sorted(metric_names):
            values = [row[key] for row in rows]
            aggregate[key] = dict(n=3, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    report = dict(protocol=PROTOCOL, seeds=list(SEEDS), completed_seeds=complete,
                  final_three_seed_report=complete == 3, per_seed=rows, mean_sample_sd=aggregate,
                  notes=['Same fixed author split; seed varies training RNG, not split.',
                         'Checkpoint per seed selected by validation full mAUC@; no best-seed selection.',
                         'Validation risk-curve proxies and official Public/Private clipped-test metrics remain separate.',
                         'Sample SD uses n-1. Three seeds do not prove paper equivalence or statistical superiority.',
                         'No final aggregate from partial/incomplete seeds; test mTTA is not fabricated.'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    columns = ['seed', 'state', 'completed_epochs', 'best_checkpoint'] + sorted({key for row in rows for key in row if '/' in key})
    with (root / 'per_seed_results.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'mean_sd.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['metric', 'n', 'mean', 'sample_sd'])
        writer.writeheader()
        writer.writerows(dict(metric=key, **value) for key,value in aggregate.items())
    return report


def command_for(args, seed, run, status):
    command = [sys.executable, str(HERE / 'run_vast.py'), '--seed', str(seed),
               '--data-root', str(Path(args.data_root).resolve()),
               '--workspace', str(run.parent), '--runtime-root', str(Path(args.runtime_root).resolve()),
               '--hours', str(args.hours_per_seed), '--job-deadline', str(args.job_deadline)]
    if status:
        if status['complete']:
            command.append('--evaluate-only')
        else:
            command += ['--resume-root', str(run)]
            if args.train:
                command.append('--train')
    elif args.train:
        command.append('--train')
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', default='/workspace/riskprop_v6_vast_3seeds')
    parser.add_argument('--runtime-root', default='/workspace/riskprop_v6_vast_runtime')
    parser.add_argument('--hours-per-seed', type=float, default=24.)
    parser.add_argument('--total-hours', type=float, default=24., help='TOTAL hard wall-clock cap, includes setup/evaluation; does not stop billing.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    args.job_deadline = time.time()+args.total_hours*3600
    if not math.isfinite(args.total_hours) or args.total_hours < 1:
        parser.error("--total-hours must be finite and at least 1.")
    if not math.isfinite(args.hours_per_seed) or args.hours_per_seed < 1:
        parser.error('--hours-per-seed must be finite and at least 1.')
    root = Path(args.output_root).resolve()
    if args.summary_only:
        print(json.dumps(collect_summary(root), indent=2))
        return
    if not Path(args.data_root).is_dir():
        parser.error('Upload/extract Nexar_HF first.')
    with exclusive_job(root/'job.lock'):
        job_env = dict(os.environ)
        job_env.pop(GROUP_ENV,None)
        job_env.pop(DEADLINE_ENV,None)
        for seed in SEEDS:
            restore_snapshot(root/('seed_%d'%seed)/'run', PROTOCOL, seed)
        try:
            # Before installing/training: test real POSIX group cleanup on THIS Linux host.
            if any(not evaluated(root/('seed_%d'%seed)/'run', saved_state(root,seed)[1]) for seed in SEEDS):
                run_checked([sys.executable, str(HERE/'test_safety.py')], env=job_env, deadline=args.job_deadline)
            print('TOTAL job cap %.1fh, including setup/evaluation; NOT an instance billing cap.' % args.total_hours, flush=True)
            print('ONE GPU, SEQUENTIAL seeds 42 -> 43 -> 44. 50 epochs PER seed. '
                  'Budget %.1fh PER seed invocation, not total; instance billing is not stopped automatically.' % args.hours_per_seed, flush=True)
            for seed in SEEDS:
                run, status, _ = saved_state(root, seed)
                if evaluated(run, status):
                    print('Seed %d already fully evaluated; skipping.' % seed, flush=True)
                    continue
                if status and status['complete'] and not args.train:
                    print('Seed %d needs final evaluation; preflight mode will not run test. Use --train.' % seed, flush=True)
                    continue
                print('Seed %d: %s' % (seed, 'fresh from original pretraining' if status is None else
                                      'resume/evaluate existing %d/50' % status['completed_epochs']), flush=True)
                command = command_for(args, seed, run, status)
                # One child at a time; no competing GPU jobs, no new training implementation.
                run_checked(command, env=job_env, deadline=args.job_deadline)
                report = collect_summary(root)
                if args.train and not evaluated(run, saved_state(root, seed)[1]):
                    print('Seed %d stopped partial. Back up outputs and rerun the SAME command to resume. '
                          'Later seeds will not start yet.' % seed, flush=True)
                    return
            report = collect_summary(root)
            print('Final mean +/- sample SD ready.' if report['final_three_seed_report'] else
                  'Preflight completed. Enable --train; no final three-seed metrics yet.', flush=True)
            print('Reports:', root, flush=True)
        except (KeyboardInterrupt, DeadlineExceeded, subprocess.CalledProcessError) as error:
            (root/'job_stop.json').write_text(json.dumps(dict(reason=type(error).__name__,
                detail=str(error), total_hours=args.total_hours, deadline=args.job_deadline,
                instruction='Back up all outputs. Rerun same command to restore last committed epoch; instance billing is still active.'),indent=2),encoding='utf-8')
            raise



if __name__ == '__main__':
    main()
