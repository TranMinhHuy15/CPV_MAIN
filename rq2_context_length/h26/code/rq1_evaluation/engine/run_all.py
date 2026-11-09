"""Six frozen checkpoints, evaluation only. System-Python supervisor; old runs untouched."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location('evaluation_process_control', str(ROOT / 'references/top/process_control.py'))
_control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_control)
run_checked, exclusive_job, DeadlineExceeded = _control.run_checked, _control.exclusive_job, _control.DeadlineExceeded


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_package():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text())
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Evaluation package changed: ' + relative)


def import_runs(backup, model, output, expected):
    """Extract only explicitly pinned run artifacts, not arbitrary archive members/code."""
    wanted = {v['archive_prefix'] + '/' + name: (v, name, sha)
              for v in expected if v['model'] == model for name, sha in v['files'].items()}
    found = set()

    def accept(name, stream):
        if name not in wanted:
            return
        if name in found:
            raise ValueError('Duplicate archive member: ' + name)
        found.add(name)
        spec, basename, sha = wanted[name]
        dest = output / 'runs' / model / ('seed_%d' % spec['seed']) / basename
        if dest.exists():
            if digest(dest) != sha:
                raise ValueError('Existing frozen input was altered: ' + str(dest))
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + '.tmp')
        h = hashlib.sha256()
        with temporary.open('wb') as target:
            for block in iter(lambda: stream.read(4 << 20), b''):
                h.update(block)
                target.write(block)
        if h.hexdigest() != sha:
            raise ValueError('Backup does not match audited original artifact: ' + name)
        temporary.replace(dest)

    backup = Path(backup).resolve()
    if backup.is_dir():
        for name in wanted:
            relative = Path(name)
            choices = [backup / relative, backup / relative.relative_to(relative.parts[0])]
            candidates = [p for p in choices if p.is_file()]
            if len(candidates) != 1:
                raise ValueError('Cannot resolve frozen artifact in unpacked backup: ' + name)
            with candidates[0].open('rb') as stream:
                accept(name, stream)
    elif zipfile.is_zipfile(backup):
        with zipfile.ZipFile(backup) as container:
            members = [n for n in container.namelist() if n.endswith('.tar')]
            if len(members) != 1:
                raise ValueError('ZIP must contain exactly one result TAR.')
            with container.open(members[0]) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as archive:
                    for member in archive:
                        if member.isfile() and member.name in wanted:
                            accept(member.name, archive.extractfile(member))
    else:
        with tarfile.open(backup) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name in wanted:
                    accept(member.name, archive.extractfile(member))
    if found != set(wanted):
        raise ValueError('Incomplete backup; missing ' + str(sorted(set(wanted) - found)))


def probe_video(path, n):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_streams', '-show_entries',
               'frame=best_effort_timestamp_time,pkt_duration_time:stream=duration,start_time,nb_frames',
               '-of', 'json', str(path)]
    report = json.loads(subprocess.check_output(command, text=True, timeout=120))
    pts = [float(f['best_effort_timestamp_time']) for f in report['frames']]
    streams = report['streams']
    if len(pts) != n or len(streams) != 1 or not all(math.isfinite(p) for p in pts):
        raise ValueError('PTS/count mismatch: ' + str(path))
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('Non-increasing PTS: ' + str(path))
    duration = float(streams[0]['duration'])
    start = float(streams[0].get('start_time', pts[0]))
    cut = start + duration
    if not math.isfinite(cut) or not pts[-1] < cut or cut - pts[-1] > .15:
        raise ValueError('Ambiguous stream-end boundary: ' + str(path))
    return {'pts': pts, 'cut_s': cut, 'duration_s': duration,
            'cut_minus_last_pts_s': cut - pts[-1]}


def prepare_data(data_root, output, deadline):
    """Test-only hashes/PTS shared by all six runs. Labels never enter blind_index."""
    snapshot = ROOT / 'references' / 'top' / 'metadata_snapshot'
    inventory = rows(ROOT / 'references' / 'top' / 'metadata_snapshot' / 'video_inventory.csv')
    annotations = {r['id']: r for r in rows(ROOT / 'references' / 'top' / 'author_preprocessing' / 'annotations.csv')}
    solution = rows(snapshot / 'solution.csv')
    ids = {r['id'] for r in solution}
    if len(solution) != len(ids) or len(ids) != 1344:
        raise ValueError('Expected 1344 unique test IDs.')
    # Never traverse an old run/output with video links when locating the input dataset.
    paths = {}
    for path in Path(data_root).rglob('*.mp4'):
        if path.stem in ids:
            paths.setdefault(path.stem, []).append(path.resolve())
    metadata = {Path(r['file_name']).stem: r for split in ('test-public', 'test-private')
                for r in rows(snapshot / split / 'positive' / 'metadata.csv')}
    expected = {r['id']: r for r in inventory}
    blind, alignment = [], {}
    (output / 'video_checks').mkdir(exist_ok=True)
    for index, video_id in enumerate(sorted(ids)):
        if time.time() >= deadline:
            raise DeadlineExceeded('Budget exhausted during CPU data preparation.')
        if len(paths.get(video_id, [])) != 1:
            raise ValueError('Video missing/duplicated: ' + video_id)
        path = paths[video_id][0]
        row = expected[video_id]
        n = int(annotations[video_id]['total_frames'])
        check_path = output / 'video_checks' / (video_id + '.json')
        stat = path.stat()
        signature = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                     'sha256_expected': row['sha256'], 'frames_expected': n}
        if check_path.exists():
            check = json.loads(check_path.read_text())
            if check['signature'] != signature:
                # Kaggle mount paths/mtimes may change; re-hash, never trust only the name.
                if stat.st_size != int(row['bytes']) or digest(path) != row['sha256']:
                    raise ValueError('Video bytes changed since previous session: ' + video_id)
                check['signature'] = signature
                save(check_path, check)
        else:
            if stat.st_size != int(row['bytes']) or digest(path) != row['sha256']:
                raise ValueError('Dataset bytes do not match original HF release: ' + video_id)
            clock = probe_video(path, n)
            check = {'signature': signature, 'clock': clock}
            save(check_path, check)
        clock = check['clock']
        blind.append({'id': video_id, 'path': str(path), 'total_frames': n,
                      'pts': clock['pts'], 'video_sha256': row['sha256']})
        if video_id in metadata:
            m = metadata[video_id]
            lead = float(m['time_to_accident'])
            gap = float(m['time_of_event']) - float(m['time_of_alert'])
            if lead not in (.5, 1., 1.5) or not math.isfinite(gap) or gap < 0:
                raise ValueError('Invalid metadata event/alert: ' + video_id)
            event = clock['cut_s'] + lead
            onset = event - gap
            alignment[video_id] = {'event_s': event, 'onset_s': onset,
                'event_last_pts_s': clock['pts'][-1] + lead,
                'onset_last_pts_s': clock['pts'][-1] + lead - gap,
                'onset_at_or_after_cut': onset >= clock['cut_s'],
                'verified': False, 'mapping_status': 'documented_clipping_rule_plus_inferred_stream_boundary'}
        if (index + 1) % 100 == 0:
            print('CPU_DATA %d/1344' % (index + 1), flush=True)
    positives = {r['id'] for r in solution if int(r['target']) == 1}
    if set(alignment) != positives or len(positives) != 672:
        raise ValueError('Positive metadata must cover exactly all 672 positive IDs.')
    save(output / 'blind_index.json', blind)
    save(output / 'blind_index_contract.json', [{k: v for k, v in r.items() if k != 'path'} for r in blind])
    save(output / 'alignment_inferred.json', {'clock': 'clip_pts_seconds', 'videos': alignment,
        'evidence': 'Bundled Nexar card: clips end lead seconds before event; original event-alert gap is shift invariant.',
        'boundary_assumption': 'start_time + video stream duration; last-frame-PTS convention evaluated separately.',
        'paper_equivalent': False, 'metadata_sha256': digest(snapshot / 'solution.csv')})
    save(output / 'data_summary.json', {'test_videos': len(blind), 'positive_videos': len(positives),
        'onset_at_or_after_cut_count': sum(v['onset_at_or_after_cut'] for v in alignment.values()),
        'test_labels_in_model_input': False, 'mapping_verified_against_author_evaluator': False})


def summary(output):
    import statistics
    result = {'complete': True, 'training_changed': False, 'paper_metric_equivalence_confirmed': False,
              'published_context_only': {'method': 'RiskProp, CVPR 2026 Table 1, Nexar',
                  'mAP': .870, 'mAUC': .869, 'partial_mAUC_0.1_as_published': .472, 'mTTA_0.1_s_as_published': .958,
                  'note': 'Normalization, threshold expectation, miss policy and clipped-tail treatment unresolved. No published TOP Nexar target.'},
              'models': {}}
    for model in ('riskprop', 'top'):
        reports = [json.loads((output / model / ('seed_%d' % s) / 'evaluation.json').read_text()) for s in (42, 43, 44)]
        result['models'][model] = {'seeds': [42, 43, 44], 'mean_sample_sd': {}}
        for key in reports[0]['summary_metrics']:
            values = [r['summary_metrics'][key] for r in reports]
            result['models'][model]['mean_sample_sd'][key] = (
                {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'n': 3}
                if all(v is not None and math.isfinite(v) for v in values) else {'mean': None, 'sample_sd': None, 'n': 0})
    save(output / 'summary.json', result)
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'metric', 'mean', 'sample_sd', 'n_seeds', 'scope', 'paper_equivalent'])
        for model, report in result['models'].items():
            for key, values in report['mean_sample_sd'].items():
                writer.writerow([model, key, values['mean'], values['sample_sd'], values['n'], 'test_combined_1344', False])
    with (output / 'paper_context.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'mAP', 'mAUC', 'partial_AUC_reported_or_sklearn', 'partial_AUC_raw_divided_0.1',
                         'mTTA_reported_or_observed_TP_seconds', 'mTTA_observed_misses_zero_seconds', 'note'])
        writer.writerow(['RiskProp paper Table 1 (context only)', .870, .869, .472, '', .958, '',
                         'Published definitions unresolved; NOT verified same normalization/evaluator.'])
        for model, report in result['models'].items():
            stats = report['mean_sample_sd']
            def display(key):
                v = stats[key]
                return ('%.6f +/- %.6f' % (v['mean'], v['sample_sd'])) if v['mean'] is not None else 'undefined: no TP'
            writer.writerow([model, display('mAP'), display('mAUC_grouped'), display('mPAUC_sklearn_standardized'),
                             display('mPAUC_raw_divided_by_0.1'), display('mTTA_observed_tp_only_s'),
                             display('mTTA_observed_misses_zero_s'),
                             'Mean/sample SD of 3 frozen seeds; observed clip/inferred clock. Paper metric equivalence NOT verified.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--riskprop-backup', required=True)
    parser.add_argument('--top-backup', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-python')
    parser.add_argument('--hours', type=float, default=10.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours must be finite and positive.')
    output = Path(args.output).resolve()
    inputs = [ROOT, Path(args.data_root).resolve(), Path(args.riskprop_backup).resolve(), Path(args.top_backup).resolve()]
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        parser.error('Output must be separate from code, data, and backup inputs.')
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    with exclusive_job(output / '.job.lock'):
        verify_package()
        expected = json.loads((ROOT / 'expected_jobs.json').read_text())
        fingerprint = {'package_manifest_sha256': digest(ROOT / 'PACKAGE_SHA256.json'),
                       'expected_jobs_sha256': digest(ROOT / 'expected_jobs.json'),
                       'protocol': 'frozen_v6_top_observed_clip_eval_v1'}
        existing = output / 'job_contract.json'
        if existing.exists() and json.loads(existing.read_text()) != fingerprint:
            raise ValueError('Different evaluator/inputs; use a new output folder, never mix curves.')
        save(existing, fingerprint)
        for model, backup in [('riskprop', args.riskprop_backup), ('top', args.top_backup)]:
            print('VERIFY_BACKUP', model, flush=True)
            import_runs(backup, model, output, expected)
        if not shutil.which('ffprobe'):
            raise RuntimeError('ffprobe missing. Install ffmpeg before starting (apt-get install ffmpeg on Vast).')
        prepare_data(args.data_root, output, deadline)
        if args.prepare_only:
            print('CPU_PREPARATION_COMPLETE; no GPU used.', flush=True)
            return
        runtime = args.runtime_python
        if runtime is None:
            setup = ROOT / 'references/top/setup_environment.py'
            run_checked([sys.executable, str(setup), '--launcher', str(ROOT / 'references/top'),
                         '--runtime-root', str(output / 'runtime'), '--world-size', '1'],
                        log_path=output / 'setup.log', deadline=deadline)
            runtime = str(output / 'runtime/venv/bin/python')
        env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        base = [runtime, str(ROOT / 'capture_worker.py'), '--output', str(output), '--deadline', str(deadline)]
        # Preflight all six checkpoints, not only the first seed/model. No long capture before all pass.
        for stage in ('preflight', 'capture', 'report'):
            for model in ('riskprop', 'top'):
                for seed in (42, 43, 44):
                    print('STAGE', stage, model, seed, flush=True)
                    run_checked(base + ['--model', model, '--seed', str(seed), '--stage', stage],
                                log_path=output / 'logs' / ('%s_%d_%s.log' % (model, seed, stage)),
                                env=env, deadline=deadline)
        print(json.dumps(summary(output), indent=2), flush=True)
        print('DONE. comparison.csv + summary.json. No training was performed.', flush=True)


if __name__ == '__main__':
    def stop_requested(signum, frame):
        raise KeyboardInterrupt('Stop requested; retain completed output to resume.')
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        main()
    except (KeyboardInterrupt, DeadlineExceeded) as error:
        print('STOPPED; completed curves are saved. Re-run SAME command/output to continue:', error, flush=True)
        sys.exit(75)
