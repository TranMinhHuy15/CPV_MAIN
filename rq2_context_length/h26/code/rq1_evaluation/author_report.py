"""Author mTTA arithmetic, with an explicitly inferred adapter for clipped Nexar test videos."""
import ast
from collections import OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Dict

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
sys.path.insert(0, str(ENGINE))
import run_all as engine


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def author_prefix():
    """Compile unchanged source statements up to (but excluding) the independent AUC section."""
    import numpy as np
    path = ROOT / 'author_reference_metrics.py'
    expected = load(ENGINE / 'references/riskprop/source_hashes.json')['taa/metrics.py']
    if engine.digest(path) != expected:
        raise ValueError('Original author metrics source changed.')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AnticipationMetric')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate')
    cut = next(k for k, n in enumerate(method.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'preds_0' for t in n.targets))
    method.body = method.body[:cut] + [ast.Return(value=ast.Name(id='eval_results', ctx=ast.Load()))]
    ast.fix_missing_locations(method)
    scope = dict(np=np, OrderedDict=OrderedDict, Dict=Dict)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope['calculate']


def index_anchor(record, anchor, fps):
    """Same frame-floor/first-grid-endpoint rule as source, extrapolated as indices, NOT predictions."""
    n = record['total_frames']
    if type(n) is not int or n < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid native video clock.')
    t0 = record['pts'][0]
    event, onset = anchor['event_s'], anchor['onset_s']
    if not all(math.isfinite(v) for v in (t0, event, onset)) or event < onset:
        raise ValueError('Invalid inferred event/onset.')
    first = (n - 1) % 3
    event_frame, onset_frame = math.floor((event - t0) * fps), math.floor((onset - t0) * fps)
    i, j = max(0, (onset_frame - first + 2) // 3), (event_frame - first + 2) // 3
    if j < i or j < 0:
        raise ValueError('Event precedes observable onset/grid.')
    return dict(onset_index=i, event_index=j, event_frame=event_frame, onset_frame=onset_frame,
                native_fps=fps, source_clock_hz=10, mapping_verified=False)


def temporal(curves, labels, anchors):
    import numpy as np
    ids = [c['id'] for c in curves]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError('Curves must match unique test IDs exactly.')
    if any(type(v) is not int or v not in (0, 1) for v in labels.values()):
        raise ValueError('Labels must be binary integers.')
    positive_ids = {v for v, y in labels.items() if y}
    if set(anchors) != positive_ids or not positive_ids:
        raise ValueError('Anchor IDs must match all positives.')
    arrays, ys, starts, events = [], [], [], []
    total_positive, valid_positive = len(positive_ids), 0
    for c in curves:
        p = np.asarray(c['scores'], dtype=float)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probability curve.')
        y = labels[c['id']]
        i, j = (anchors[c['id']]['onset_index'], anchors[c['id']]['event_index']) if y else (0, 0)
        if type(i) is not int or type(j) is not int or not 0 <= i <= j:
            raise ValueError('Invalid anchor indices.')
        # Source argmax cannot handle an empty observed window. Such positives remain
        # zero in the ALL-positive denominator; no artificial future risk is appended.
        if y and i >= len(p):
            continue
        valid_positive += int(y)
        arrays.append(p)
        ys.append(y)
        starts.append(i)
        events.append(j)
    negatives = [p for p, y in zip(arrays, ys) if not y]
    if not negatives:
        raise ValueError('Require negative curves for empirical source thresholds.')
    neg = np.concatenate(negatives)
    k = int(len(neg) * .1)
    if k < 1:
        raise ValueError('Too few negative observations.')
    if valid_positive:
        values = author_prefix()(SimpleNamespace(fpr_max=.1), arrays, ys, starts, events, '@')
        for key in values:
            if key.startswith(('tta@', 'mtta@')):
                values[key] = float(values[key]) * valid_positive / total_positive
    else:
        values = {'threshold@0.10': float(np.sort(neg)[::-1][k])}
        values.update({key: 0. for key in ('tta@0.01', 'tta@0.05', 'tta@0.1', 'mtta@0.1', 'tta@1')})
    values = {key: float(value) for key, value in values.items()}
    threshold = np.sort(neg)[::-1][k - 1]
    # Keep source rank multiplicity and >= ties, even when actual FAR exceeds .1.
    return dict(values=values, positive_videos=total_positive,
                positives_without_observed_onset_window=total_positive - valid_positive,
                negative_observations=len(neg), threshold_ranks_averaged=k,
                maximum_actual_negative_frame_FAR=float(np.count_nonzero(neg >= threshold) / len(neg)),
                threshold_selection='test-negative-frame empirical ranks, unchanged source descriptive evaluation; not deployment calibration',
                misses_contribute_zero=True, prediction_tail_extrapolated=False,
                clock='source snippet-index difference / 10, NOT guaranteed physical seconds',
                paper_test_equivalence_confirmed=False)


def dataset_contract(output):
    """Reconstruct anchors from pinned metadata plus independently checked stream boundaries."""
    snapshot = ENGINE / 'references/top/metadata_snapshot'
    solution = engine.rows(snapshot / 'solution.csv')
    labels = {r['id']: int(r['target']) for r in solution}
    if len(solution) != 1344 or len(labels) != 1344 or sum(labels.values()) != 672:
        raise ValueError('Expected exact combined 1344 test videos / 672 positives.')
    records = load(output / 'blind_index_contract.json')
    if len(records) != 1344 or {r['id'] for r in records} != set(labels):
        raise ValueError('Blind contract does not match complete test IDs.')
    annotations = {r['id']: r for r in engine.rows(ENGINE / 'references/top/author_preprocessing/annotations.csv')}
    inventory = {r['id']: r for r in engine.rows(snapshot / 'video_inventory.csv')}
    metadata = {Path(r['file_name']).stem: r for split in ('test-public', 'test-private')
                for r in engine.rows(snapshot / split / 'positive/metadata.csv')}
    if set(metadata) != {v for v, y in labels.items() if y}:
        raise ValueError('Positive metadata coverage mismatch.')
    anchors = {}
    for r in records:
        video_id, n, pts = r['id'], r['total_frames'], r['pts']
        if set(r) != {'id', 'total_frames', 'pts', 'video_sha256'}:
            raise ValueError('Unexpected blind record fields (possible label leakage).')
        if (n != int(annotations[video_id]['total_frames']) or r['video_sha256'] != inventory[video_id]['sha256']
                or len(pts) != n or not all(math.isfinite(t) for t in pts)
                or any(b <= a for a, b in zip(pts, pts[1:]))):
            raise ValueError('Video contract/PTS mismatch: ' + video_id)
        if labels[video_id]:
            check = load(output / 'video_checks' / (video_id + '.json'))
            if (check['signature']['sha256_expected'] != r['video_sha256']
                    or check['clock']['pts'] != pts):
                raise ValueError('Independent stream clock does not match: ' + video_id)
            cut = check['clock']['cut_s']
            if not pts[-1] < cut <= pts[-1] + .15:
                raise ValueError('Invalid stream boundary.')
            m = metadata[video_id]
            lead = float(m['time_to_accident'])
            gap = float(m['time_of_event']) - float(m['time_of_alert'])
            if lead not in (.5, 1., 1.5) or not math.isfinite(gap) or gap < 0:
                raise ValueError('Invalid test event metadata.')
            event = cut + lead
            anchor = dict(event_s=event, onset_s=event - gap)
            anchors[video_id] = dict(index_anchor(r, anchor, float(annotations[video_id]['fps'])),
                onset_after_last_observed_pts=anchor['onset_s'] > pts[-1], **anchor)
    return records, solution, labels, anchors


def validated_curves(output, job, records):
    import numpy as np
    frozen = output / 'runs' / job['model'] / ('seed_%d' % job['seed'])
    for name, expected in job['files'].items():
        if engine.digest(frozen / name) != expected:
            raise ValueError('Frozen run artifact changed: ' + str(frozen / name))
    status = load(frozen / 'training_status.json')
    if status['complete'] is not True or status['completed_epochs'] != 50:
        raise ValueError('Training was not completed.')
    checkpoint_sha = job['files'][status['best_checkpoint']]
    source = load(frozen / 'run_manifest.json')['contract']['source_commit']
    prior = load(frozen / 'test_predictions.json')
    old_scores = {r['id']: r['score'] for r in prior}
    directory = output / job['model'] / ('seed_%d' % job['seed'])
    contract = load(directory / 'capture_contract.json')
    expected = dict(checkpoint_sha256=checkpoint_sha, source_commit=source,
        evaluator_manifest_sha256=engine.digest(ENGINE / 'PACKAGE_SHA256.json'),
        capture_worker_sha256=engine.digest(ENGINE / 'capture_worker.py'),
        blind_index_sha256=engine.digest(output / 'blind_index_contract.json'),
        sampling='native_end_anchored_stride3_5past_frames', views=10,
        torch='2.1.0+cu118', cuda='11.8', precision='FP32 tensors, no autocast')
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('Capture provenance changed: ' + str(directory))
    preflight = load(directory / 'preflight.json')
    if preflight.get('passed') is not True or preflight.get('videos') != 8 or preflight.get('contract') != contract:
        raise ValueError('Require matching eight-video preflight for every checkpoint.')
    manifest = load(directory / 'capture_manifest.json')
    if (manifest.get('complete') is not True or manifest.get('test_count') != 1344
            or manifest.get('checkpoint_sha256') != checkpoint_sha or manifest.get('training_changed') is not False):
        raise ValueError('Incomplete capture; no final metrics.')
    ids = {r['id'] for r in records}
    if len(prior) != 1344 or set(old_scores) != ids or {p.stem for p in (directory / 'curves').glob('*.json')} != ids:
        raise ValueError('Missing/extra/duplicate test IDs.')
    curves, max_delta = [], 0.
    for r in records:
        c = load(directory / 'curves' / (r['id'] + '.json'))
        ends = list(range((r['total_frames'] - 1) % 3, r['total_frames'], 3))
        p = np.asarray(c['scores'], dtype=float)
        delta = abs(float(p[-1]) - old_scores[r['id']]) if len(p) else float('inf')
        if (c['id'] != r['id'] or c['frame_inds'] != ends or c['times_s'] != [r['pts'][i] for i in ends]
                or p.shape != (len(ends),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                or c['checkpoint_sha256'] != checkpoint_sha or c['video_sha256'] != r['video_sha256']
                or c['uses_future'] is not False or c['prediction_tail_extrapolated'] is not False
                or c['coverage'] != 'full_observed_clip' or delta > 1e-4):
            raise ValueError('Corrupt/noncausal/stale curve: ' + r['id'])
        max_delta = max(max_delta, delta)
        curves.append(c)
    return curves, prior, max_delta


def report(output):
    import numpy as np
    import sklearn
    from metrics import scalar_metrics
    if np.__version__ != '1.24.4' or sklearn.__version__ != '1.3.2':
        raise RuntimeError('Use the frozen NumPy1.24.4 / scikit-learn1.3.2 runtime for reproducible metric arithmetic.')
    output = Path(output).resolve()
    engine.verify_package()
    records, solution, labels, anchors = dataset_contract(output)
    reports = []
    for job in load(ENGINE / 'expected_jobs.json'):
        curves, prior, delta = validated_curves(output, job, records)
        scalar = scalar_metrics(solution, prior)
        replay = scalar_metrics(solution, [{'id': c['id'], 'score': c['scores'][-1]} for c in curves])
        for key, value in scalar['Combined'].items():
            if isinstance(value, (int, float)) and abs(value - replay['Combined'][key]) > .001:
                raise ValueError('Scalar metric replay drift: ' + key)
        tta = temporal(curves, labels, anchors)
        reports.append(dict(model=job['model'], seed=job['seed'], scope='combined_1344_test_videos',
            scalar_metrics=scalar, temporal_metrics=tta, max_scalar_replay_delta=delta,
            capture_contract=load(output / job['model'] / ('seed_%d' % job['seed']) / 'capture_contract.json')))
        print('REPORT', job['model'], job['seed'], 'test author-index10 inferred mTTA', tta['values']['mtta@0.1'], flush=True)
    means = {}
    hardware_keys = ('GPU', 'torch', 'cuda', 'precision', 'tf32_cudnn', 'sampling', 'views')
    hardware = {k: reports[0]['capture_contract'].get(k) for k in hardware_keys}
    if any({k: r['capture_contract'].get(k) for k in hardware_keys} != hardware for r in reports):
        raise ValueError('Models/seeds were captured under different GPU/runtime settings.')
    for model in ('riskprop', 'top'):
        entries = [r for r in reports if r['model'] == model]
        measures = {k: [r['scalar_metrics']['Combined'][k] for r in entries]
                    for k in ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1')}
        measures['mTTA_test_author_index10_inferred_s'] = [r['temporal_metrics']['values']['mtta@0.1'] for r in entries]
        means[model] = {key: dict(mean=statistics.mean(v), sample_sd=statistics.stdev(v), n=3)
                        for key, v in measures.items()}
    result = dict(complete=True, training_changed=False, test_videos=1344, seeds=[42, 43, 44],
        paper_test_equivalence_confirmed=False, mapping_verified=False,
        source_mTTA_arithmetic_unchanged=True,
        reporting_runtime=dict(python=sys.version, numpy=np.__version__, scikit_learn=sklearn.__version__),
        original_source_sha256=engine.digest(ROOT / 'author_reference_metrics.py'),
        protocol_sha256=engine.digest(ROOT / 'PROTOCOL.json'),
        report_package_manifest_sha256=engine.digest(ROOT / 'PACKAGE_SHA256.json'),
        physical_onset_after_last_pts_count=sum(a['onset_after_last_observed_pts'] for a in anchors.values()),
        source_index_onset_outside_curve_count=reports[0]['temporal_metrics']['positives_without_observed_onset_window'],
        paper_table1_nexar_context_only=dict(mAP=.870, mAUC=.869, partial_mAUC_0_1=.472, mTTA_0_1_s=.958,
            directly_comparable=False, source='RiskProp CVPR 2026, Table 1; test metric equivalence unresolved'),
        warnings=['NOT an exact reproduction of the unpublished Table-1 test evaluator.',
                  'Onset/event anchors inferred from Nexar metadata and stream boundary.',
                  'Source /10 clock is not physical seconds for non-30-FPS clips.',
                  'Source floor/ceil frame quantization may differ from exact-PTS onset inclusion.',
                  'Source threshold ties retained; actual negative-frame FAR may exceed 0.1.',
                  'Table-1 partial-AUC normalization/evaluator equivalence not established.'],
        mean_sample_sd=means, reports=reports)
    dest = output / 'author_code_test_v1'
    dest.mkdir(exist_ok=True)
    engine.save(dest / 'inferred_index_anchors.json', anchors)
    engine.save(dest / 'summary.json', result)
    temporary = dest / 'comparison.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'metric', 'mean', 'sample_sd', 'n_seeds', 'scope', 'paper_equivalent'])
        for model, measures in means.items():
            for key, values in measures.items():
                writer.writerow([model, key, values['mean'], values['sample_sd'], 3, 'combined_1344_test', False])
    temporary.replace(dest / 'comparison.csv')
    print('DONE:', dest, '; no training; paper equivalence NOT verified.', flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    report(Path(parser.parse_args().output))
