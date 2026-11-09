"""Explicit inferred-clock observed-clip metrics; never claim Table-1 equivalence."""
import json
from pathlib import Path

import numpy as np


def scalar_metrics(solution, predictions):
    # Exact shared scalar evaluator from the frozen TOP/V6 benchmark.
    import ast
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    path = Path(__file__).resolve().parent / 'references/top/evaluate.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'partial_auc_raw_normalized', 'validate_submission', 'grouped_test_metrics'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 3:
        raise ValueError('Reference evaluator changed.')
    scope = dict(np=np, average_precision_score=average_precision_score, roc_auc_score=roc_auc_score, roc_curve=roc_curve)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['grouped_test_metrics'](solution, predictions)


def temporal_metrics(curves, solution, alignment, last_pts=False):
    lookup = {r['id']: r for r in solution}
    if len(lookup) != len(solution) or {r['id'] for r in curves} != set(lookup):
        raise ValueError('Exactly matching, unique test IDs are required.')
    if len(curves) != len(lookup):
        raise ValueError('Duplicate curve ID.')
    for r in curves:
        p, t = np.asarray(r['scores']), np.asarray(r['times_s'])
        if p.ndim != 1 or t.shape != p.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(t).all():
            raise ValueError('Corrupt curve vectors.')
        if np.any((p < 0) | (p > 1)) or np.any(np.diff(t) <= 0):
            raise ValueError('Invalid risk/timestamp.')
        if r['coverage'] != 'full_observed_clip' or r['uses_future'] is not False:
            raise ValueError('Short-tail/noncausal curves are not accepted.')
    positives = [r for r in curves if int(lookup[r['id']]['target']) == 1]
    negatives = [r for r in curves if int(lookup[r['id']]['target']) == 0]
    if not positives or not negatives:
        raise ValueError('Both classes are required.')
    neg = np.sort(np.concatenate([r['scores'] for r in negatives]))
    rank_count = int(len(neg) * .1)
    if rank_count < 1:
        raise ValueError('Too few negative frames.')
    # Preserve empirical rank multiplicity, but compute each distinct threshold once.
    threshold, weight = np.unique(neg[-rank_count:], return_counts=True)
    far = (len(neg) - np.searchsorted(neg, threshold, side='left')) / len(neg)
    accepted = far <= .1 + 1e-12
    threshold, weight, far = threshold[accepted], weight[accepted], far[accepted]
    if not len(threshold):
        raise ValueError('No valid low-FAR threshold (ties); cannot fabricate a metric.')
    totals, counts = np.zeros(len(threshold)), np.zeros(len(threshold), dtype=int)
    unobservable = 0
    for r in positives:
        anchor = alignment[r['id']]
        event = anchor['event_last_pts_s' if last_pts else 'event_s']
        onset = anchor['onset_last_pts_s' if last_pts else 'onset_s']
        t, p = np.asarray(r['times_s']), np.asarray(r['scores'])
        mask = (t >= onset) & (t <= event)
        t, p = t[mask], p[mask]
        if not len(t):
            unobservable += 1
            continue
        for start in range(0, len(threshold), 256):
            th = threshold[start:start + 256]
            hit_matrix = p[:, None] >= th
            hit = hit_matrix.any(0)
            tta = np.where(hit, event - t[hit_matrix.argmax(0)], 0.)
            totals[start:start + len(th)] += tta
            counts[start:start + len(th)] += hit
    tp = counts > 0
    false_video = np.zeros(len(threshold))
    for r in negatives:
        false_video += np.max(r['scores']) >= threshold
    return {
        'mTTA_observed_tp_only_s': float(np.average(totals[tp] / counts[tp], weights=weight[tp])) if tp.any() else None,
        'mTTA_observed_misses_zero_s': float(np.average(totals / len(positives), weights=weight)),
        'mean_recall_at_admissible_thresholds': float(np.average(counts / len(positives), weights=weight)),
        'mean_negative_video_alarm_rate': float(np.average(false_video / len(negatives), weights=weight)),
        'max_measured_frame_FAR': float(far.max()), 'negative_frames': len(neg),
        'threshold_rank_count': rank_count, 'accepted_rank_count': int(weight.sum()),
        'distinct_accepted_thresholds': len(threshold), 'positive_videos': len(positives),
        'positive_videos_with_no_observed_post_onset_endpoint': unobservable,
        'clock_convention': 'last_frame_pts_plus_lead' if last_pts else 'video_stream_end_plus_lead',
        'threshold_measure': 'empirical_negative_frame_rank_multiplicity; FRAME_FAR <= .1, not video/hour FAR',
        'event_onset_alignment': 'inferred_from_Nexar_metadata',
        'prediction_tail_extrapolated': False, 'paper_metric_equivalence_confirmed': False,
    }


def report(output, model, seed, snapshot, save):
    output = Path(output)
    directory = output / model / ('seed_%d' % seed)
    manifest = json.loads((directory / 'capture_manifest.json').read_text())
    if not manifest['complete'] or manifest['test_count'] != 1344:
        raise ValueError('Full capture must complete before final evaluation.')
    frozen = output / 'runs' / model / ('seed_%d' % seed)
    prior = json.loads((frozen / 'test_predictions.json').read_text())
    from run_all import rows
    solution = rows(Path(snapshot) / 'solution.csv')
    curves = [json.loads(p.read_text()) for p in sorted((directory / 'curves').glob('*.json'))]
    if any(r['checkpoint_sha256'] != manifest['checkpoint_sha256'] for r in curves):
        raise ValueError('Mixed checkpoint curves.')
    old = scalar_metrics(solution, prior)
    captured = scalar_metrics(solution, [{'id': r['id'], 'score': r['scores'][-1]} for r in curves])
    keys = ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1')
    scalar_delta = {k: captured['Combined'][k] - old['Combined'][k] for k in keys}
    if any(abs(delta) > .001 for delta in scalar_delta.values()):
        raise ValueError('Scalar metric drift >0.001. Investigate before reporting mTTA.')
    alignment = json.loads((output / 'alignment_inferred.json').read_text())['videos']
    temporal = temporal_metrics(curves, solution, alignment)
    sensitivity = temporal_metrics(curves, solution, alignment, last_pts=True)
    values = {k: old['Combined'][k] for k in keys}
    values['mPAUC_raw_area'] = old['Combined']['mPAUC_raw_divided_by_0.1'] * .1
    for group in old['Combined']['groups']:
        lead = group['lead_time_s']
        values['AP@%.1fs' % lead] = group['AP']
        values['AUC_full@%.1fs' % lead] = group['AUC']
    values.update({k: temporal[k] for k in ('mTTA_observed_tp_only_s', 'mTTA_observed_misses_zero_s',
                                          'mean_recall_at_admissible_thresholds')})
    values['mTTA_last_pts_boundary_tp_only_s'] = sensitivity['mTTA_observed_tp_only_s']
    result = {'model': model, 'seed': seed, 'training_changed': False,
              'scalar_frozen_original_predictions': old, 'scalar_curve_replay': captured,
              'scalar_metric_replay_delta': scalar_delta, 'temporal_observed_inferred_clock': temporal,
              'one_frame_boundary_sensitivity': sensitivity, 'summary_metrics': values,
              'paper_metric_equivalence_confirmed': False}
    save(directory / 'evaluation.json', result)
    return result
