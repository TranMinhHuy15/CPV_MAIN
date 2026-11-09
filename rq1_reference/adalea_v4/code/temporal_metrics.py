"""Descriptive test mTTA: V6/TOP index10 convention + strict-FAR PTS variant.

Test anchors are INFERRED from clip-end and released positive metadata. They
never enter prediction. Neither metric is asserted equivalent to Table 1.
Misses and videos with no observable post-onset window contribute zero.
"""
import math
from pathlib import Path

import numpy as np

from data import ROOT, read_csv


def make_anchors(curves, solution):
    labels = {r['id']: int(r['target']) for r in solution}
    if len(labels) != len(solution) or {c['id'] for c in curves} != set(labels):
        raise ValueError('Anchor/solution ID mismatch.')
    metadata = {}
    for split in ('test-public', 'test-private'):
        for row in read_csv(ROOT/'metadata_snapshot'/split/'positive/metadata.csv'):
            video_id = Path(row['file_name']).stem
            if video_id in metadata:
                raise ValueError('Duplicate positive metadata.')
            metadata[video_id] = row
    if set(metadata) != {i for i, y in labels.items() if y}:
        raise ValueError('Positive metadata must cover all positive test videos exactly.')
    anchors = {}
    for curve in curves:
        video_id = curve['id']
        if not labels[video_id]:
            continue
        row = metadata[video_id]
        lead = float(row['time_to_accident'])
        gap = float(row['time_of_event']) - float(row['time_of_alert'])
        if lead not in (.5, 1., 1.5) or not math.isfinite(gap) or gap < 0:
            raise ValueError('Invalid released lead/onset metadata.')
        event = curve['cut_s'] + lead
        onset = event - gap
        fps, t0 = curve['native_fps'], curve['first_pts_s']
        first = (curve['total_frames']-1) % 3
        event_frame = math.floor((event-t0)*fps)
        onset_frame = math.floor((onset-t0)*fps)
        i = max(0, (onset_frame-first+2)//3)
        j = (event_frame-first+2)//3
        if j < i:
            raise ValueError('Event index precedes onset.')
        anchors[video_id] = dict(event_s=event, onset_s=onset, onset_index=i,
            event_index=j, mapping_verified=False, source='released metadata + probed stream boundary')
    return labels, anchors


def temporal_metrics(curves, labels, anchors, max_fpr=.1):
    if not 0 < max_fpr < 1:
        raise ValueError('Invalid FAR cutoff.')
    ids = [c['id'] for c in curves]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError('Unique curve IDs must match labels.')
    if any(type(y) is not int or y not in (0, 1) for y in labels.values()):
        raise ValueError('Require integer binary labels.')
    positives = {v for v, y in labels.items() if y}
    if not positives or set(anchors) != positives:
        raise ValueError('Require anchors for all positives and negatives for FAR.')
    arrays = {}
    for curve in curves:
        p, t = np.asarray(curve['scores'], dtype=float), np.asarray(curve['times_s'], dtype=float)
        if (p.ndim != 1 or p.shape != t.shape or not len(p) or not np.isfinite(p).all()
                or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
                or np.any((p < 0) | (p > 1))):
            raise ValueError('Invalid temporal probability/PTS curve.')
        arrays[curve['id']] = p, t
    negatives = [p for v, (p, _) in arrays.items() if not labels[v]]
    if not negatives:
        raise ValueError('No negative observations.')
    negative = np.sort(np.concatenate(negatives))[::-1]
    k = int(len(negative)*max_fpr)
    if k < 1:
        raise ValueError('Too few negatives for low-FAR integration.')
    thresholds = negative[:k]
    # Same rank multiplicity as source. nextafter gives strict handling of ties
    # for the separate physical-time metric (never silently overwrite source).
    strict = np.nextafter(thresholds, np.inf)
    sums_index, sums_pts, tp_counts = np.zeros(k), np.zeros(k), np.zeros(k, dtype=int)
    outside_index, outside_pts = 0, 0
    for video_id in sorted(positives):
        p, times = arrays[video_id]
        a = anchors[video_id]
        i, j = a['onset_index'], a['event_index']
        if type(i) is not int or type(j) is not int or not 0 <= i <= j:
            raise ValueError('Invalid index anchor.')
        event, onset = a['event_s'], a['onset_s']
        if not math.isfinite(event) or not math.isfinite(onset) or event < onset:
            raise ValueError('Invalid physical anchor.')
        window = p[i:j+1]
        valid_pts = (times >= onset) & (times < event)
        outside_index += int(not len(window))
        outside_pts += int(not valid_pts.any())
        pp, tt = p[valid_pts], times[valid_pts]
        for start in range(0, k, 256):
            stop = min(start+256, k)
            if len(window):
                hits = window[:, None] >= thresholds[start:stop]
                detected = hits.any(0)
                sums_index[start:stop] += (j-i-hits.argmax(0))/10. * detected
            if len(pp):
                hits = pp[:, None] >= strict[start:stop]
                detected = hits.any(0)
                sums_pts[start:stop] += (event-tt[hits.argmax(0)])*detected
                tp_counts[start:stop] += detected
    # Compute exact empirical FAR for ties using a sorted ascending search.
    ascending = negative[::-1]
    far_source = (len(negative)-np.searchsorted(ascending, thresholds, side='left'))/len(negative)
    far_strict = (len(negative)-np.searchsorted(ascending, strict, side='left'))/len(negative)
    if np.max(far_strict) > max_fpr+1e-12:
        raise RuntimeError('Strict FAR guarantee failed.')
    n = len(positives)
    return dict(
        mTTA_test_author_index10_inferred_s=float((sums_index/n).mean()),
        mTTA_test_physical_strictFAR_inferred_s=float((sums_pts/n).mean()),
        mTTA_test_physical_TPonly_strictFAR_inferred_s=float(
            np.divide(sums_pts, tp_counts, out=np.zeros(k), where=tp_counts>0).mean()),
        positive_videos=n, negative_observations=len(negative), threshold_ranks_averaged=k,
        positives_without_observed_index_window=outside_index,
        positives_without_observed_physical_onset_window=outside_pts,
        maximum_actual_negative_frame_FAR_source=float(far_source.max()),
        maximum_actual_negative_frame_FAR_physical_strict=float(far_strict.max()),
        misses_contribute_zero=True, prediction_tail_extrapolated=False,
        paper_test_equivalence_confirmed=False, mapping_verified=False,
        index_clock='sample index difference /10, not necessarily physical seconds',
        physical_clock='ffprobe best_effort_timestamp_time, no future score extrapolation',
        threshold_policy='descriptive test-negative-frame ranks, not validation calibration or deployment operating point',
        strict_threshold_policy='nextafter each source rank threshold toward +inf; ties count as a group',
    )


def grouped_temporal(curves, solution):
    labels, anchors = make_anchors(curves, solution)
    lookup = {c['id']: c for c in curves}
    output = {}
    for scope in ('Combined', 'Public', 'Private'):
        rows = [r for r in solution if scope == 'Combined' or r['Usage'] == scope]
        ids = [r['id'] for r in rows]
        output[scope] = temporal_metrics([lookup[v] for v in ids],
            {v: labels[v] for v in ids}, {v: anchors[v] for v in ids if labels[v]})
    return output, anchors
