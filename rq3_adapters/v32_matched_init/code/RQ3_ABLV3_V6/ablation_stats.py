"""Shared paired-bootstrap machinery for ablation V2 (both frozen bases). Analysis only, no GPU.

One index matrix [R, 1344] is drawn ONCE from RNG_SEED, stratified by (lead-time group x label) in the order of
solution.csv. Every role, seed and BOTH bases are evaluated on exactly the same resampled videos, so per-base
contrasts and the cross-base interaction I share indices. Metrics follow the grouped evaluator: mAP = mean AP
(average_precision_score) over groups 0/1/2 (0.5/1.0/1.5 s), AP@1.5s = AP of group 2, mAUC = mean ROC-AUC over groups.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = (42, 43, 44)
METRICS = ('mAP', 'AP@1.5s', 'mAUC')
FAMILY_METRICS = ('mAP', 'AP@1.5s')          # Holm family: main contrasts x these two metrics, per base
RESAMPLES = 5000
RNG_SEED = 20261106
ALPHA = .05
# Main (Holm-corrected) contrasts, fixed before the V2 runs; left - right.
MAIN = (
    ('a7_m1', 'c7_current_repeat', 'inter-snippet past in the adapter (same layers/params)'),
    ('a7_m1', 'r7_last_state', 'attention+age vs last state (differs by 524,289 params)'),
    ('g3_learned', 'u3_uniform', 'learned vs fixed uniform fusion (+4,611 params)'),
    ('u3_uniform', 'a26_m1', 'three windows (uniform) vs longest window only (not compute-matched)'),
    ('g3_learned', 'a26_m1', 'three windows (learned) vs longest window only (not compute-matched)'),
)
ARMS6 = ('c7_current_repeat', 'r7_last_state', 'a7_m1', 'a26_m1', 'u3_uniform', 'g3_learned')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_solution(path):
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['id'] for r in rows]
    if len(ids) != 1344 or len(set(ids)) != 1344:
        raise ValueError('Expected the 1,344-video Nexar test solution.')
    labels = np.array([int(r['target']) for r in rows]); groups = np.array([int(r['group']) for r in rows])
    if set(labels) != {0, 1} or set(groups) != {0, 1, 2}:
        raise ValueError('Label/group values differ from the evaluator contract.')
    return ids, labels, groups


def load_scores(path, ids):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    lookup = {}
    for r in rows:
        if r['id'] in lookup:
            raise ValueError('Duplicate prediction ID %s in %s' % (r['id'], path))
        value = float(r['score'])
        if not np.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError('Score outside [0,1] or nonfinite in %s' % path)
        lookup[r['id']] = value
    if set(lookup) != set(ids):
        raise ValueError('Prediction IDs differ from solution: ' + str(path))
    return np.array([lookup[i] for i in ids])


def shared_indices(labels, groups, resamples=RESAMPLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero((groups == k) & (labels == t)) for k in (0, 1, 2) for t in (0, 1)]
    out = np.empty((resamples, labels.size), dtype=np.int32)
    for b in range(resamples):
        out[b] = np.concatenate([rng.choice(m, size=m.size, replace=True) for m in strata])
    return out


def metrics(labels, groups, scores, index=None):
    if index is not None:
        labels, groups, scores = labels[index], groups[index], scores[index]
    ap = [average_precision_score(labels[groups == k], scores[groups == k]) for k in (0, 1, 2)]
    auc = [roc_auc_score(labels[groups == k], scores[groups == k]) for k in (0, 1, 2)]
    return {'mAP': float(np.mean(ap)), 'AP@1.5s': float(ap[2]), 'mAUC': float(np.mean(auc))}


def bootstrap_counts(indices, size):
    if (indices.ndim!=2 or indices.shape[1]!=size or len(indices)<1
            or not np.issubdtype(indices.dtype,np.integer) or indices.min()<0 or indices.max()>=size):
        raise ValueError('Invalid shared bootstrap indices.')
    return np.array([np.bincount(row,minlength=size) for row in indices],dtype=np.int32)


def weighted_group_draws(labels, scores, counts):
    """Exactly tied-score weighted AP/ROC-AUC, equivalent to repeated sampled videos."""
    order=np.argsort(-scores,kind='stable');s=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    ap=np.empty(len(counts));auc=np.empty(len(counts))
    for start in range(0,len(counts),200):
        weight=counts[start:start+200,order].astype(np.float64)
        tp=np.cumsum(weight*labels[order],axis=1)[:,ends]
        fp=np.cumsum(weight*(1-labels[order]),axis=1)[:,ends]
        if (tp[:,-1]<=0).any() or (fp[:,-1]<=0).any():raise ValueError('Each resampled group needs both classes.')
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)!=0)
        increments=np.diff(tp,axis=1,prepend=0.)
        ap[start:start+len(weight)]=(increments*precision).sum(axis=1)/tp[:,-1]
        dfp=np.diff(fp,axis=1,prepend=0.)
        auc[start:start+len(weight)]=(dfp*(tp+(tp-increments))/2).sum(axis=1)/(tp[:,-1]*fp[:,-1])
    return ap,auc


def role_draws(labels, groups, scores, indices, counts=None):
    point=metrics(labels,groups,scores)
    counts=bootstrap_counts(indices,len(labels)) if counts is None else counts
    if counts.shape!=indices.shape:raise ValueError('Count/index shape mismatch.')
    aps=[];aucs=[]
    for k in (0,1,2):
        mask=groups==k
        ap,auc=weighted_group_draws(labels[mask],scores[mask],counts[:,mask]);aps.append(ap);aucs.append(auc)
    return point,{'mAP':np.mean(aps,axis=0),'AP@1.5s':aps[2],'mAUC':np.mean(aucs,axis=0)}


def summarize(point_deltas, draw_deltas):
    """point_deltas: per-seed point Δ; draw_deltas: per-seed draws [3,R]. Returns pp summary of the 3-seed mean."""
    mean_point = float(np.mean(point_deltas)); mean_draws = np.mean(draw_deltas, axis=0)
    p_two = float(min(1., 2 * min(np.mean(mean_draws <= 0), np.mean(mean_draws >= 0))))
    return dict(delta_pp=100 * mean_point, ci95_low_pp=100 * float(np.percentile(mean_draws, 2.5)),
                ci95_high_pp=100 * float(np.percentile(mean_draws, 97.5)), p_two_sided=p_two,
                seeds_positive=int(sum(d > 0 for d in point_deltas)), seeds_negative=int(sum(d < 0 for d in point_deltas)),
                per_seed_pp=[100 * float(d) for d in point_deltas])


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i]); adjusted = [0.] * len(pvalues); running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[i])); adjusted[i] = running
    return adjusted


def verdict(s, adjusted=None):
    significant = (adjusted if adjusted is not None else s['p_two_sided']) < ALPHA
    if significant and s['ci95_low_pp'] > 0 and s['seeds_positive'] == 3:
        return 'effect present (left > right)'
    if significant and s['ci95_high_pp'] < 0 and s['seeds_negative'] == 3:
        return 'effect reversed (left < right)'
    if s['ci95_low_pp'] > 0 or s['ci95_high_pp'] < 0:
        return 'trend (CI excludes 0 but %s)' % ('Holm-adjusted p >= 0.05' if adjusted is not None and not significant
                                                 else 'seeds disagree')
    return 'no detectable effect (CI includes 0)'


def save_draws(path, ids, indices, points, draws, metadata=None):
    """points/draws keyed by (role, seed). Stored so the cross-base interaction reuses identical resamples."""
    arrays = {'indices_sha256': np.array(sha_bytes(indices.tobytes())), 'ids': np.array(ids)}
    if metadata is not None:arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))
    for (role, seed), d in draws.items():
        for m in METRICS:
            arrays['draw|%s|%d|%s' % (role, seed, m)] = d[m]
            arrays['point|%s|%d|%s' % (role, seed, m)] = np.array(points[(role, seed)][m])
    np.savez_compressed(path, **arrays)


def load_draws(path):
    z = np.load(path, allow_pickle=False); points, draws = {}, {}
    for key in z.files:
        if key.startswith(('draw|', 'point|')):
            kind, role, seed, m = key.split('|')
            target = draws if kind == 'draw' else points
            target.setdefault((role, int(seed)), {})[m] = z[key] if kind == 'draw' else float(z[key])
    return str(z['indices_sha256']), list(z['ids']), points, draws
