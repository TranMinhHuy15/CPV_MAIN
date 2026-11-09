"""Pre-registered paired bootstrap for the frozen-V6 study (analysis only, no GPU).

Reads the per-video test_predictions.json written by every test path, the b0 (frozen V6) per-video
risk curves, and the Nexar solution.csv shipped in the source archive. Resamples test videos with
replacement inside each (lead-time group x label) stratum, with the SAME indices for both sides
(paired). Metrics match the grouped evaluator: mAP = mean AP over the 0.5/1.0/1.5 s groups,
AP@1.5s = AP of group 2, mAUC = mean ROC-AUC over groups. Pooled 1,344-video set.

Roles
  b0       frozen RiskProp V6, re-scored in the same test run (the baseline)
  m1, m1_m2, m1_dx   history modules on the frozen V6
  smooth7  zero-parameter control: causal mean of the last 7 points of the b0 risk curve
           (same 7-token history length as M1; fixed in advance, not tuned)

Usage (after `report` has finished):
  python3 bootstrap_paired.py --output-root OUT --solution /path/to/base_h7/metadata_snapshot/solution.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = (42, 43, 44)
MODULES = ('m1', 'm1_m2', 'm1_dx')
CONTROL = 'smooth7'
SMOOTH_POINTS = 7
METRICS = ('AP@1.5s', 'mAP', 'mAUC')
RESAMPLES = 5000
RNG_SEED = 20261105


def load_solution(path):
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['id'] for r in rows]
    if len(ids) != 1344 or len(set(ids)) != 1344:
        raise ValueError('Expected the 1,344-video Nexar test solution.')
    return ids, np.array([int(r['target']) for r in rows]), np.array([int(r['group']) for r in rows])


def load_scores(path, ids):
    lookup = {r['id']: float(r['score']) for r in json.loads(Path(path).read_text(encoding='utf-8'))}
    if set(lookup) != set(ids):
        raise ValueError('Prediction IDs differ from solution: ' + str(path))
    return np.array([lookup[i] for i in ids])


def smoothed_scores(curves_dir, ids, points=SMOOTH_POINTS):
    """Causal: only the last `points` scores up to and including the scored (final) endpoint."""
    values = []
    for i in ids:
        scores = json.loads((Path(curves_dir) / (i + '.json')).read_text(encoding='utf-8'))['scores']
        if not scores:
            raise ValueError('Empty curve: ' + i)
        values.append(float(np.mean(scores[-points:])))
    return np.array(values)


def metrics(labels, groups, scores, index):
    y, g, s = labels[index], groups[index], scores[index]
    ap = [average_precision_score(y[g == k], s[g == k]) for k in (0, 1, 2)]
    auc = [roc_auc_score(y[g == k], s[g == k]) for k in (0, 1, 2)]
    return {'AP@1.5s': ap[2], 'mAP': float(np.mean(ap)), 'mAUC': float(np.mean(auc))}


def strata_indices(labels, groups, rng):
    parts = []
    for k in (0, 1, 2):
        for t in (0, 1):
            members = np.flatnonzero((groups == k) & (labels == t))
            parts.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(parts)


def paired(labels, groups, module, baseline, rng, resamples):
    full = np.arange(labels.size)
    a, c = metrics(labels, groups, module, full), metrics(labels, groups, baseline, full)
    point = {m: a[m] - c[m] for m in METRICS}
    draws = {m: np.empty(resamples) for m in METRICS}
    for b in range(resamples):
        index = strata_indices(labels, groups, rng)
        a, c = metrics(labels, groups, module, index), metrics(labels, groups, baseline, index)
        for m in METRICS:
            draws[m][b] = a[m] - c[m]
    return point, draws


def summary(point, draws):
    return {m: dict(delta_pp=100 * point[m], ci95_low_pp=100 * float(np.percentile(draws[m], 2.5)),
                    ci95_high_pp=100 * float(np.percentile(draws[m], 97.5)),
                    p_delta_le_0=float(np.mean(draws[m] <= 0))) for m in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--solution', required=True)
    parser.add_argument('--resamples', type=int, default=RESAMPLES)
    args = parser.parse_args()
    root = Path(args.output_root)
    if not (root / 'comparison/TEST_COMPARISON_COMPLETE.json').is_file():
        raise RuntimeError('Run only after the sealed test comparison is complete.')
    ids, labels, groups = load_solution(args.solution)
    rng = np.random.default_rng(RNG_SEED)

    def scores(role, seed):
        run = root / 'test/runs' / ('b0' if role == CONTROL else role) / ('seed_%d' % seed) / 'run'
        if role == CONTROL:
            return smoothed_scores(run / 'test_capture/curves', ids)
        return load_scores(run / 'test_predictions.json', ids)

    # (left, right, kind): 'primary' uses the decision rule; 'control' applies the same rule to the
    # zero-parameter smoother; 'descriptive' is reported only.
    comparisons = [(m, 'b0', 'primary') for m in MODULES] + [(CONTROL, 'b0', 'control')]
    comparisons += [(m, CONTROL, 'descriptive') for m in MODULES]
    rows, pooled = [], {}
    for left, right, kind in comparisons:
        per_seed = []
        for seed in SEEDS:
            point, draws = paired(labels, groups, scores(left, seed), scores(right, seed), rng, args.resamples)
            per_seed.append((point, draws))
            for m, s in summary(point, draws).items():
                rows.append(dict(left=left, right=right, kind=kind, seed=seed, metric=m, **s))
            print('done %s vs %s seed%d dAP1.5=%+.2fpp dmAP=%+.2fpp' % (left, right, seed, 100 * point['AP@1.5s'],
                  100 * point['mAP']), flush=True)
        mean_point = {m: float(np.mean([p[m] for p, _ in per_seed])) for m in METRICS}
        mean_draws = {m: np.mean([d[m] for _, d in per_seed], axis=0) for m in METRICS}
        pooled[(left, right, kind)] = summary(mean_point, mean_draws)
        for m, s in pooled[(left, right, kind)].items():
            rows.append(dict(left=left, right=right, kind=kind, seed='mean_3_seeds', metric=m, **s))
    out = root / 'comparison'
    with (out / 'paired_bootstrap.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decisions = {}
    for (left, right, kind), s in pooled.items():
        seeds_positive = sum(r['delta_pp'] > 0 for r in rows if r['left'] == left and r['right'] == right
                             and r['metric'] == 'AP@1.5s' and r['seed'] != 'mean_3_seeds')
        supported = (s['AP@1.5s']['ci95_low_pp'] > 0 and seeds_positive == 3 and s['mAP']['delta_pp'] >= -1.0)
        decisions['%s_vs_%s' % (left, right)] = dict(
            kind=kind, supported_improvement=supported if kind != 'descriptive' else None,
            seeds_with_positive_AP15=seeds_positive,
            mean_delta_AP15_pp=s['AP@1.5s']['delta_pp'], ci95_AP15_pp=[s['AP@1.5s']['ci95_low_pp'], s['AP@1.5s']['ci95_high_pp']],
            mean_delta_mAP_pp=s['mAP']['delta_pp'], ci95_mAP_pp=[s['mAP']['ci95_low_pp'], s['mAP']['ci95_high_pp']],
            mean_delta_mAUC_pp=s['mAUC']['delta_pp'], ci95_mAUC_pp=[s['mAUC']['ci95_low_pp'], s['mAUC']['ci95_high_pp']],
            verdict=('descriptive only' if kind == 'descriptive' else
                     'supported improvement' if supported else 'not supported (trend or no effect)'))
    (out / 'paired_bootstrap_decisions.json').write_text(json.dumps(
        dict(rule='PREREGISTERED_ANALYSIS.md', resamples=args.resamples, rng_seed=RNG_SEED,
             smoothing_points=SMOOTH_POINTS, decisions=decisions), indent=2) + '\n')
    for name, d in decisions.items():
        print('%-22s %-34s dAP1.5=%+.2fpp CI95[%.2f,%.2f] seeds+=%d | dmAP=%+.2fpp CI95[%.2f,%.2f]' % (
            name, d['verdict'], d['mean_delta_AP15_pp'], *d['ci95_AP15_pp'], d['seeds_with_positive_AP15'],
            d['mean_delta_mAP_pp'], *d['ci95_mAP_pp']), flush=True)


if __name__ == '__main__':
    main()
