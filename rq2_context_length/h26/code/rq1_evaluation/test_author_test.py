"""CPU regression checks; no model training, GPU, or test-metric fitting."""
import ast
from collections import OrderedDict
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict
import unittest
from unittest.mock import patch

import numpy as np
from author_report import author_prefix, index_anchor, temporal, ROOT


def full_calculator():
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    path = ROOT / 'author_reference_metrics.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AnticipationMetric')
    calc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate')
    auc = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'auc')
    scope = dict(np=np, Dict=Dict, OrderedDict=OrderedDict, average_precision_score=average_precision_score,
                 roc_auc_score=roc_auc_score, roc_curve=roc_curve)
    exec(compile(ast.Module(body=[calc, auc], type_ignores=[]), str(path), 'exec'), scope)
    return scope['calculate']


class Checks(unittest.TestCase):
    def test_prefix_is_exact_full_method(self):
        rng = np.random.RandomState(42)
        full, prefix = full_calculator(), author_prefix()
        for trial in range(25):
            p = [rng.rand(30), rng.rand(30), rng.rand(30)]
            if trial % 2:
                p[2] = np.round(p[2], 1)
            result = full(SimpleNamespace(fpr_max=.1), p, [1, 1, 0], [1, 3, 0], [29, 27, 0], '@')
            shorter = prefix(SimpleNamespace(fpr_max=.1), p, [1, 1, 0], [1, 3, 0], [29, 27, 0], '@')
            self.assertEqual(shorter, {k: v for k, v in result.items() if k in shorter})

    def test_independent_oracle_with_virtual_event_and_ties(self):
        rng = np.random.RandomState(43)
        for trial in range(100):
            curves = [dict(id=str(k), scores=np.round(rng.rand(20 + k), 1).tolist()) for k in range(5)]
            labels = {str(k): int(k < 3) for k in range(5)}
            anchors = {str(k): dict(onset_index=int(rng.randint(0, 25)), event_index=35) for k in range(3)}
            result = temporal(curves, labels, anchors)
            neg = sorted(curves[3]['scores'] + curves[4]['scores'], reverse=True)
            thresholds = neg[:int(len(neg) * .1)]
            values = []
            for threshold in thresholds:
                total = 0.
                for c in curves[:3]:
                    a = anchors[c['id']]
                    hit = next((i for i in range(a['onset_index'], min(a['event_index'] + 1, len(c['scores'])))
                                if c['scores'][i] >= threshold), None)
                    total += (a['event_index'] - hit) / 10 if hit is not None else 0
                values.append(total / 3)
            self.assertAlmostEqual(result['values']['mtta@0.1'], sum(values) / len(values), places=12)

    def test_misses_and_unobserved_onset_zero(self):
        curves = [dict(id='hit', scores=[.9] * 20), dict(id='unseen', scores=[.9] * 20),
                  dict(id='miss', scores=[.1] * 20), dict(id='neg', scores=[.8] * 20)]
        labels = dict(hit=1, unseen=1, miss=1, neg=0)
        anchors = {k: dict(onset_index=20 if k == 'unseen' else 0, event_index=29) for k in ('hit', 'unseen', 'miss')}
        result = temporal(curves, labels, anchors)
        self.assertAlmostEqual(result['values']['mtta@0.1'], 2.9 / 3)
        self.assertEqual(result['maximum_actual_negative_frame_FAR'], 1.)
        self.assertEqual(result['positives_without_observed_onset_window'], 1)
        self.assertFalse(result['prediction_tail_extrapolated'])
        for k in anchors:
            anchors[k]['onset_index'] = 20
        self.assertEqual(temporal(curves, labels, anchors)['values']['mtta@0.1'], 0.)

    def test_clock_and_rounding(self):
        record = dict(total_frames=300, pts=[0.])
        result = index_anchor(record, dict(event_s=11., onset_s=8.), 30.)
        self.assertEqual((result['onset_index'], result['event_index']), (80, 110))
        self.assertEqual(index_anchor(record, dict(event_s=11., onset_s=-1.), 30.)['onset_index'], 0)
        shifted = index_anchor(dict(record, pts=[4.]), dict(event_s=15., onset_s=12.), 30.)
        self.assertEqual(result, shifted)
        slow = index_anchor(record, dict(event_s=11., onset_s=8.), 25.)
        self.assertEqual(slow['event_frame'], 275)
        self.assertEqual(slow['source_clock_hz'], 10)
        self.assertNotEqual(slow['event_index'], result['event_index'])

    def test_bad_inputs_rejected(self):
        curves = [dict(id='p', scores=[.9] * 20), dict(id='n', scores=[.5] * 20)]
        labels = dict(p=1, n=0)
        anchors = dict(p=dict(onset_index=0, event_index=30))
        with self.assertRaises(ValueError):
            temporal(curves + curves[:1], labels, anchors)
        with self.assertRaises(ValueError):
            temporal(curves, labels, {})
        for bad in (float('nan'), -1., 2.):
            changed = [dict(curves[0], scores=[bad]), curves[1]]
            with self.assertRaises(ValueError):
                temporal(changed, labels, anchors)
        with self.assertRaises(ValueError):
            temporal(curves, labels, dict(p=dict(onset_index=4, event_index=3)))

    def test_cli_rejects_missing_inputs_and_bad_budget(self):
        import run_author_eval
        for args in (['--output', 'not_created', '--hours', 'nan'],
                     ['--output', 'not_created', '--report-only'],
                     ['--output', 'not_created']):
            with patch.object(sys, 'argv', ['run_author_eval.py'] + args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_author_eval.main()
                self.assertEqual(error.exception.code, 2)

    def test_report_rejects_unpinned_metric_runtime(self):
        import sklearn
        from author_report import report
        with patch.object(sklearn, '__version__', 'not_frozen'):
            with self.assertRaises(RuntimeError):
                report(Path('not_opened'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
