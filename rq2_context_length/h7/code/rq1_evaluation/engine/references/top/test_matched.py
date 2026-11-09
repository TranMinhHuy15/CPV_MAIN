"""Matched-V6 benchmark checks, including metric ties and pooled test scoring."""
import argparse
import ast
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_config_matches_fixed_v6_profile(self):
        from repro import config, run_contract
        settings = dict(workspace=str(HERE / 'unused'), seed=42, epochs=50, batch_per_gpu=2,
                        effective_batch_size=16, world_size=1, amp=True, workers_per_gpu=2,
                        verify_video_hashes=True, deadline=9999999999)
        cfg = config(settings, SOURCE, HERE / 'author_preprocessing')
        profile = json.loads((HERE / 'v6_profile.json').read_text())
        self.assertEqual(cfg.model.backbone, profile['backbone'])
        self.assertEqual(cfg.model.data_preprocessor, profile['data_preprocessor'])
        self.assertEqual(cfg.load_from, profile['load_from'])
        self.assertEqual(cfg.optim_wrapper.optimizer,
                         dict(type='SGD', lr=.002, momentum=.9, weight_decay=.0001))
        self.assertEqual(cfg.optim_wrapper.accumulative_counts, 8)
        self.assertEqual(cfg.optim_wrapper.clip_grad, dict(max_norm=40, norm_type=2))
        self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'mAUC@')
        self.assertEqual(list(cfg.param_scheduler[0].milestones), [20, 40])
        self.assertEqual(cfg.train_dataloader.dataset.pipeline_frame[0].type, 'V6SampleFramesBeforeAccident')
        contract = run_contract(settings)
        self.assertIn('v6_reference.py', contract['launcher_sha256'])
        self.assertIn('v6_profile.json', contract['launcher_sha256'])
        # Original TOP semantics are retained, not replaced with RiskProp losses.
        self.assertTrue(cfg.model.cls_head.with_decoder)
        self.assertEqual(cfg.model.cls_head.pos_weight, 10)
        self.assertEqual(cfg.model.cls_head.label_with, 'annotation')

    def test_metric_ties_are_order_invariant(self):
        import numpy as np
        from v6_reference import auc, v6_calculate
        for labels in ([0, 0, 1, 1], [1, 1, 0, 0]):
            self.assertEqual(auc(np.array(labels), np.full(4, .5), 1.)[0], .5)
        pred = [np.full(30, .5) for _ in range(4)]
        for labels in ([False, False, True, True], [True, True, False, False]):
            values = v6_calculate(SimpleNamespace(fpr_max=.1), pred, labels, [0]*4, [29]*4, '@')
            self.assertEqual(values['mAUC@'], .5)
            self.assertEqual(values['mAUC@0.1'], .5)
            self.assertEqual(values['mAP@'], .5)

    def test_sampler_matches_v6_on_all_label_branches(self):
        import numpy as np
        from v6_reference import V6SampleFramesBeforeAccident
        # A negative with an annotated collision in the same source video:
        # V6 validation chooses the safe beginning, NOT TOP's near-event segment.
        sampler = V6SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=True)
        data = dict(total_frames=1200, fps=30, start_index=0, target=False, accident_frame=600)
        values = sampler.transform(dict(data))
        self.assertEqual(values['frame_inds'][0], 0)
        self.assertEqual(values['frame_inds'][-1], 99)
        positive = sampler.transform(dict(data, target=True))
        self.assertEqual(positive['frame_inds'][-1], 600)
        test = sampler.transform(dict(data, target=None))
        self.assertEqual(test['frame_inds'][-1], 1199)
        np.random.seed(42)
        train = V6SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=False)
        values = train.transform(dict(data, target=True))
        self.assertGreaterEqual(values['frame_inds'][-1], 600)
        self.assertLessEqual(values['frame_inds'][-1], 602)

    def test_combined_test_recomputed_from_records(self):
        from evaluate import grouped_test_metrics
        from sklearn.metrics import average_precision_score
        solution, predictions = [], []
        for group in range(3):
            for usage, pairs in [('Public', [(0, .1), (1, .8)]),
                                 ('Private', [(0, .9), (0, .7), (1, .6), (1, .4)])]:
                for index, (target, score) in enumerate(pairs):
                    identity = '%s_%d_%d' % (usage, group, index)
                    solution.append(dict(id=identity, Usage=usage, group=str(group), target=str(target)))
                    predictions.append(dict(id=identity, score=score))
        values = grouped_test_metrics(solution, predictions)
        expected = average_precision_score([0, 1, 0, 0, 1, 1], [.1, .8, .9, .7, .6, .4])
        self.assertAlmostEqual(values['Combined']['mAP'], expected)
        average_of_reports = (values['Public']['mAP'] + values['Private']['mAP']) / 2
        self.assertNotAlmostEqual(values['Combined']['mAP'], average_of_reports)
        self.assertEqual(sum(g['n'] for g in values['Combined']['groups']), len(solution))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    SOURCE = Path(args.source)
    sys.path.insert(0, str(SOURCE))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (HERE / 'matched_checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        scope='CPU configuration, V6 sampler, tie-safe metric, pooled scoring; NOT RTX4080 preflight'), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
