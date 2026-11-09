"""CPU/fixture checks; synthetic queue metrics are NOT trained-model results."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'RiskProp_V6_VastAI_RTX4080'
SOURCE = HERE.parent / 'RiskProp_Kaggle_AuthorAligned/local_verified/author_source'
DATA = HERE.parent / 'RiskProp_Kaggle_AuthorAligned/local_verified/prepared_nexar'
sys.path.insert(0, str(SOURCE))
import repro
import run_vast
import run_three_seeds as queue


def settings(root, seed):
    return dict(workspace=str(root), input_root=str(DATA), resume_root=None,
                seed=seed, epochs=50, batch_per_gpu=2, effective_batch_size=16,
                world_size=1, amp=True, workers_per_gpu=2, verify_video_hashes=True, deadline=9999999999)


def fixture(root, seed, complete=True, score=.8):
    run = root / ('seed_%d' % seed) / 'run'
    run.mkdir(parents=True, exist_ok=True)
    repro.write_json(run / 'run_manifest.json', dict(contract=repro.run_contract(settings(run.parent, seed))))
    repro.write_json(run / 'training_status.json', dict(completed_epochs=50 if complete else 17,
        target_epochs=50, complete=complete, best_checkpoint='best.pth'))
    if complete:
        repro.write_json(run / 'validation_metrics.json', {key:score for key in ('mAP@','mAUC@','mAUC@0.1','mtta@0.1')})
        group = {'mAP':score, 'mAUC_grouped':score, 'mPAUC_sklearn_standardized':score,
            'mPAUC_raw_divided_by_0.1':score, 'groups':[dict(lead_time_s=lead, AP=score) for lead in (.5,1.,1.5)]}
        repro.write_json(run / 'official_test_metrics.json', dict(Public=group, Private=group))
        repro.write_json(run / 'evaluation_notes.json', dict(selected_checkpoint='best.pth',
            completed_epochs=50, test_was_not_used_for_selection=True))
    return run


class Checks(unittest.TestCase):
    def test_seed_configs_and_fixed_real_splits(self):
        from mmengine.utils import import_modules_from_strings
        from mmengine.registry import init_default_scope
        from mmaction.registry import DATASETS
        expected = {}
        for seed in queue.SEEDS:
            value = settings(HERE / 'unused', seed)
            repro.validate_profile(value)
            cfg = repro.config(value, SOURCE, DATA, 'smoke')
            self.assertEqual(cfg.randomness.seed, seed)
            self.assertEqual(cfg.launcher, 'none')
            self.assertEqual(cfg.optim_wrapper.accumulative_counts, 8)
            self.assertEqual(cfg.train_cfg.max_epochs, 50)
            self.assertEqual(cfg.val_dataloader.batch_size, 1)
            import_modules_from_strings(**cfg.custom_imports)
            init_default_scope('mmaction')
            splits = []
            for loader, count in ((cfg.train_dataloader,1200),(cfg.val_dataloader,300),(cfg.test_dataloader,1344)):
                dataset = DATASETS.build(loader.dataset)
                ids = {dataset.get_data_info(i)['video_id'] for i in range(len(dataset))}
                self.assertEqual(len(ids), count)
                splits.append(ids)
            if expected:
                self.assertEqual(splits, expected['ids'])
            expected['ids'] = splits
            self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
        for seed in (41, True, '42'):
            with self.assertRaises(ValueError):
                repro.validate_profile(settings(HERE,seed))

    def test_training_reference_and_built_in_eval_fix(self):
        for name in ('amp_adapter.py','evaluate.py','setup_environment.py','requirements-kaggle.txt',
                     'source_hashes.json','author_preprocessing/annotations.csv'):
            self.assertEqual((HERE/name).read_bytes(), (BASE/name).read_bytes())
        before = (BASE/'kaggle_adapter.py').read_text().split('@METRICS.register_module()\nclass TestPredictionMetric')[0]
        after = (HERE/'kaggle_adapter.py').read_text().split('@METRICS.register_module()\nclass TestPredictionMetric')[0]
        self.assertEqual(before,after)
        self.assertNotIn('patch_evaluation_worker', (HERE/'run_vast.py').read_text())
        self.assertIn('cfg.default_hooks["checkpoint"] = None', (HERE/'worker.py').read_text())
        from kaggle_adapter import TestPredictionMetric
        import torch
        metric = TestPredictionMetric()
        metric.process(None,[dict(target=None,is_test=True,video_id='example',pred_score=torch.tensor([.2,.8]))])
        self.assertAlmostEqual(metric.results[0]['score'],.8)
        with self.assertRaises(RuntimeError):
            metric.process(None,[dict(target=True,is_test=True,video_id='example',pred_score=torch.tensor([.8]))])

    def test_aggregate_requires_all_three_and_sample_sd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root,42,score=.7)
            fixture(root,43,score=.8)
            fixture(root,44,complete=False)
            report = queue.collect_summary(root)
            self.assertFalse(report['final_three_seed_report'])
            self.assertEqual(report['mean_sample_sd'], {})
            fixture(root,44,score=.9)
            report = queue.collect_summary(root)
            self.assertTrue(report['final_three_seed_report'])
            self.assertAlmostEqual(report['mean_sample_sd']['validation/mAP@']['mean'],.8)
            self.assertAlmostEqual(report['mean_sample_sd']['validation/mAP@']['sample_sd'],.1)
            manifest = root/'seed_44/run/run_manifest.json'
            data = queue.read_json(manifest)
            data['contract']['batch_per_gpu'] = 1
            repro.write_json(manifest,data)
            with self.assertRaisesRegex(RuntimeError,'different experiment contracts'):
                queue.collect_summary(root)

    def test_queue_resume_skip_and_no_parallel_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'data'
            data.mkdir()
            output = root/'results'
            args = ['--data-root',str(data),'--output-root',str(output),'--runtime-root',str(root/'runtime'),'--train']
            seen = []
            def partial(command,**unused):
                if "--seed" not in command:
                    return
                seed = int(command[command.index('--seed')+1])
                seen.append(command)
                fixture(output,seed,complete=False)
            with patch.object(queue,'run_checked',side_effect=partial), contextlib.redirect_stdout(io.StringIO()):
                queue.main(args)
            self.assertEqual(len(seen),1)
            self.assertNotIn('--resume-root',seen[0])
            seen.clear()
            def complete(command,**unused):
                if "--seed" not in command:
                    return
                seed = int(command[command.index('--seed')+1])
                seen.append(command)
                fixture(output,seed,score=.7+.1*(seed-42))
            with patch.object(queue,'run_checked',side_effect=complete), contextlib.redirect_stdout(io.StringIO()):
                queue.main(args)
            self.assertEqual([int(c[c.index('--seed')+1]) for c in seen],[42,43,44])
            self.assertIn('--resume-root',seen[0])
            self.assertNotIn('--resume-root',seen[1])
            with patch.object(queue,'run_checked') as child, contextlib.redirect_stdout(io.StringIO()):
                queue.main(args)
            child.assert_not_called()

    def test_seed_mismatch_resume_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root,42,complete=False)
            data = root/'data'
            data.mkdir()
            args = argparse.Namespace(data_root=str(data),workspace=str(run.parent),runtime_root=str(root/'runtime'),
                hours=24.,resume_root=str(run),evaluate_only=False,seed=43)
            inventory = [data/('%06d.mp4'%i) for i in range(2844)]
            with patch.object(Path,'rglob',return_value=iter(inventory)), self.assertRaisesRegex(RuntimeError,'seed does not match'):
                run_vast.validate_request(args)

    def test_syntax(self):
        for path in HERE.glob('*.py'):
            compile(path.read_text(encoding='utf-8'), str(path),'exec')
        # Terminal-first release; all entry points compiled above.


if __name__=='__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = dict(passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        scope='CPU configs, actual fixed video-ID splits, reference parity, test adapter, synthetic queue/resume/statistics; NOT three full GPU runs')
    (HERE/'three_seed_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
