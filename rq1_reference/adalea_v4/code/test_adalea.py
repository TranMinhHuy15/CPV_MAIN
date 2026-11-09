"""Small CPU fixtures are NOT production GPU checks or evidence of paper scores."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

from data import ROOT, prediction_record, sample_frames, validation_metrics
from features import FeatureStore, native_jpeg_frames
from model import AdaLEA, adaptive_loss, validation_attc
from runner import BudgetStop, aggregate, atomic_checkpoint, cpu_state, infer, make_model, train_epoch, train_seed
from shared_test_metrics import grouped_test_metrics
from worker import contract_for, measure_features, cache_disk_budget
from process_control import DeadlineExceeded, exclusive_job, run_checked


torch.set_num_threads(1)


def tiny_config():
    cfg = json.loads((ROOT/'config.json').read_text())
    cfg.update(feature_dim=8, embedding=4, hidden=8, max_objects=2, dropout=.1, amp=False,
               epochs=2, milestones=[1], cache_reserve_gib=0)
    return cfg


def record(video_id, positive=True, test=False):
    return dict(id=video_id, path='fixture.mp4', target=None if test else positive, native_fps=30.,
                event_frame=None if test or not positive else 135, onset_frame=None if test or not positive else 110,
                total_frames=160)


class FakeExtractor:
    def __call__(self, frames):
        output = []
        for frame in frames:
            value = float(frame.mean()/255.)
            output.append((np.full(8, value, dtype=np.float16), np.full((2, 8), value, dtype=np.float16), np.array([1, 0], bool)))
        return output


class FixtureStore:
    def get(self, record, frames):
        x = np.asarray(frames, dtype=np.float32)[:, None] / 200.
        g = np.repeat(x, 8, axis=1).astype(np.float16)
        l = np.repeat(g[:, None], 2, axis=1)
        mask = np.tile([1, 0], (len(frames), 1)).astype(bool)
        return g, l, mask


class Checks(unittest.TestCase):
    def test_loss_equation_frames_negative_and_post_event(self):
        logits = torch.tensor([[0., 0., 100.], [0., 0., 0.]], requires_grad=True)
        frames = torch.tensor([[0., 9., 11.], [0., 9., 11.]])
        loss = adaptive_loss(logits, torch.tensor([1, 0]), frames, torch.tensor([10., -1.]),
                             torch.tensor([10., 10.]), .2, 5.)
        expected = np.log(2)*((np.exp(-3)+1)/2+1)/2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            adaptive_loss(logits, torch.tensor([1, 0]), frames, torch.tensor([-1., -1.]), torch.ones(2), .2)

    def test_adaptive_changes_weights_not_negative(self):
        z = torch.zeros(1, 2)
        args = (torch.ones(1, dtype=torch.bool), torch.tensor([[0., 10.]]), torch.tensor([10.]), torch.tensor([10.]))
        self.assertGreater(adaptive_loss(z, *args, 1.).item(), adaptive_loss(z, *args, 0.).item())
        labels = torch.zeros(1, dtype=torch.bool)
        self.assertEqual(adaptive_loss(z, labels, args[1], torch.tensor([-1.]), args[3], 0.).item(),
                         adaptive_loss(z, labels, args[1], torch.tensor([-1.]), args[3], 1.).item())

    def test_causal_model_empty_objects_gradients(self):
        torch.manual_seed(5)
        model = AdaLEA(8, 4, 8, 0.).eval()
        g, l, m = torch.randn(2, 6, 8), torch.randn(2, 6, 3, 8), torch.zeros(2, 6, 3, dtype=torch.bool)
        m[0, :, 0] = True
        logits, weights = model(g, l, m)
        g2, l2 = g.clone(), l.clone()
        g2[:, 3:] += 100
        l2[:, 3:] -= 100
        self.assertTrue(torch.equal(logits[:, :3], model(g2, l2, m)[0][:, :3]))
        self.assertTrue(torch.allclose(logits[:, :3], model(g[:, :3], l[:, :3], m[:, :3])[0], atol=1e-6))
        self.assertEqual(float(weights[1].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.qrnn.weight.grad)

    def test_attc_uses_native_clock_and_excludes_after_event(self):
        r = dict(target=1, pred=[.1, .9, 1.], frame_inds=[0, 10, 30], event_frame=20, native_fps=10.)
        # Two thresholds, same recall1; upper envelope picks earliest warning=2s.
        self.assertEqual(validation_attc([r]), 2.)
        r['native_fps'] = 20.
        self.assertEqual(validation_attc([r]), 1.)
        with self.assertRaises(ValueError):
            validation_attc([dict(target=0)])

    def test_sampler_v6_and_test_hidden(self):
        r = record('A')
        frames = sample_frames(r, False)
        self.assertEqual(frames[-1], r['event_frame'])
        self.assertTrue(np.array_equal(np.diff(frames), np.full(29, 3)))
        t = record('T', test=True)
        self.assertEqual(sample_frames(t, False)[-1], 159)
        pred = prediction_record(t, sample_frames(t, False), np.full(30, .5))
        self.assertNotIn('target', pred)
        self.assertEqual(pred['coverage'], 'last_30_v6_endpoints')
        from data import read_csv
        inventory = read_csv(ROOT/'inference_inventory.csv')
        self.assertEqual(len(inventory), 2844)
        for row in inventory:
            if row['split'] != 'train':
                self.assertEqual(row['target'], '')
                self.assertEqual(row['time_of_event'], '')
                self.assertEqual(row['time_of_alert'], '')

    def test_atomic_cache_real_video_and_warm_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/'A.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30., (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(160):
                writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
            writer.release()
            r = dict(record('A'), path=str(video))
            store = FeatureStore(Path(tmp)/'cache', FakeExtractor(), tiny_config())
            first = store.get(r, [0, 12, 12, 30])
            second = store.get(r, [0, 12, 12, 30])
            self.assertEqual(store.misses, 3)
            self.assertEqual(store.hits, 3)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
            self.assertEqual(len(native_jpeg_frames(video, [12, 0], 160)), 2)
            with self.assertRaises(ValueError):
                native_jpeg_frames(video, [160], 160)

    def test_accumulation_partial_window_and_budget(self):
        cfg = tiny_config()
        model = make_model(cfg, torch.device('cpu'))
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        result = train_epoch(model, opt, scaler, cfg, [record(str(i), i%2==0) for i in range(18)],
                             FixtureStore(), torch.device('cpu'), 1, 0., time.time()+60)
        self.assertEqual(result['microbatches'], 9)
        self.assertEqual(result['optimizer_steps'], 2)
        with self.assertRaises(BudgetStop):
            train_epoch(model, opt, scaler, cfg, [record('A')], FixtureStore(), torch.device('cpu'), 1, 0., 0)

    def test_large_cache_decompresses_each_member_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeatureStore(tmp, FakeExtractor(), tiny_config())
            r = dict(record('A'), total_frames=1000)
            cfg = tiny_config()
            np.savez_compressed(Path(tmp)/'A.npz', indices=np.arange(1000, dtype=np.int32),
                global_features=np.zeros((1000, 8), np.float16),
                local_features=np.zeros((1000, 2, 8), np.float16), mask=np.zeros((1000, 2), bool))
            counts = {}
            original = np.lib.npyio.NpzFile.__getitem__
            def counted(file, name):
                counts[name] = counts.get(name, 0)+1
                return original(file, name)
            with patch.object(np.lib.npyio.NpzFile, '__getitem__', counted), patch('features.native_jpeg_frames') as decode:
                result = store.get(r, [999, 0, 999, 45])
                decode.assert_not_called()
            self.assertEqual(counts, dict(indices=1, global_features=1, local_features=1, mask=1))
            self.assertEqual(result[0].shape, (4, 8))
            self.assertEqual(store.hits, 3)
            self.assertEqual(store.misses, 0)
            # Partial cache merge must retain both old values and duplicate request order.
            with patch('features.native_jpeg_frames', return_value={1000: np.full((32, 32, 3), 255, np.uint8)}):
                result = store.get(dict(r, total_frames=1001), [999, 1000, 999])
            self.assertEqual(store.misses, 1)
            self.assertTrue(np.array_equal(result[0][0], result[0][2]))
            self.assertEqual(float(result[0][1, 0]), 1.)
            result2 = store.get(dict(r, total_frames=1001), [999, 1000, 999])
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(result, result2)))

    def test_preflight_measurement_is_cold_even_with_warm_production_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FeatureStore(root/'production_cache', FakeExtractor(), tiny_config())
            selected, frames = [record('A'), record('B', False)], [[0, 3], [0, 3]]
            fake_frames = {i: np.full((32, 32, 3), i, np.uint8) for i in (0, 3)}
            with patch('features.native_jpeg_frames', return_value=fake_frames):
                for r, f in zip(selected, frames):
                    store.get(r, f)
                for repeat in range(2):
                    values, measured = measure_features(store, selected, frames, root)
                    self.assertEqual(measured['unique_frames'], 4)
                    self.assertGreater(measured['extraction_seconds'], 0.)
                    self.assertGreater(measured['cache_write_seconds'], 0.)
                    self.assertGreater(measured['warm_cache_read_s'], 0.)
            self.assertEqual(store.misses, 4, 'Probe must not modify production cache counters')
            self.assertFalse(list(root.glob('preflight_probe_*')))

    def test_invalid_cache_and_nonfinite_cli_stop_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FeatureStore(root/'cache', FakeExtractor(), tiny_config())
            np.savez_compressed(root/'cache/A.npz', indices=np.array([0, 0]),
                global_features=np.zeros((2, 8), np.float16),
                local_features=np.zeros((2, 2, 8), np.float16), mask=np.zeros((2, 2), bool))
            with self.assertRaisesRegex(RuntimeError, 'Corrupt/incompatible'):
                store.get(record('A'), [0])
            import run_vast
            for hours in ('nan', 'inf', '-inf', '0', '169'):
                argv = ['run_vast.py', 'setup', '--output', str(root/'must_not_exist'), '--hours='+hours]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run_vast.main()
            self.assertFalse((root/'must_not_exist').exists())

    def test_disk_resume_credits_existing_cache_without_changing_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config()
            records = dict(train=[record('A')], validation=[record('B', False)], test=[record('T', test=True)])
            cold = cache_disk_budget(cfg, records, tmp)
            for split, rows in records.items():
                for r in rows:
                    from full_eval import endpoints
                    indices = (np.arange(r['total_frames'], dtype=np.int32) if split == 'train' else
                        np.unique(np.append(endpoints(r['total_frames']), 0)) if split == 'test' else np.unique(sample_frames(r, False)))
                    n = len(indices)
                    np.savez_compressed(Path(tmp)/(r['id']+'.npz'), indices=indices,
                        global_features=np.zeros((n, 8), np.float16),
                        local_features=np.zeros((n, 2, 8), np.float16), mask=np.zeros((n, 2), bool))
            warm = cache_disk_budget(cfg, records, tmp)
            self.assertEqual(cold['full_uncompressed_upper_bytes'], warm['full_uncompressed_upper_bytes'])
            self.assertGreater(warm['cached_frames_credited'], 0)
            self.assertEqual(warm['remaining_uncompressed_upper_bytes'], 0)
            self.assertGreater(warm['reserve_bytes'], 0)

    def test_checkpoint_resume_adaptive_rng_exact(self):
        cfg, device, store = tiny_config(), torch.device('cpu'), FixtureStore()
        records = {'train': [record('A'), record('B', False)],
                   'validation': [record('C'), record('D', False)], 'test': [record('T', test=True)]}
        # Controlled stop on test (after last epoch) leaves a coherent training checkpoint.
        original_infer = infer
        def stop_test(model, rows, store, device, deadline):
            if rows is records['test']:
                raise BudgetStop('fixture: no actual test ground truth used')
            return original_infer(model, rows, store, device, deadline)
        with tempfile.TemporaryDirectory() as tmp, patch('runner.infer', side_effect=stop_test):
            full, partial = Path(tmp)/'full', Path(tmp)/'partial'
            with self.assertRaises(BudgetStop):
                train_seed(cfg, 42, records, store, full, device, {'fixture': 1}, time.time()+60)
            # Stop after epoch1 checkpoint, before epoch2 begins; resume same unchanged contract.
            import runner
            real_atomic = atomic_checkpoint
            def save_then_stop(path, value):
                real_atomic(path, value)
                if value['completed_epochs'] == 1:
                    raise BudgetStop('fixture: saved epoch1')
            with patch('runner.atomic_checkpoint', side_effect=save_then_stop), self.assertRaises(BudgetStop):
                train_seed(cfg, 42, records, store, partial, device, {'fixture': 1}, time.time()+60)
            with self.assertRaises(BudgetStop):
                train_seed(cfg, 42, records, store, partial, device, {'fixture': 1}, time.time()+60)
            a = torch.load(full/'resume_latest.pth', weights_only=False)
            b = torch.load(partial/'resume_latest.pth', weights_only=False)
            self.assertEqual(a['completed_epochs'], b['completed_epochs'])
            self.assertEqual(a['previous_attc_s'], b['previous_attc_s'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertTrue(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']))
            self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
            with self.assertRaises(RuntimeError):
                train_seed(cfg, 42, records, store, partial, device, {'fixture': 2}, time.time()+60)

    def test_shared_validation_and_pooled_test_metrics(self):
        records = [prediction_record(record('A'), sample_frames(record('A')), np.linspace(.1, .9, 30)),
                   prediction_record(record('B', False), sample_frames(record('B', False)), np.linspace(.1, .3, 30))]
        self.assertEqual(len(validation_metrics(records)), 22)
        solution, pred = [], []
        for usage in ('Public', 'Private'):
            for group in range(3):
                for target in (0, 1):
                    video_id = '%s_%d_%d' % (usage, group, target)
                    solution.append(dict(id=video_id, Usage=usage, group=str(group), target=str(target)))
                    pred.append(dict(id=video_id, score=.8 if target else .2))
        result = grouped_test_metrics(solution, pred)
        self.assertEqual(result['Combined']['mAP'], 1.)
        self.assertEqual(sum(r['n'] for r in result['Combined']['groups']), 12)
        self.assertIsNone(result['Combined']['mTTA'])
        with self.assertRaises(ValueError):
            grouped_test_metrics(solution, pred[:-1])

    def test_aggregate_refuses_missing_seed(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            aggregate(tmp, tiny_config(), {'fixture': 1})

    def test_full_train_val_test_three_seed_fixture(self):
        cfg, store, device = tiny_config(), FixtureStore(), torch.device('cpu')
        solution = json.loads(json.dumps(__import__('data').read_csv(ROOT/'metadata_snapshot/solution.csv')))
        records = {'train': [record('A'), record('B', False)],
                   'validation': [record('C'), record('D', False)],
                   'test': [record(r['id'], test=True) for r in solution]}
        with tempfile.TemporaryDirectory() as tmp:
            for seed in cfg['seeds']:
                metrics = train_seed(cfg, seed, records, store, Path(tmp)/('seed_%d'%seed), device,
                                     {'fixture': 'three-seed'}, time.time()+180)
                self.assertEqual(sum(g['n'] for g in metrics['Combined']['groups']), 1344)
                self.assertIsNone(metrics['Combined']['mTTA'])
            summary = aggregate(tmp, cfg, {'fixture': 'three-seed'})
            self.assertEqual(summary['n'], 3)
            self.assertFalse(summary['paper_equivalence_confirmed'])
            self.assertTrue((Path(tmp)/'three_seed_summary.csv').exists())

    def test_process_deadline_and_output_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'child.log'
            with self.assertRaises(DeadlineExceeded):
                run_checked([sys.executable, '-u', '-c', "import time; print('started',flush=True); time.sleep(20)"],
                            log_path=log, deadline=time.time()+1.)
            self.assertIn('started', log.read_text())
            lock = Path(tmp)/'.job.lock'
            with exclusive_job(lock):
                with self.assertRaises(RuntimeError):
                    with exclusive_job(lock):
                        pass
            with exclusive_job(lock):
                pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
