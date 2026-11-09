"""Fast checks: source losses, causal sampling, metric parity and epoch accumulation."""
import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest


class ReproductionTests(unittest.TestCase):
    def test_native_jpeg_decode(self):
        import cv2
        import numpy as np
        from kaggle_adapter import NativeJpegDecode
        from mmaction.datasets.transforms import RawFrameDecode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "fixture.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 29.3, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(15):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = index * 13
                frame[:, index:index + 7, 1] = 240
                writer.write(frame)
            writer.release()
            cap = cv2.VideoCapture(str(video))
            for index in range(15):
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertTrue(cv2.imwrite(str(root / f"{index:06}.jpg"), frame))
            cap.release()
            data = dict(filename=str(video), frame_dir=str(root), total_frames=15, fps=30,
                        filename_tmpl="{:06}.jpg", modality="RGB", frame_inds=np.array([4, 7, 10, 7, 13]))
            expected = RawFrameDecode().transform(dict(data))
            actual = NativeJpegDecode().transform(dict(data))
            for left, right in zip(expected['imgs'], actual['imgs']):
                self.assertTrue(np.array_equal(left, right))
            self.assertEqual(actual['fps'], 30)
            self.assertEqual(actual['total_frames'], 15)
            actual['imgs'][1][:] = 0
            self.assertGreater(int(actual['imgs'][3].sum()), 0)  # Duplicates are independent arrays.
            with self.assertRaises(RuntimeError):
                NativeJpegDecode().transform(dict(data, frame_inds=np.array([15])))

    def test_published_preprocessing(self):
        from repro import preprocessing_annotations, read_csv, sha256, ANNOTATIONS_SHA256
        path, records = preprocessing_annotations(AUTHOR_ROOT)
        self.assertEqual(sha256(path), ANNOTATIONS_SHA256)
        inventory = read_csv(Path(__file__).parent / 'metadata_snapshot/video_inventory.csv')
        self.assertEqual(set(records), {r['id'] for r in inventory})
        differs = 0
        for row in inventory:
            original = records[row['id']]
            if row['split'] != 'train':
                self.assertEqual([original[k] for k in ('target', 'accident_frame', 'abnormal_start_frame')], ['-1'] * 3)
            elif row['target'] == '1':
                for key, timestamp in [('accident_frame', 'time_of_event'), ('abnormal_start_frame', 'time_of_alert')]:
                    self.assertEqual(int(original[key]), int(float(row[timestamp]) * float(original['fps'])))
                differs += int(int(original['accident_frame']) != int(float(row['time_of_event']) * 30))
        self.assertEqual(differs, 300)  # Guard against silently restoring the old 30-Hz anchors.

    def test_published_sampler(self):
        import numpy as np
        from v6_reference import V6SampleFramesBeforeAccident
        sampler = V6SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=True)
        data = dict(total_frames=1200, fps=30, accident_frame=600, start_index=0, target=True)
        result = sampler.transform(data)
        matrix = result["frame_inds"].reshape(30, 5)
        self.assertEqual(matrix[-1, -1], 600)
        self.assertTrue(np.all(np.diff(matrix, axis=1) == 3))
        self.assertTrue(np.all(np.diff(matrix[:, -1]) == 3))
        self.assertTrue(np.all(matrix <= matrix[:, -1, None]))
        test = sampler.transform(dict(data, target=None))
        self.assertEqual(test["frame_inds"][-1], 1199)

    def test_nexar_records_carry_have_accident(self):
        """The released MultiDataset stores `target`; the adapter must expose the same truth as `have_accident`."""
        import shutil
        from kaggle_adapter import NexarSubsetDataset
        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(Path(__file__).parent / "author_preprocessing/annotations.csv", Path(directory) / "annotations.csv")
            nexar = dict(data_root=directory, ann_file="annotations.csv", filename_tmpl="{:06}.jpg", start_index=0)
            for split, count in (("train", 1200), ("validation", 300), ("test", 1344)):
                dataset = NexarSubsetDataset(split=split, nexar=nexar, pipeline_video=[], pipeline_frame=[],
                                             modality="rgb", test_mode=split != "train")
                self.assertEqual(len(dataset), count)
                for index in range(len(dataset)):
                    info = dataset.get_data_info(index)
                    self.assertEqual(info["have_accident"], info["target"])
                    if split == "test":
                        self.assertIsNone(info["have_accident"])
                    else:
                        self.assertIsInstance(info["have_accident"], bool)

    def test_top_loss_matches_released_definition(self):
        """Independent re-derivation of the released 20-horizon BCE (pos_weight=10, annotation labels)."""
        import math
        import numpy as np
        import torch
        from accident_anticipation.models.top import AnticipationHead
        head = AnticipationHead(num_clips=30, clip_len=5, with_decoder=True, label_with="annotation", pos_weight=10)
        self.assertIn("pos_weight", dict(head.named_buffers()))
        last = 600 - 3 * np.arange(29, -1, -1)
        frames = (last[:, None] - 3 * np.arange(4, -1, -1)[None, :]).reshape(-1)
        positive = SimpleNamespace(have_accident=True, frame_inds=frames, clip_len=5, frame_interval=3,
                                   accident_frame=600, abnormal_start_frame=540)
        negative = SimpleNamespace(have_accident=False, frame_inds=frames, clip_len=5, frame_interval=3,
                                   accident_frame=None, abnormal_start_frame=None)
        torch.manual_seed(5)
        logits = torch.randn(60, 20, requires_grad=True)
        values = head.loss_by_feat(logits, [positive, negative])
        labels = np.zeros((60, 20), dtype=np.float32)
        for row, frame in enumerate(last):
            horizon = math.ceil((600 - frame) / 3)
            if horizon < 20 and frame >= 540:
                labels[row, horizon] = 1.
        self.assertEqual(int(labels.sum()), 20)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(labels), pos_weight=torch.tensor(10.))
        self.assertTrue(torch.allclose(values["loss_cls"], expected))
        self.assertEqual(set(values), {"loss_cls"})  # TOP has no auxiliary FFR/AMC terms.
        grad, = torch.autograd.grad(values["loss_cls"], logits)
        self.assertGreater(float(grad[:30].abs().sum()), 0)
        self.assertGreater(float(grad[30:].abs().sum()), 0)  # Negative videos contribute all-zero targets.

    def test_top_metric_and_submission_rule(self):
        import numpy as np
        from kaggle_adapter import SavingAnticipationMetric
        rng = np.random.RandomState(0)
        results = []
        for index in range(40):
            positive = index % 2 == 0
            pred = rng.rand(30, 20) * (.5 + .5 * positive)
            results.append(dict(pred=pred, target=positive, abnormal_start_ind=10 if positive else 0,
                                accident_ind=29 if positive else 0, video_id="%05d" % index, dataset="nexar",
                                frame_dir="", filename_tmpl="", type=None, frame_inds=np.arange(30) * 3,
                                abnormal_start_frame=30 if positive else None, accident_frame=87 if positive else None,
                                is_val=True, is_test=False))
        with tempfile.TemporaryDirectory() as directory:
            import os
            previous = os.getcwd()
            os.chdir(directory)
            try:
                os.environ['TOP_RUN_DIR'] = directory
                metric = SavingAnticipationMetric(fpr_max=0.1, vis_list=[], output_dir=None)
                values = metric.compute_metrics(results)
            finally:
                os.chdir(previous)
        for key in ("mAUC@", "mAUC@0.1", "mAP@", "mtta@0.1"):
            self.assertIn(key, values)
            self.assertTrue(np.isfinite(values[key]))
        from kaggle_adapter import TestPredictionMetric
        import torch
        scores = torch.tensor([[.1, .2], [.3, .9]])
        checker = TestPredictionMetric()
        checker.process(None, [dict(target=None, is_test=True, video_id="x", pred_score=scores)])
        self.assertAlmostEqual(checker.results[0]["score"], .9)

    def test_effective_batch_flush(self):
        import torch
        from mmengine.optim import OptimWrapper
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        wrapper = OptimWrapper(torch.optim.SGD(model.parameters(), lr=.1), accumulative_counts=4)
        for _ in range(2):
            wrapper.initialize_count_status(model, init_counts=0, max_counts=6)
            for _ in range(6):
                wrapper.update_params(model(torch.ones(1, 1)).square().mean())
            self.assertIsNone(model.weight.grad)
        self.assertAlmostEqual(float(model.weight.item()), .4096, places=5)

    def test_official_metric_parity(self):
        import importlib.util
        import pandas as pd
        from evaluate import grouped_test_metrics, partial_auc_raw_normalized, validate_submission
        snapshot = Path(__file__).parent / "metadata_snapshot"
        from repro import read_csv
        solution = read_csv(snapshot / "solution.csv")
        scores = read_csv(snapshot / "sample_submission.csv")
        result = grouped_test_metrics(solution, scores)
        spec = importlib.util.spec_from_file_location("official_nexar_evaluation", snapshot / "evaluate_submission.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        for usage in ("Public", "Private"):
            selected = [r for r in solution if r["Usage"] == usage]
            frame = pd.DataFrame(selected)
            frame["target"] = frame["target"].astype(int)
            submission = pd.DataFrame(scores)
            submission["score"] = submission["score"].astype(float)
            self.assertAlmostEqual(official.score(frame, submission), result[usage]["mAP"], places=12)
        with self.assertRaises(ValueError):
            validate_submission(solution, scores[:-1])
        with self.assertRaises(ValueError):
            validate_submission(solution, scores + scores[:1])
        self.assertAlmostEqual(partial_auc_raw_normalized([0, 0, 1, 1], [.1, .2, .8, .9]), 1.)
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(roc_auc_score([0, 1], [1., 0.], max_fpr=.1), .47368421052631576)

    def test_original_model_forward_and_backward(self):
        import torch
        from mmengine.config import Config
        from mmaction.registry import MODELS
        from mmaction.utils import register_all_modules
        register_all_modules()
        cfg = Config.fromfile(str(AUTHOR_ROOT / "configs/accident_anticipation/top/predict_occurrence_snippet.py"))
        cfg.model.backbone.pretrained = None  # Unit-test initialization only, not training config.
        cfg.model.cls_head.num_clips = 2
        model = MODELS.build(cfg.model)
        from mmengine.logging import MMLogger
        MMLogger.get_current_instance().setLevel('ERROR')
        model.init_weights()
        torch.set_num_threads(2)
        model.train()
        features = model.backbone(torch.randn(2, 3, 5, 32, 32))
        scores = model.cls_head(features)
        self.assertEqual(tuple(scores.shape), (2, 20))
        scores.square().mean().backward()
        self.assertIsNotNone(model.backbone.conv1.conv.weight.grad)

    def test_report_pipeline(self):
        from repro import read_csv, write_json
        from evaluate import create_reports
        snapshot = Path(__file__).parent / 'metadata_snapshot'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Synthetic fixture only: NOT a trained model result.
            write_json(root / 'training_status.json',
                       {'complete': True, 'completed_epochs': 50, 'best_checkpoint': 'SYNTHETIC.pth'})
            write_json(root / 'validation_metrics.json',
                       {'mAP@': .5, 'mAUC@': .5, 'mAUC@0.1': .5, 'mtta@0.1': 0.})
            write_json(root / 'test_predictions.json', read_csv(snapshot / 'sample_submission.csv'))
            print('SYNTHETIC metric self-check from sample_submission.csv; NOT trained model scores.')
            create_reports(root, snapshot)
            rows = read_csv(root / 'paper_comparison.csv')
            self.assertEqual(len(rows), 21)
            self.assertTrue(all(r['directly_comparable'] == 'False' for r in rows))
            self.assertTrue((root / 'official_evaluator_stdout.txt').exists())
            write_json(root / 'training_status.json', {'complete': False})
            with self.assertRaises(RuntimeError):
                create_reports(root, snapshot)

    def test_resume_artifact_guard(self):
        from repro import run_contract, restore_run, write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / 'previous'
            settings = dict(workspace=str(root / 'new'), resume_root=str(previous),
                            seed=42, epochs=50, batch_per_gpu=1, effective_batch_size=64,
                            world_size=2, amp=True)
            write_json(previous / 'run_manifest.json', {'contract': run_contract(settings)})
            write_json(previous / 'training_status.json', {'completed_epochs': 1,
                                                         'best_checkpoint': 'best_fixture.pth'})
            # Placeholder files test copying/guards only, not Torch loading.
            names = ['resume_latest.pth', 'best_fixture.pth', 'rng_epoch1_rank0.pth', 'rng_epoch1_rank1.pth']
            for name in names:
                write_json(previous / name, {'fixture': True})
            restore_run(settings)
            self.assertTrue(all((root / 'new/run' / name).exists() for name in names))
            with self.assertRaises(RuntimeError):
                restore_run(dict(settings, seed=43))
            old = run_contract(settings)
            old['protocol'] = 'riskprop-published-code-nexar-v1'
            write_json(previous / 'run_manifest.json', {'contract': old})
            with self.assertRaises(RuntimeError):
                restore_run(settings)
            write_json(previous / 'run_manifest.json', {'contract': run_contract(settings)})
            (previous / 'rng_epoch1_rank1.pth').unlink()
            with self.assertRaises(RuntimeError):
                restore_run(settings)


def main():
    global AUTHOR_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    AUTHOR_ROOT = Path(args.source)
    sys.path.insert(0, str(AUTHOR_ROOT))
    import accident_anticipation
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReproductionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        from repro import write_json
        write_json(args.report, {
            "tests_run": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
            "passed": result.wasSuccessful(), "scope": "CPU / library checks; not a GPU training benchmark",
        })
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
