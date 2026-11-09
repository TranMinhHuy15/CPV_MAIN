"""CPU-only fixture: real model/loader/Runner, checkpoint/snapshot and resume.

Never a paid job or a scientific result. Spatial size32, random initialization,
two train and two validation videos, two epochs with partial accumulation flush.
"""
import argparse
import copy
import csv
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    package, source = Path(args.package).resolve(), Path(args.source).resolve()
    sys.path[:0] = [str(package), str(source)]
    import cv2
    import numpy as np
    import torch
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    from repro import config, read_csv, write_json, author_validation
    from checkpoint_safety import restore_snapshot
    torch.set_num_threads(2)
    metadata = read_csv(package / 'author_preprocessing/annotations.csv')
    val_ids = author_validation(source)
    chosen = []
    for validation in (False, True):
        for positive in (False, True):
            chosen.append(next(r for r in metadata if r['is_test'] == '0'
                               and (r['id'] in val_ids) == validation
                               and (r['target'] == '1') == positive))
    chosen.append(next(r for r in metadata if r['is_test'] == '1'))
    with tempfile.TemporaryDirectory(prefix='v6_matched_fixture_') as directory:
        work = Path(directory)
        data = work / 'data'
        data.mkdir()
        for row in chosen:
            row.update(total_frames='150', fps='30', accident_frame='110' if row['target']=='1' else '-1',
                       abnormal_start_frame='80' if row['target']=='1' else '-1')
            folder = data / ('test' if row['is_test']=='1' else 'train')
            folder.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(folder / (row['id']+'.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 30., (48, 48))
            assert writer.isOpened()
            for i in range(150):
                frame = np.full((48, 48, 3), i % 240, dtype=np.uint8)
                frame[:, i % 40:i % 40 + 8, 1] = 255
                writer.write(frame)
            writer.release()
        with (data/'annotations.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
            writer.writeheader()
            writer.writerows(chosen)
        settings = dict(workspace=str(work), world_size=1, batch_per_gpu=2, effective_batch_size=16,
                        seed=42, amp=False, epochs=50, workers_per_gpu=0, deadline=time.time()+86400)
        cfg = config(settings, source, data)
        import_modules_from_strings(**cfg.custom_imports)
        cfg.launcher, cfg.model_wrapper_cfg = 'none', None
        cfg.log_level = 'ERROR'
        cfg.model.backbone.pretrained = None
        cfg.load_from, cfg.resume = None, False
        cfg.train_cfg.max_epochs = 2
        for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
            for transform in loader.dataset.pipeline_frame:
                if transform.type == 'Resize':
                    transform.scale = (32, 32)
            loader.dataset.pipeline_video = copy.deepcopy(loader.dataset.pipeline_frame)
        cfg.default_hooks.logger.interval = 9999
        cfg.custom_hooks[1]['deadline'] = time.time()  # Force safe stop after first completed epoch.
        run = Path(cfg.work_dir)
        write_json(run/'run_manifest.json', dict(contract=dict(protocol='fixture-only', seed=42)))
        os.environ['TOP_RUN_DIR' if 'TOP_' in package.name else 'ADALEA_RUN_DIR'] = str(run)
        runner = Runner.from_cfg(cfg)
        runner.train()
        status = json.loads((run/'training_status.json').read_text())
        assert status['completed_epochs']==1 and not status['complete']
        checkpoint = torch.load(run/'resume_latest.pth', map_location='cpu')
        assert checkpoint['meta']['epoch']==1 and 'optimizer' in checkpoint and 'param_schedulers' in checkpoint
        assert (run/status['best_checkpoint']).is_file()
        assert restore_snapshot(run, 'fixture-only', 42)
        del runner, checkpoint
        cfg.resume, cfg.load_from = True, str(run/'resume_latest.pth')
        cfg.custom_hooks[1]['deadline'] = time.time()+86400
        runner = Runner.from_cfg(cfg)
        runner.train()
        status = json.loads((run/'training_status.json').read_text())
        assert status['completed_epochs']==2 and status['complete']
        history = [json.loads(r) for r in (run/'validation_history.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in history] == [1, 2]
        assert all('mAUC@' in r['metrics'] and 'mAUC@0.1' in r['metrics'] for r in history)
        sample = runner.test_dataloader.dataset[0]
        assert sample['data_samples'].target is None and sample['data_samples'].accident_frame is None
        runner.model.eval()
        with torch.no_grad():
            prediction = runner.model.test_step(dict(inputs=[sample['inputs']], data_samples=[sample['data_samples']]))
        runner.test_evaluator.process(data_samples=prediction, data_batch=None)
        score = runner.test_evaluator.metrics[0].results[0]['score']
        assert np.isfinite(score) and 0 <= score <= 1
        write_json(package/'runner_fixture_checks.json', dict(passed=True,
            scope='CPU synthetic MP4 /32px/random init/2train+2val/2epochs/partial accumulation flush; NOT GPU or model performance',
            actual_runner_train_val_checkpoint_snapshot_resume=True, resume_epochs=[1,2],
            model_test_predictions_finite=True, test_labels_hidden=True,
            production_profile_not_executed=True))
        print('RUNNER FIXTURE PASS:', package.name)
        del runner
        # MMEngine holds log file handles; close before Windows TemporaryDirectory cleanup.
        logging.shutdown()


if __name__ == '__main__':
    main()
