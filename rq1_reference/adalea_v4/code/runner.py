"""One-process sequential 3-seed runner; all adaptive/resume state in one atomic checkpoint."""
import copy
import csv
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from data import ROOT, digest, prediction_record, read_csv, sample_frames, save_json, validation_metrics
from model import AdaLEA, adaptive_loss, validation_attc
from shared_test_metrics import grouped_test_metrics


class BudgetStop(RuntimeError):
    pass


def check_budget(deadline):
    if time.time() >= deadline:
        raise BudgetStop('Total budget reached; only completed epoch checkpoint is retained.')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(config, device):
    return AdaLEA(config['feature_dim'], config['embedding'], config['hidden'], config['dropout']).to(device)


def features_batch(records, store, device, train):
    frames = [sample_frames(r, train) for r in records]
    features = [store.get(r, f) for r, f in zip(records, frames)]
    tensors = [torch.from_numpy(np.stack([f[k] for f in features])).to(device) for k in range(3)]
    return (tensors[0].float(), tensors[1].float(), tensors[2].bool()), frames


@torch.no_grad()
def infer(model, records, store, device, deadline):
    model.eval()
    predictions = []
    for index, record in enumerate(records):
        check_budget(deadline)
        features, frames = features_batch([record], store, device, False)
        # No autocast for validation/test. No test targets/event times enter the predictor.
        logits, _ = model(*features)
        scores = logits[0].float().sigmoid().cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError('Non-finite inference score.')
        predictions.append(prediction_record(record, frames[0], scores))
        if (index+1) % 100 == 0:
            print('Inference %d/%d' % (index+1, len(records)), flush=True)
    return predictions


def train_epoch(model, optimizer, scaler, config, records, store, device, epoch, previous_attc, deadline):
    model.train()
    order = torch.randperm(len(records)).tolist()
    batches = [order[i:i+config['batch_size']] for i in range(0, len(order), config['batch_size'])]
    accumulation = config['accumulation']
    steps, losses, examples, total = 0, [], 0, 0.
    for window_start in range(0, len(batches), accumulation):
        window = batches[window_start:window_start+accumulation]
        n_window = sum(len(b) for b in window)
        optimizer.zero_grad(set_to_none=True)
        for selected in window:
            check_budget(deadline)
            batch = [records[i] for i in selected]
            features, frames = features_batch(batch, store, device, True)
            labels = torch.tensor([r['target'] for r in batch], device=device)
            event = torch.tensor([r['event_frame'] if r['target'] else -1 for r in batch], device=device)
            fps = torch.tensor([r['native_fps'] for r in batch], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16,
                                enabled=config['amp'] and device.type=='cuda'):
                logits, _ = model(*features)
            # Loss operations in FP32 even when predictor uses AMP.
            loss = adaptive_loss(logits, labels, torch.tensor(np.stack(frames), device=device), event,
                                 fps, previous_attc, config['gamma_frames'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; no silent fallback/continuation.')
            scaler.scale(loss * (len(batch) / n_window)).backward()
            examples += len(batch)
            total += float(loss.detach()) * len(batch)
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradients. Stop; do not silently skip optimizer steps.')
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        if steps % 10 == 0:
            print('Epoch %d: %d/%d examples, loss %.6f, optimizer_steps %d, ATTC_prev %.6fs' %
                  (epoch, examples, len(records), total/examples, steps, previous_attc), flush=True)
    if examples != len(records) or steps != (len(batches)+accumulation-1)//accumulation:
        raise RuntimeError('Incomplete epoch/incorrect accumulation.')
    return {'loss': total/examples, 'optimizer_steps': steps, 'microbatches': len(batches)}


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def train_seed(config, seed, records, store, directory, device, contract, deadline):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / 'resume_latest.pth'
    set_seed(seed)
    model = make_model(config, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['lr_gamma'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'] and device.type=='cuda', init_scale=1024., growth_interval=2000)
    completed, best_metric, best_state, best_epoch, previous_attc, history = 0, -float('inf'), None, None, 0., []
    if checkpoint_path.exists():
        # Trusted checkpoint generated by this package, not an arbitrary downloaded pickle.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['contract'] != contract or checkpoint['seed'] != seed:
            raise RuntimeError('Checkpoint/data/config/code/runtime contract mismatch; refuse resume.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        completed = checkpoint['completed_epochs']
        best_metric, best_state, best_epoch = checkpoint['best_metric'], checkpoint['best_model'], checkpoint['best_epoch']
        previous_attc, history = checkpoint['previous_attc_s'], checkpoint['history']
        if len(history) != completed or checkpoint['adaptive_epoch'] != completed:
            raise RuntimeError('Adaptive/history/epoch state mismatch.')
        restore_rng(checkpoint['rng'])
        print('Resume seed %d from completed epoch %d, ATTC %.6fs' % (seed, completed, previous_attc), flush=True)
    for epoch in range(completed+1, config['epochs']+1):
        check_budget(deadline)
        started = time.monotonic()
        train = train_epoch(model, optimizer, scaler, config, records['train'], store, device, epoch, previous_attc, deadline)
        predictions = infer(model, records['validation'], store, device, deadline)
        values = validation_metrics(predictions)
        attc = validation_attc(predictions)
        measured = values[config['selection']]
        if not np.isfinite(measured) or not np.isfinite(attc):
            raise RuntimeError('Invalid validation/curriculum statistic.')
        if measured > best_metric:
            best_metric, best_state, best_epoch = measured, cpu_state(model), epoch
        history.append({'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'], 'train': train,
                        'ATTC_previous_s': previous_attc, 'ATTC_next_s': attc, 'metrics': values,
                        'elapsed_s': time.monotonic()-started})
        previous_attc = attc
        scheduler.step()
        completed = epoch
        checkpoint = dict(contract=contract, seed=seed, completed_epochs=completed,
            model=cpu_state(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), rng=rng_state(), previous_attc_s=previous_attc,
            adaptive_epoch=completed, best_metric=best_metric, best_epoch=best_epoch,
            best_model=best_state, history=history)
        atomic_checkpoint(checkpoint_path, checkpoint)
        # These human-readable files are derived; resume authority is the single atomic checkpoint.
        save_json(directory / 'validation_history.json', history)
        save_json(directory / 'validation_predictions_last.json', predictions)
        save_json(directory / 'training_status.json', dict(seed=seed, completed_epochs=completed,
            target_epochs=config['epochs'], complete=completed==config['epochs'], best_epoch=best_epoch,
            best_metric=best_metric, previous_attc_s=previous_attc, checkpoint_sha256=digest(checkpoint_path)))
        print('Seed %d epoch %d: mAUC %.4f, mAP %.4f, ATTC_next %.4fs; %.1fs' %
              (seed, epoch, values['mAUC@'], values['mAP@'], attc, history[-1]['elapsed_s']), flush=True)
    if completed != config['epochs'] or best_state is None:
        raise RuntimeError('Test cannot run before full training schedule.')
    model.load_state_dict(best_state)
    predictions = infer(model, records['validation'], store, device, deadline)
    save_json(directory / 'validation_predictions.json', predictions)
    save_json(directory / 'validation_metrics.json', validation_metrics(predictions))
    if any(r['target'] is not None or r['event_frame'] is not None or r['onset_frame'] is not None for r in records['test']):
        raise RuntimeError('Test data must hide targets/anchors.')
    test = infer(model, records['test'], store, device, deadline)
    save_json(directory / 'test_tail_curves.json', test)
    scalar = [{'id': r['id'], 'score': r['score']} for r in test]
    save_json(directory / 'test_predictions.json', scalar)
    # FIRST test-label read in the runner, after checkpoint selection and prediction frozen.
    solution = read_csv(ROOT / 'metadata_snapshot/solution.csv')
    metrics = grouped_test_metrics(solution, scalar)
    save_json(directory / 'official_test_metrics.json', metrics)
    submission = directory / 'submission.csv'
    with submission.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'score'])
        writer.writeheader()
        writer.writerows(scalar)
    checked = subprocess.run([sys.executable, str(ROOT / 'metadata_snapshot/evaluate_submission.py'),
                              str(submission), str(ROOT / 'metadata_snapshot/solution.csv')],
                             text=True, capture_output=True, check=True)
    for line in checked.stdout.splitlines():
        if line.startswith('mAP ('):
            usage = line.split('(')[1].split(')')[0]
            if abs(float(line.rsplit(':', 1)[1])-metrics[usage]['mAP']) > 1e-6:
                raise RuntimeError('Official evaluator parity failure.')
    (directory / 'official_evaluator_stdout.txt').write_text(checked.stdout, encoding='utf-8')
    save_json(directory / 'completion.json', dict(seed=seed, contract=contract, complete=True,
        completed_epochs=completed, best_epoch=best_epoch, all_test_ids=len(scalar),
        checkpoint_sha256=digest(checkpoint_path), predictions_sha256=digest(directory/'test_predictions.json')))
    return metrics


def aggregate(output, config, contract):
    output = Path(output)
    if len(config['seeds']) != 3 or len(set(config['seeds'])) != 3:
        raise ValueError('Production aggregation requires three distinct seeds.')
    results = []
    for seed in config['seeds']:
        directory = output / ('seed_%d' % seed)
        completion = json.loads((directory / 'completion.json').read_text())
        if (not completion['complete'] or completion['contract'] != contract or completion['seed'] != seed
                or completion['completed_epochs'] != config['epochs'] or completion['all_test_ids'] != 1344
                or completion['checkpoint_sha256'] != digest(directory/'resume_latest.pth')
                or completion['predictions_sha256'] != digest(directory/'test_predictions.json')):
            raise RuntimeError('Incomplete/changed result cannot enter aggregate.')
        # Recompute from frozen predictions instead of trusting an edited metrics table.
        solution = read_csv(ROOT / 'metadata_snapshot/solution.csv')
        predictions = json.loads((directory/'test_predictions.json').read_text())
        results.append(grouped_test_metrics(solution, predictions))
    summary = {'seeds': config['seeds'], 'n': 3, 'sample_sd_ddof': 1,
               'paper_equivalence_confirmed': False, 'scopes': {}}
    for scope in ('Combined', 'Public', 'Private'):
        summary['scopes'][scope] = {}
        for key in ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1'):
            values = [r[scope][key] for r in results]
            summary['scopes'][scope][key] = {'per_seed': values, 'mean': statistics.mean(values), 'sd': statistics.stdev(values)}
        summary['scopes'][scope]['mTTA'] = None
    save_json(output / 'three_seed_summary.json', summary)
    with (output/'three_seed_summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['scope', 'metric', 'seed42', 'seed43', 'seed44', 'mean', 'sd'])
        writer.writeheader()
        for scope, metrics in summary['scopes'].items():
            for key, values in metrics.items():
                if values is not None:
                    writer.writerow(dict(scope=scope, metric=key, seed42=values['per_seed'][0], seed43=values['per_seed'][1],
                                         seed44=values['per_seed'][2], mean=values['mean'], sd=values['sd']))
    return summary
