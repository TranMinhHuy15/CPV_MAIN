"""Full observed-clip causal rolling inference; resumable per-video artifacts."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from data import ROOT, digest, read_csv, save_json
from runner import check_budget, make_model
from shared_test_metrics import grouped_test_metrics
from temporal_metrics import grouped_temporal

TEMPORAL_KEYS = ('mTTA_test_author_index10_inferred_s',
    'mTTA_test_physical_strictFAR_inferred_s', 'mTTA_test_physical_TPonly_strictFAR_inferred_s')


@torch.no_grad()
def gpu_fullcurve_check(config, device, deadline):
    """Real target-device allocation/FP32 causal check, synthetic features only."""
    check_budget(deadline)
    if device.type != 'cuda':
        raise ValueError('This preflight requires actual CUDA.')
    model = make_model(config, device).eval()
    model.requires_grad_(False)
    n, d, o = config['eval_batch_windows'], config['feature_dim'], config['max_objects']
    g = torch.zeros(n,30,d,device=device)
    local = torch.zeros(n,30,o,d,device=device)
    mask = torch.ones(n,30,o,device=device,dtype=torch.bool)
    mask[0] = False
    full, attention = model(g,local,mask)
    prefix = model(g[:,:10],local[:,:10],mask[:,:10])[0]
    if (full.dtype != torch.float32 or not torch.isfinite(full).all()
            or not torch.isfinite(attention).all() or float(attention[0].abs().sum()) != 0.
            or not torch.allclose(full[:,:10],prefix,atol=1e-5,rtol=1e-5)):
        raise RuntimeError('Full-curve GPU preflight failed; no automatic batch reduction.')
    torch.cuda.synchronize()
    check_budget(deadline)
    return dict(passed=True,gpu=torch.cuda.get_device_name(device),batch_windows=n,
        feature_dim=d,sequence_length=30,eval_precision='FP32',synthetic_features_only=True,
        extractor_resident=True,peak_memory_allocated_gib=torch.cuda.max_memory_allocated()/(1<<30))


def endpoints(n):
    if type(n) is not int or n < 1:
        raise ValueError('Invalid frame count.')
    return np.arange((n-1)%3, n, 3, dtype=np.int64)


def windows(ends):
    ends = np.asarray(ends, dtype=np.int64)
    return np.maximum(ends[:, None]-np.arange(29, -1, -1)[None, :]*3, 0)


def probe_clock(record, deadline):
    check_budget(deadline)
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_streams',
        '-show_entries', 'frame=best_effort_timestamp_time:stream=duration,start_time', '-of', 'json', record['path']]
    checked = subprocess.run(command, check=True, capture_output=True, text=True,
        timeout=min(120., max(1., deadline-time.time())))
    result = json.loads(checked.stdout)
    pts = [float(f['best_effort_timestamp_time']) for f in result['frames']]
    streams = result['streams']
    if (len(pts) != record['total_frames'] or len(streams) != 1 or not all(math.isfinite(t) for t in pts)
            or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError('Video PTS/frame-count mismatch: '+record['id'])
    cut = float(streams[0].get('start_time', pts[0]))+float(streams[0]['duration'])
    if not math.isfinite(cut) or not pts[-1] < cut <= pts[-1]+.15:
        raise ValueError('Ambiguous stream boundary: '+record['id'])
    check_budget(deadline)
    return pts, cut


@torch.no_grad()
def rolling_scores(model, record, store, device, deadline, batch_windows=8):
    if any(record.get(k) is not None for k in ('target', 'event_frame', 'onset_frame')):
        raise ValueError('Full test predictor must receive blind record.')
    if type(batch_windows) is not int or batch_windows < 1:
        raise ValueError('Invalid inference batch.')
    model.eval()
    ends = endpoints(record['total_frames'])
    # Extract each frame once, not 30 times for the overlapping recurrent windows.
    needed = np.unique(np.append(ends, 0))
    check_budget(deadline)
    feature_arrays = store.get(record, needed)
    scores = []
    for start in range(0, len(ends), batch_windows):
        check_budget(deadline)
        inds = windows(ends[start:start+batch_windows])
        positions = np.searchsorted(needed, inds)
        if np.any(needed[positions] != inds) or np.any(inds > ends[start:start+batch_windows, None]):
            raise RuntimeError('Noncausal/incorrect feature mapping.')
        features = [torch.from_numpy(a[positions]).to(device) for a in feature_arrays]
        features = (features[0].float(), features[1].float(), features[2].bool())
        logits, _ = model(*features)
        scores.extend(logits[:, -1].float().sigmoid().cpu().tolist())
    values = np.asarray(scores)
    if values.shape != ends.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError('Invalid full-curve inference.')
    return ends, scores


def validate_curve(curve, record, capture_contract, expected_scalar=None):
    ends = endpoints(record['total_frames']).tolist()
    p = np.asarray(curve['scores'], dtype=float)
    t = np.asarray(curve['times_s'], dtype=float)
    if (curve['id'] != record['id'] or curve['frame_inds'] != ends
            or curve['capture_contract'] != capture_contract
            or curve['video_sha256'] != record['video_sha256']
            or curve['coverage'] != 'full_observed_clip' or curve['uses_future'] is not False
            or curve['prediction_tail_extrapolated'] is not False
            or p.shape != (len(ends),) or p.shape != t.shape or not np.isfinite(p).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or np.any((p < 0) | (p > 1)) or not math.isfinite(curve['cut_s'])
            or curve['cut_s'] <= t[-1] or not math.isfinite(curve['native_fps']) or curve['native_fps'] <= 0
            or curve['total_frames'] != record['total_frames']
            or curve['native_fps'] != record['native_fps'] or curve['first_pts_s'] > t[0]):
        raise ValueError('Changed/corrupt full curve: '+record['id'])
    delta = 0. if expected_scalar is None else abs(float(p[-1])-float(expected_scalar))
    if delta > 1e-4:
        raise RuntimeError('Rolling last score differs from original tail inference: '+record['id'])
    return delta


def load_selected_checkpoint(directory, config, seed, contract=None, legacy=False):
    directory = Path(directory)
    path = directory/'resume_latest.pth'
    completion = json.loads((directory/'completion.json').read_text())
    if (completion['seed'] != seed or completion['complete'] is not True
            or completion['completed_epochs'] != 50 or completion['all_test_ids'] != 1344
            or digest(path) != completion['checkpoint_sha256']
            or digest(directory/'test_predictions.json') != completion['predictions_sha256']):
        raise ValueError('Require unchanged completed 50-epoch checkpoint and 1344 original scores.')
    if not legacy and completion['contract'] != contract:
        raise ValueError('Training contract mismatch.')
    if legacy:
        expected = json.loads((ROOT/'legacy_v3_contract_files.json').read_text())
        previous = completion['contract']
        if previous['files'] != expected['files'] or previous['config'] != expected['config']:
            raise ValueError('Legacy run is not the audited V3 Fixed package/config.')
    # Only user-owned checkpoints. CLI requires explicit trust for legacy pickle.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['contract'] != completion['contract'] or checkpoint['seed'] != seed
            or checkpoint['completed_epochs'] != 50 or checkpoint['best_epoch'] != completion['best_epoch']
            or checkpoint['adaptive_epoch'] != 50 or len(checkpoint['history']) != 50):
        raise ValueError('Checkpoint selection/training state mismatch.')
    history = checkpoint['history']
    if [r['epoch'] for r in history] != list(range(1, 51)):
        raise ValueError('Incomplete history.')
    selection = checkpoint['contract']['config']['selection']
    values = [float(r['metrics'][selection]) for r in history]
    if not all(math.isfinite(v) for v in values) or int(np.argmax(values))+1 != checkpoint['best_epoch']:
        raise ValueError('Checkpoint must be selected by fixed validation criterion.')
    return checkpoint, path, completion


def evaluate_seed(config, seed, records, store, directory, device, contract, deadline,
                  training_directory=None, legacy=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    training_directory = Path(training_directory or directory)
    checkpoint, path, completion = load_selected_checkpoint(training_directory, config, seed, contract, legacy)
    model = make_model(config, device)
    model.load_state_dict(checkpoint['best_model'], strict=True)
    model.requires_grad_(False)
    del checkpoint
    # Labels/events are not read in this function until capture is complete.
    if any(any(r.get(k) is not None for k in ('target', 'event_frame', 'onset_frame')) for r in records):
        raise ValueError('Test records must be blind.')
    prior_rows = json.loads((training_directory/'test_predictions.json').read_text())
    prior = {r['id']: r['score'] for r in prior_rows}
    if len(prior_rows) != 1344 or len(records) != 1344 or len(prior) != 1344 or set(prior) != {r['id'] for r in records}:
        raise ValueError('Need exact unique1344 test IDs for capture.')
    capture_contract = dict(checkpoint_sha256=completion['checkpoint_sha256'],
        config=config, evaluation_contract=contract, training_contract=completion['contract'],
        sampling='30past_endpoints_native_stride3_end_anchored_zero_left_padding',
        precision='FP32 no autocast', prediction_tail_extrapolated=False,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU fixture',
        tf32_cudnn=torch.backends.cudnn.allow_tf32, tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
    manifest_path = directory/'capture_contract.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != capture_contract:
        raise ValueError('Capture contract changed; refuse mixing curves.')
    save_json(manifest_path, capture_contract)
    saved_hashes = {}
    completed_capture = directory/'capture_complete.json'
    if completed_capture.exists():
        previous_capture = json.loads(completed_capture.read_text())
        if previous_capture['capture_contract_sha256'] != digest(manifest_path):
            raise ValueError('Completed capture provenance changed.')
        saved_hashes = previous_capture['curve_sha256']
    curves, hashes, max_delta = [], {}, 0.
    for index, record in enumerate(records):
        check_budget(deadline)
        destination = directory/'full_test_curves'/(record['id']+'.json')
        if destination.exists():
            if saved_hashes and digest(destination) != saved_hashes.get(record['id']):
                raise ValueError('Completed curve was modified: '+record['id'])
            curve = json.loads(destination.read_text())
        else:
            pts, cut = probe_clock(record, deadline)
            ends, scores = rolling_scores(model, record, store, device, deadline, config['eval_batch_windows'])
            curve = dict(id=record['id'], scores=scores, frame_inds=ends.tolist(),
                times_s=[pts[i] for i in ends], first_pts_s=pts[0], cut_s=cut,
                total_frames=record['total_frames'], native_fps=record['native_fps'],
                video_sha256=record['video_sha256'], capture_contract=capture_contract,
                coverage='full_observed_clip', uses_future=False, prediction_tail_extrapolated=False)
            validate_curve(curve, record, capture_contract, prior[record['id']])
            save_json(destination, curve)
        max_delta = max(max_delta, validate_curve(curve, record, capture_contract, prior[record['id']]))
        hashes[record['id']] = digest(destination)
        curves.append(curve)
        if (index+1)%50 == 0:
            print('FULL CURVES seed%d %d/1344, max scalar delta %.8f' % (seed, index+1, max_delta), flush=True)
    expected_paths = {r['id']+'.json' for r in records}
    if {p.name for p in (directory/'full_test_curves').glob('*.json')} != expected_paths:
        raise ValueError('Extra/missing test curves.')
    save_json(directory/'capture_complete.json', dict(complete=True, curve_sha256=hashes,
        capture_contract_sha256=digest(manifest_path), test_videos=1344,
        checkpoint_sha256=digest(path), max_scalar_replay_delta=max_delta))
    # FIRST temporal-label/anchor use: predictions are now fixed and hashed.
    solution = read_csv(ROOT/'metadata_snapshot/solution.csv')
    save_json(directory/'original_test_predictions.json', prior_rows)
    scalar = grouped_test_metrics(solution, prior_rows)
    temporal, anchors = grouped_temporal(curves, solution)
    result = dict(complete=True, seed=seed, test_videos=1344, scalar=scalar, temporal=temporal,
        max_scalar_replay_delta=max_delta, training_changed=False,
        paper_test_equivalence_confirmed=False, mapping_verified=False,
        original_scalar_scores_preserved=True, selection='validation only before full test capture')
    save_json(directory/'inferred_test_anchors.json', anchors)
    save_json(directory/'full_test_metrics.json', result)
    save_json(directory/'full_eval_completion.json', dict(complete=True, seed=seed,
        capture_complete_sha256=digest(directory/'capture_complete.json'),
        metrics_sha256=digest(directory/'full_test_metrics.json'),
        original_predictions_sha256=digest(directory/'original_test_predictions.json'), evaluation_contract=contract))
    return result


def aggregate_full(output, config, contract, seeds=None):
    output = Path(output)
    seeds = list(config['seeds'] if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in (42, 43, 44) for s in seeds):
        raise ValueError('Expected distinct configured seeds.')
    results = []
    for seed in seeds:
        directory = output/('seed_%d'%seed)
        completion = json.loads((directory/'full_eval_completion.json').read_text())
        capture = json.loads((directory/'capture_complete.json').read_text())
        if (completion['complete'] is not True or completion['seed'] != seed or completion['evaluation_contract'] != contract
                or completion['metrics_sha256'] != digest(directory/'full_test_metrics.json')
                or completion['capture_complete_sha256'] != digest(directory/'capture_complete.json')
                or completion['original_predictions_sha256'] != digest(directory/'original_test_predictions.json')
                or capture['test_videos'] != 1344 or len(capture['curve_sha256']) != 1344
                or capture['capture_contract_sha256'] != digest(directory/'capture_contract.json')):
            raise ValueError('Incomplete/altered full evaluation.')
        curves = []
        for video_id, expected in capture['curve_sha256'].items():
            path = directory/'full_test_curves'/(video_id+'.json')
            if digest(path) != expected:
                raise ValueError('Changed curve: '+video_id)
            curves.append(json.loads(path.read_text()))
        recorded = json.loads((directory/'full_test_metrics.json').read_text())
        solution = read_csv(ROOT/'metadata_snapshot/solution.csv')
        scalar = grouped_test_metrics(solution, json.loads((directory/'original_test_predictions.json').read_text()))
        temporal, _ = grouped_temporal(curves, solution)
        # Scalars in this table intentionally retain original tail scores.
        for scope in temporal:
            for key in ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1'):
                if abs(scalar[scope][key]-recorded['scalar'][scope][key]) > 1e-12:
                    raise ValueError('Scalar metric replay mismatch.')
            for key in TEMPORAL_KEYS:
                if abs(temporal[scope][key]-recorded['temporal'][scope][key]) > 1e-12:
                    raise ValueError('Temporal metric replay mismatch.')
        results.append(recorded)
    summary = dict(complete=True, seeds=seeds, n=len(seeds), sample_sd_ddof=1,
        paper_equivalence_confirmed=False, mapping_verified=False, scopes={})
    rows = []
    for scope in ('Combined', 'Public', 'Private'):
        measures = {}
        for key in ('mAP', 'mAUC_grouped', 'mPAUC_sklearn_standardized', 'mPAUC_raw_divided_by_0.1')+TEMPORAL_KEYS:
            part = 'temporal' if key in TEMPORAL_KEYS else 'scalar'
            values = [r[part][scope][key] for r in results]
            measures[key] = dict(per_seed=values, mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values)>1 else None)
            row = dict(scope=scope, metric=key, n=len(seeds), mean=measures[key]['mean'], sd=measures[key]['sd'])
            row.update({'seed%d'%s: v for s, v in zip(seeds, values)})
            rows.append(row)
        summary['scopes'][scope] = measures
    save_json(output/'full_metrics_summary.json', summary)
    with (output/'full_metrics_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scope','metric','n','seed42','seed43','seed44','mean','sd'])
        writer.writeheader()
        writer.writerows(rows)
    return summary
