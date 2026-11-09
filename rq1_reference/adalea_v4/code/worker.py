import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import tempfile
import math

import numpy as np
import torch

from data import ROOT, digest, load_records, sample_frames, save_json
from features import FeatureStore, FrozenExtractor
from model import adaptive_loss
from runner import BudgetStop, aggregate, check_budget, make_model, set_seed, train_seed
from full_eval import aggregate_full, evaluate_seed, endpoints, gpu_fullcurve_check


def contract_for(config):
    # All code, dataset annotations and evaluator bytes are pinned for resume.
    paths = (list(ROOT.glob('*.py')) + list((ROOT/'author_preprocessing').glob('*.csv'))
        + list((ROOT/'metadata_snapshot').rglob('*.csv')) + [ROOT/'inference_inventory.csv',
            ROOT/'metadata_snapshot/evaluate_submission.py', ROOT/'legacy_v3_contract_files.json'])
    paths = sorted(set(paths))
    return dict(config=config, files={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
        runtime={n: importlib.metadata.version(n) for n in
                 ('torch', 'torchvision', 'numpy', 'scikit-learn', 'opencv-python')},
        python=platform.python_version(), cuda=torch.version.cuda)


def runtime_checks():
    import torchvision
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Production expects exactly one visible CUDA GPU. No CPU/DDP fallback.')
    if '4080' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This release targets RTX4080. Use a new declared hardware protocol for another GPU.')
    if torch.__version__ != '2.1.0+cu118' or torchvision.__version__ != '0.16.0+cu118':
        raise RuntimeError('Use pinned isolated runtime or existing V6 runtime: torch2.1/cu118 vision0.16.')
    if torch.cuda.get_device_properties(0).total_memory < 15 * (1<<30):
        raise RuntimeError('Expected RTX4080-class 16GB; insufficient VRAM.')
    return torch.device('cuda')


def measure_features(store, selected, frames, output):
    """Always cold extraction + warm I/O, even if production cache already exists."""
    with tempfile.TemporaryDirectory(prefix='preflight_probe_', dir=str(output)) as tmp:
        probe = FeatureStore(tmp, store.extractor, store.config)
        started = time.perf_counter()
        extracted = [probe.get(r, f) for r, f in zip(selected, frames)]
        cold_seconds = time.perf_counter()-started
        started = time.perf_counter()
        warm = [probe.get(r, f) for r, f in zip(selected, frames)]
        warm_seconds = time.perf_counter()-started
        if not all(np.array_equal(a, b) for c, w in zip(extracted, warm) for a, b in zip(c, w)):
            raise RuntimeError('Feature cache warm/cold outputs differ.')
        if probe.misses <= 0 or not math.isfinite(probe.extraction_seconds) or probe.extraction_seconds <= 0:
            raise RuntimeError('No valid cold feature timing; refuse a misleading zero estimate.')
        measurements = dict(unique_frames=probe.misses, extraction_seconds=probe.extraction_seconds,
            cold_decode_extract_cache_s=cold_seconds, warm_cache_read_s=warm_seconds,
            cache_write_seconds=probe.cache_write_seconds, cache_read_seconds=probe.cache_read_seconds)
    return extracted, measurements


def cache_disk_budget(config, records, directory):
    """Reserve only missing frames on resume, plus space for atomic cache rewrites."""
    directory = Path(directory)
    total, remaining = 0, 0
    for split, rows in records.items():
        for record in rows:
            required = np.unique(np.append(endpoints(record['total_frames']), 0)) if split == 'test' else sample_frames(record, False)
            needed = record['total_frames'] if split == 'train' else len(required)
            credited = 0
            path = directory/(record['id']+'.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as arrays:
                    if set(arrays.files) != {'indices', 'global_features', 'local_features', 'mask'}:
                        raise RuntimeError('Invalid feature cache schema during disk check.')
                    indices = arrays['indices']
                if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0)
                        or np.any(indices >= record['total_frames']) or np.any(np.diff(indices.astype(np.int64)) <= 0)):
                    raise RuntimeError('Invalid cached indices during disk check.')
                credited = len(indices) if split == 'train' else len(set(indices.tolist()) & set(required.tolist()))
            total += needed
            remaining += needed-credited
    bytes_per_frame = (1+config['max_objects'])*config['feature_dim']*2+config['max_objects']+4
    # The per-video rewrite can temporarily coexist with its old cache file.
    largest_video = max((r['total_frames'] for rows in records.values() for r in rows), default=0)*bytes_per_frame
    reserve = max(config['cache_reserve_gib']*(1<<30), largest_video)
    return dict(full_uncompressed_upper_bytes=total*bytes_per_frame,
                remaining_uncompressed_upper_bytes=remaining*bytes_per_frame,
                reserve_bytes=reserve, cached_frames_credited=total-remaining)


def preflight(config, records, store, device, output, deadline, contract):
    set_seed(7)
    positive = next(r for r in records['train'] if r['target'])
    negative = next(r for r in records['train'] if not r['target'])
    selected = [positive, negative]
    frames = [sample_frames(r, False) for r in selected]
    extracted, measurements = measure_features(store, selected, frames, output)
    max_crops_verified = store.extractor.check_max_crop_batch()
    feature_peak_vram = torch.cuda.max_memory_allocated()/(1<<30)
    features = [torch.from_numpy(np.stack([v[k] for v in extracted])).to(device) for k in range(3)]
    features = (features[0].float(), features[1].float(), features[2].bool())
    model = make_model(config, device)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    # Main reset occurs BEFORE extractor construction/extraction, not after it.
    started = time.perf_counter()
    steps = 0
    actual_updates = [0]
    original_step = optimizer.step
    def counted_step(*args, **kwargs):
        actual_updates[0] += 1
        return original_step(*args, **kwargs)
    optimizer.step = counted_step
    for micro in range(16):
        check_budget(deadline)
        with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
            logits, _ = model(*features)
        loss = adaptive_loss(logits, torch.tensor([1, 0], device=device), torch.tensor(np.stack(frames), device=device),
            torch.tensor([positive['event_frame'], -1], device=device),
            torch.tensor([r['native_fps'] for r in selected], device=device), 0., config['gamma_frames'])
        if not torch.isfinite(loss):
            raise RuntimeError('Preflight non-finite loss.')
        scaler.scale(loss / config['accumulation']).backward()
        if (micro+1) % config['accumulation'] == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            if not torch.isfinite(norm):
                raise RuntimeError('Preflight non-finite gradients.')
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    torch.cuda.synchronize()
    predictor_seconds = time.perf_counter()-started
    if steps != 2 or actual_updates[0] != 2 or not any(not torch.equal(p, before[k]) for k, p in model.named_parameters()):
        raise RuntimeError('Preflight requires two actual parameter updates.')
    model.eval()
    # Keep predictor/optimizer resident while testing maximum VGG crop count:
    # this also covers extraction during training, not only extractor-alone VRAM.
    store.extractor.check_max_crop_batch()
    with torch.no_grad():
        prefix = model(*[f[:, :10] for f in features])[0]
        full = model(*features)[0][:, :10]
        if full.dtype != torch.float32 or not torch.isfinite(full).all() or not torch.isfinite(prefix).all():
            raise RuntimeError('Preflight requires finite FP32 evaluation outputs.')
        if not torch.allclose(prefix, full, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Preflight causal-prefix violation.')
    all_frames_upper = sum(r['total_frames'] for r in records['train']) + 30*300 + sum(len(endpoints(r['total_frames']))+1 for r in records['test'])
    disk = cache_disk_budget(config, records, store.directory)
    # Upper bound ignores compression; actual feature count depends on sampling and cache hits.
    upper_feature_hours = measurements['extraction_seconds']/measurements['unique_frames'] * all_frames_upper/3600
    report = dict(contract=contract, gpu=torch.cuda.get_device_name(0), success=True,
        optimizer_steps=actual_updates[0], microbatches=16, max_vgg_crops_verified=max_crops_verified,
        effective_batch=config['batch_size']*config['accumulation'],
        native_frame_extraction_measured_s=measurements['cold_decode_extract_cache_s'],
        unique_frames_measured=measurements['unique_frames'], feature_measurements=measurements,
        extraction_seconds=measurements['extraction_seconds'], predictor_16microbatch_s=predictor_seconds,
        feature_extraction_peak_vram_gib=feature_peak_vram,
        peak_vram_gib=torch.cuda.max_memory_allocated()/(1<<30),
        cache_uncompressed_upper_gib=disk['full_uncompressed_upper_bytes']/(1<<30),
        cache_remaining_upper_gib=disk['remaining_uncompressed_upper_bytes']/(1<<30), cache_disk_budget=disk,
        cache_free_gib=shutil.disk_usage(store.directory).free/(1<<30),
        all_native_train_frames_extraction_upper_hours=upper_feature_hours,
        timing_policy='Isolated cold cache on every preflight; never infer zero cost from warm production cache.',
        caution='Decode+feature upper estimate ONLY, not total 3-seed runtime; excludes repeated I/O/train/val/test.',
        test_read_during_preflight=False, paper_equivalence_confirmed=False)
    if report['cache_free_gib'] < (disk['remaining_uncompressed_upper_bytes']+disk['reserve_bytes'])/(1<<30):
        report['success'] = False
        report['disk_gate'] = 'Free disk smaller than missing-feature upper bound + atomic-update reserve.'
        save_json(output/'preflight.json', report)
        raise RuntimeError(report['disk_gate'])
    save_json(output/'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'train', 'evaluate'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--legacy-output')
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42,43,44], default=[42,43,44])
    parser.add_argument('--trust-v3-checkpoints', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= time.time():
        parser.error('--deadline must be finite and in the future.')
    config = json.loads((ROOT/'config.json').read_text())
    if args.mode == 'train' and args.seeds != [42,43,44]:
        raise ValueError('Training always runs all three configured seeds.')
    if args.legacy_output and (args.mode != 'evaluate' or not args.trust_v3_checkpoints):
        raise ValueError('Legacy evaluation requires explicit trust in your own V3 checkpoints.')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds.')
    if config['seeds'] != [42, 43, 44] or config['epochs'] != 50 or config['batch_size'] != 2 or config['accumulation'] != 8:
        raise RuntimeError('Production package contract is 3 seeds x50 epoch, batch2acc8. Create new protocol for changes.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = runtime_checks()
    contract = contract_for(config)
    contract['legacy_output'] = str(Path(args.legacy_output).resolve()) if args.legacy_output else None
    manifest = output/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise RuntimeError('Output belongs to a different experiment; refusing overwrite/resume.')
    if not manifest.exists() and (any(output.glob('seed_*')) or (output/'features').exists()):
        raise RuntimeError('Untracked seed outputs/features cannot be reused.')
    records = load_records(args.data_root, verify_hashes=True)
    save_json(manifest, contract)
    check_budget(args.deadline)
    budget_records = records if args.mode != 'evaluate' else dict(train=[], validation=[], test=records['test'])
    disk = cache_disk_budget(config, budget_records, output/'features')
    if shutil.disk_usage(output).free < disk['remaining_uncompressed_upper_bytes']+disk['reserve_bytes']:
        raise RuntimeError('Need missing-feature cache upper bound + atomic-update reserve FREE before extraction.')
    torch.cuda.reset_peak_memory_stats()
    extractor = FrozenExtractor(config, device)
    provenance_path = output/'feature_weights.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != extractor.provenance:
        raise RuntimeError('Feature weights changed; cached features cannot be reused.')
    save_json(provenance_path, extractor.provenance)
    fallback = None
    if args.legacy_output:
        previous = Path(args.legacy_output).resolve()
        if previous == output or previous in output.parents or output in previous.parents:
            raise ValueError('New output and legacy output must be separate, not nested.')
        if json.loads((previous/'feature_weights.json').read_text()) != extractor.provenance:
            raise ValueError('Legacy feature weights do not match frozen extraction protocol.')
        fallback = previous/'features'
    store = FeatureStore(output/'features', extractor, config, read_only_fallback=fallback)
    save_json(output/'gpu_fullcurve_check.json', gpu_fullcurve_check(config, device, args.deadline))
    preflight_path = output/'preflight.json'
    if args.mode == 'preflight':
        preflight(config, records, store, device, output, args.deadline, contract)
        return
    if args.mode == 'evaluate':
        for seed in args.seeds:
            evaluate_seed(config, seed, records['test'], store, output/('seed_%d'%seed), device,
                contract, args.deadline, training_directory=(Path(args.legacy_output)/('seed_%d'%seed)) if args.legacy_output else None,
                legacy=bool(args.legacy_output))
        print(json.dumps(aggregate_full(output, config, contract, args.seeds), indent=2), flush=True)
        return
    if not preflight_path.exists():
        raise RuntimeError('Run preflight and inspect cost/timing before paid full training.')
    report = json.loads(preflight_path.read_text())
    if not report['success'] or report['contract'] != contract or report['gpu'] != torch.cuda.get_device_name(0):
        raise RuntimeError('Preflight missing/failed/does not match current job.')
    for seed in config['seeds']:
        check_budget(args.deadline)
        directory = output/('seed_%d' % seed)
        completion_path = directory/'completion.json'
        if completion_path.exists():
            completed = json.loads(completion_path.read_text())
            if (completed['contract'] != contract or not completed['complete']
                    or completed['checkpoint_sha256'] != digest(directory/'resume_latest.pth')
                    or completed['predictions_sha256'] != digest(directory/'test_predictions.json')):
                raise RuntimeError('Completed seed altered; do not rerun silently into existing results.')
            print('Seed %d already completed; skip unchanged result.' % seed, flush=True)
        else:
            train_seed(config, seed, records, store, directory, device, contract, args.deadline)
        evaluate_seed(config, seed, records['test'], store, directory, device, contract, args.deadline)
    summary = aggregate_full(output, config, contract)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BudgetStop as error:
        print('BUDGET_STOP:', error, flush=True)
        sys.exit(75)
