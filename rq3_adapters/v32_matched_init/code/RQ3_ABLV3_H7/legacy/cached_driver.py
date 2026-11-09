"""Extract once per H7 seed, then fit one arm with the audited MMEngine Runner."""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from cache_store import (PROTOCOL, SEEDS, ARMS, load, save, sha, object_sha,
                         bind, commit, read_record, seal, verify, fixed_rng)

HERE = Path(__file__).resolve().parent


def tools_hashes():
    return {name: sha(HERE / name) for name in ('cache_store.py', 'cached_plugin.py', 'cached_driver.py', 'runtime_profile.py')}


def budget(deadline, reserve=0):
    if time.time() + reserve >= deadline:
        raise TimeoutError('SESSION_BUDGET_STOP: committed files preserved; resume next session.')


def validate_inputs(value):
    if set(value) != {'data_root','out_root','h7_runs','arms','encode_chunk'}:
        raise ValueError('Unexpected input fields.')
    if set(value['h7_runs']) != {str(s) for s in SEEDS}:
        raise ValueError('Only active seeds42/43; seed44 is held for separate verification.')
    if value['arms'] != list(ARMS) or type(value['encode_chunk']) is not int or not 1 <= value['encode_chunk'] <= 16:
        raise ValueError('Keep declared four-arm matrix and encode chunk1..16.')
    out, data = Path(value['out_root']).resolve(), Path(value['data_root']).resolve()
    for source in [data] + [Path(p).resolve() for p in value['h7_runs'].values()]:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('Output must not overlap dataset/original H7 input.')
    return value


def json_value(value):
    import numpy as np
    import torch
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError('Unsupported metadata: ' + str(type(value)))


def install_patch():
    import integration
    import study
    if getattr(integration, '_cached_v1_installed', False):
        return
    old_config, old_contract = integration.config, integration.contract

    def config(settings, source, data, phase='train'):
        cfg = old_config(settings, source, data, phase)
        # Full approved H7 checkpoint supplies all weights; avoid unused ImageNet download.
        cfg.model.backbone.pretrained = None
        if settings.get('cache_mode') == 'fit' and phase != 'evaluate':
            import cached_plugin
            cached_plugin.register()
            cfg.model.type = 'RQ3CachedRecognizer'
            cfg.model.data_preprocessor = dict(type='CachedFeaturePreprocessor')
            for name, split in (('train_dataloader', 'train'), ('val_dataloader', 'validation')):
                getattr(cfg, name).dataset = dict(type='CachedFeatureDataset',
                    cache_root=settings['cache_root'], split=split,
                    manifest_sha256=settings['cache_manifest_sha256'])
            from runtime_profile import register_hook
            register_hook()
            cfg.custom_hooks.append(dict(type='CacheEpochTimingHook'))
        return cfg

    def contract(settings):
        value = old_contract(settings)
        value.update(protocol=PROTOCOL, cache_tools=tools_hashes(),
                     study_seeds=list(SEEDS),
                     cache_manifest_sha256=settings.get('cache_manifest_sha256'),
                     cache_header=settings.get('cache_header'),
                     training_views='one fixed sampled/augmented train view per video; validation deterministic',
                     scientific_scope='exploratory cached-feature study; separate from online fresh-augmentation RQ3')
        return value

    integration.config, integration.contract = config, contract
    integration._cached_v1_installed = True
    study.PROTOCOL = PROTOCOL  # recover() uses the new snapshot contract.


def prepare_shared(inputs, deadline):
    import run_modules as run
    import repro
    original_sha = repro.sha256
    def checked_sha(path):
        budget(deadline)
        result = original_sha(path)
        budget(deadline)
        return result
    repro.sha256 = checked_sha
    try:
        return run.shared_data(inputs, deadline)
    finally:
        repro.sha256 = original_sha


def support_export(shared, export):
    """Portable source + metadata only. No MP4 links/copies in the feature export."""
    export = Path(export)
    files = {}
    source = Path(shared) / 'author_source'
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        relative = Path('author_source') / path.relative_to(source)
        target = export / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(path)
        if target.exists() and sha(target) != digest:
            raise ValueError('Export support changed: ' + str(relative))
        if not target.exists():
            shutil.copyfile(path, target)
        files[relative.as_posix()] = digest
    for name in ('annotations.csv', 'data_manifest.json'):
        path = Path(shared) / 'prepared_nexar' / name
        target = export / 'prepared_nexar' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(path):
            raise ValueError('Export metadata changed.')
        if not target.exists():
            shutil.copyfile(path, target)
        files[('prepared_nexar/' + name)] = sha(path)
    return files


def extract(inputs, base, runtime, seed, cache_root, deadline):
    import torch
    import run_modules as run
    from integration import extension_hashes
    from cached_plugin import cached_extract
    shared, source, data = prepare_shared(inputs, deadline)
    budget(deadline, 120)
    header = dict(protocol=PROTOCOL, seed=seed,
        checkpoint_sha256=base['checkpoint_sha256'], h7_contract_sha256=base['contract_sha256'],
        runtime=runtime, tools=tools_hashes(), extension_hashes=extension_hashes(),
        data_manifest=load(data / 'data_manifest.json'), view=0,
        sampling='source train pipeline once per video, stable per-video CPU RNG; source validation pipeline',
        representation='FP32 packed [55,2048,T+1,1,1]; final column exact original spatial/temporal mean')
    header_sha = bind(cache_root, header)
    settings = run.settings_for(inputs, {str(seed): base}, shared, runtime, 'm1', seed, deadline)
    settings['cache_mode'] = 'extract'
    runner = run.build(settings, 'smoke')
    run.load_weights(runner, base['checkpoint'], baseline_only=True)
    model = runner.model.eval()
    records, timings = [], []
    total = 1500
    for split, dataset in (('train', runner.train_dataloader.dataset),
                           ('validation', runner.val_dataloader.dataset)):
        expected = 1200 if split == 'train' else 300
        if len(dataset) != expected:
            raise ValueError('Unexpected source dataset count: %s %d' % (split, len(dataset)))
        for index in range(len(dataset)):
            budget(deadline, 120)
            info = dataset.get_data_info(index)
            video_id = str(info['video_id'])
            marker = Path(cache_root) / split / (video_id + '.json')
            if marker.exists():
                row = load(marker)
                if row.get('video_id') != video_id or row.get('split') != split:
                    raise ValueError('Committed cache ID differs from source.')
                read_record(cache_root, row, header_sha)
                records.append(row)
                if len(records) % 25 == 0:
                    print('CACHE RESUME verified %d/%d' % (len(records), total), flush=True)
                continue
            started = time.monotonic()
            with fixed_rng(seed, split, video_id):
                sample = dataset[index]
            metadata = json_value(sample['data_samples'].to_dict())
            # Fields travel as algorithm data in the original source PackActionInputs.
            metadata['target'] = bool(metadata['target'])
            metadata['is_val'] = bool(metadata['is_val'])
            metadata['is_test'] = bool(metadata['is_test'])
            batch = model.data_preprocessor(dict(inputs=[sample['inputs']],
                                            data_samples=[sample['data_samples']]), False)
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                packed, extra = model.extract_feat(batch['inputs'], stage='neck')
                features = packed.detach().cpu().numpy()
                # Verify CNN path and cache path before publishing this cache.
                if index == 0:
                    direct = model.extract_feat(batch['inputs'], stage='head')[0]
                    replay = cached_extract(model, packed.unsqueeze(0), stage='head')[0]
                    if not torch.allclose(direct, replay, atol=1e-5, rtol=1e-5):
                        raise ValueError('Online vs cached baseline parity failed.')
                    save(Path(cache_root) / ('parity_' + split + '.json'),
                         dict(passed=True, max_absolute_error=float((direct-replay).abs().max()),
                              scope='approved frozen baseline logits on first source sample'))
            # NPZ is uncompressed to avoid spending session time compressing FP32.
            if shutil.disk_usage(cache_root).free < features.nbytes + (2 << 30):
                raise RuntimeError('Disk reserve reached; preserve output and resume using another session.')
            row = commit(cache_root, split, video_id, features, metadata, header_sha)
            read_record(cache_root, row, header_sha)
            records.append(row)
            timings.append(time.monotonic() - started)
            remaining = total - len(records)
            print('EXTRACT seed%d %s %d/%d | total %d/1500 | %.1fs/video | ETA %.2fh | %.1fMiB/file' %
                  (seed, split, index+1, len(dataset), len(records), timings[-1],
                   remaining * sum(timings[-20:]) / len(timings[-20:]) / 3600,
                   row['bytes'] / 2**20), flush=True)
            del batch, packed, features, sample
    # Membership is built from the actual audited source datasets, never directory globbing.
    support = support_export(shared, cache_root)
    value = seal(cache_root, header, records, support)
    print('CACHE COMPLETE seed%d: %s | %d files | %.2fGiB arrays; Save Version output before ending session.' %
          (seed, cache_root, len(records), sum(r['bytes'] for r in records) / 2**30), flush=True)
    save(Path(cache_root) / 'extraction_timing.json', dict(new_records=len(timings),
        seconds_new_records=sum(timings), complete=value['complete']))


def fit(inputs, base, runtime, seed, arm, cache_root, deadline, mode):
    import run_modules as run
    import integration
    from training_state import assert_committed
    cache, manifest_sha = verify(cache_root, seed, base['checkpoint_sha256'], tools_hashes())
    # A byte-verified cache carries exactly the source/data contracts used at extraction.
    if cache['header']['extension_hashes'] != integration.extension_hashes():
        raise ValueError('Extraction source differs from current source.')
    settings = run.settings_for(inputs, {str(seed): base}, Path(cache_root), runtime, arm, seed, deadline)
    settings.update(cache_mode='fit', cache_root=str(Path(cache_root).resolve()),
                    cache_manifest_sha256=manifest_sha, cache_header=cache['header'])
    expected = integration.contract(settings)
    workspace = Path(settings['workspace'])
    receipt_path = workspace / 'smoke_receipt.json'
    if mode in ('smoke','profile'):
        run.smoke(settings)
        if mode == 'profile':
            from runtime_profile import benchmark,summarize_profiles
            benchmark(settings,load(receipt_path))
            print('PROFILE TABLE:',summarize_profiles(inputs['out_root'],seed),flush=True)
        return
    receipt = load(receipt_path) if receipt_path.exists() else {}
    if (receipt.get('contract') != expected or receipt.get('passed') is not True
            or receipt.get('optimizer_updates') != 3 or receipt.get('recurrent_gradient_seen') is not True
            or receipt.get('trainable_change') is not True or receipt.get('frozen_parameters_unchanged') is not True
            or receipt.get('initial_correction_max') != 0.):
        print('SMOKE START seed%d %s: 24 microbatches / 3 optimizer updates (discarded weights).' % (seed, arm), flush=True)
        run.smoke(settings)
        gc.collect()
    timing_path=workspace/'timing_profile.json'
    if not timing_path.exists() or load(timing_path).get('contract') != expected:
        from runtime_profile import benchmark,summarize_profiles
        benchmark(settings,load(receipt_path))
        summarize_profiles(inputs['out_root'],seed)
    budget(deadline, 900)
    started = time.monotonic()
    run.train(settings)
    status = load(workspace / 'run/training_status.json')
    assert_committed(workspace / 'run', status)
    print('ARM %s seed%d DONE invocation: %d/50 epochs, complete=%s, selected=%s, %.2fh this train call' %
          (arm, seed, status['completed_epochs'], status['complete'], status['best_checkpoint'],
           (time.monotonic()-started)/3600), flush=True)
    print('Validation mAP/mAUC/AP0.5/1.0/1.5 -> monitoring CSV. Test remains separate until all study runs complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--mode', required=True, choices=('extract', 'smoke', 'profile', 'train'))
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    parser.add_argument('--arm', default='m1', choices=ARMS)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    if not math.isfinite(args.deadline):
        parser.error('Finite absolute deadline required.')
    budget(args.deadline, 60)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import study
    study.verify_package()  # Original source remains byte-identical to the delivered manifest.
    inputs = validate_inputs(load(args.inputs))
    import run_modules as run
    runtime = run.runtime()
    base = study.baseline_input(inputs['h7_runs'][str(args.seed)], args.seed)
    install_patch()
    from process_control import exclusive_job
    output = Path(inputs['out_root']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve()
    if args.mode == 'extract':
        cache_root.mkdir(parents=True, exist_ok=True)
    elif not cache_root.is_dir():
        raise FileNotFoundError('Attach COMPLETE cache output Input; do not use the old pooled NPY cache.')
    # Serialize jobs sharing this output, including source preparation and snapshot commits.
    with exclusive_job(output / '.cache_study.lock'):
        if args.mode == 'extract':
            with exclusive_job(cache_root / '.extract.lock'):
                extract(inputs, base, runtime, args.seed, cache_root, args.deadline)
        else:
            fit(inputs, base, runtime, args.seed, args.arm, cache_root, args.deadline, args.mode)


if __name__ == '__main__':
    try:
        main()
    except TimeoutError as error:
        # A planned session stop is resumable; source runtime/data/model errors still fail loudly.
        print(str(error), flush=True)
