"""One frozen model per process. No optimizer, trainer, or model updates."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['riskprop', 'top'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--stage', choices=['preflight', 'capture', 'report'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--local-check', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--local-videos', type=int, default=8, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.local_videos < 1 or (not args.local_check and args.local_videos != 8):
        parser.error('Reduced video fixtures are allowed only in local verification.')
    ref = ROOT / 'references' / args.model
    sys.path.insert(0, str(ref))
    from run_all import save, digest, DeadlineExceeded
    output = Path(args.output).resolve()
    if args.stage == 'report':
        from metrics import report
        report(output, args.model, args.seed, ref / 'metadata_snapshot', save)
        return
    import numpy as np
    import torch
    import cv2
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmengine.registry import init_default_scope
    from repro import prepare_source, config
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for real-checkpoint inference.')
    if not args.local_check:
        if sys.version_info[:3] != (3, 8, 5):
            raise RuntimeError('Pinned inference runtime requires Python 3.8.5.')
        required = {'torch': '2.1.0+cu118', 'torchvision': '0.16.0+cu118', 'mmengine': '0.10.7',
                    'mmcv': '2.2.0', 'numpy': '1.24.4', 'opencv-python': '4.8.1.78', 'scikit-learn': '1.3.2'}
        for name, version in required.items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError('Pinned inference runtime mismatch: ' + name)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    frozen = output / 'runs' / args.model / ('seed_%d' % args.seed)
    manifest = json.loads((frozen / 'run_manifest.json').read_text())
    status = json.loads((frozen / 'training_status.json').read_text())
    contract = manifest['contract']
    if not status['complete'] or status['completed_epochs'] != 50:
        raise ValueError('Checkpoint must be selected after completing 50 training epochs.')
    # Match training launcher bytes before reconstructing the model and transforms.
    for name, sha in contract['launcher_sha256'].items():
        if digest(ref / name) != sha:
            raise ValueError('Frozen launcher mismatch: ' + name)
    workspace = output / 'sources' / args.model
    workspace.mkdir(parents=True, exist_ok=True)
    archive = workspace / 'author_source_original.zip'
    if not archive.exists():
        import shutil
        shutil.copyfile(ref / 'original_source.zip', archive)
    if args.model == 'riskprop':
        # The frozen V6 source loader searches this name, rather than using
        # author_source_original.zip directly. Keep its implementation unchanged.
        import shutil
        candidate = workspace / 'RiskProp-main.zip'
        if not candidate.exists():
            shutil.copyfile(ref / 'original_source.zip', candidate)
    settings = dict(contract, workspace=str(workspace), workers_per_gpu=0,
                    deadline=args.deadline, resume_root=None, input_root=str(workspace))
    source = prepare_source(settings)
    cfg = config(settings, source, output, phase='evaluate')
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope('mmaction')
    from mmaction.registry import MODELS
    from mmengine.runner.checkpoint import load_checkpoint
    from kaggle_adapter import NativeJpegDecode
    cfg.model.backbone.pretrained = None
    if cfg.model.cls_head.get('with_rnn', False) or cfg.model.cls_head.get('two_stream', False):
        raise ValueError('Full-curve chunking is verified only for independent, single-stream causal snippets.')
    cfg.model.test_cfg = dict(max_testing_views=10)
    checkpoint = frozen / status['best_checkpoint']
    model = MODELS.build(cfg.model)
    model.init_weights()
    # Checkpoints are the user's own artifacts, pinned by SHA256 during CPU preparation.
    load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
    model = model.cuda().eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    directory = output / args.model / ('seed_%d' % args.seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'curves').mkdir(exist_ok=True)
    fingerprint = {'checkpoint_sha256': digest(checkpoint), 'source_commit': contract['source_commit'],
                   'evaluator_manifest_sha256': digest(ROOT / 'PACKAGE_SHA256.json'),
                   'capture_worker_sha256': digest(__file__),
                   'blind_index_sha256': digest(output / 'blind_index_contract.json'),
                   'GPU': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'precision': 'FP32 tensors, no autocast', 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                   'sampling': 'native_end_anchored_stride3_5past_frames', 'views': 10}
    contract_path = directory / 'capture_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != fingerprint:
        if not (args.local_check and args.stage == 'preflight' and not any((directory / 'curves').glob('*.json'))):
            raise ValueError('GPU/runtime/checkpoint/data changed; refuse mixing resumed curves.')
        save(directory / ('local_debug_previous_contract_%d.json' % time.time_ns()), json.loads(contract_path.read_text()))
    save(contract_path, fingerprint)
    records = json.loads((output / 'blind_index.json').read_text())
    prior = {r['id']: r['score'] for r in json.loads((frozen / 'test_predictions.json').read_text())}
    pipeline = copy.deepcopy(cfg.val_pipeline_frame)
    if [p['type'] for p in pipeline[1:]] != ['NativeJpegDecode', 'Resize', 'Flow', 'FormatShape', 'PackActionInputs']:
        raise ValueError('Unexpected reference evaluation transform order.')
    resize = Compose([pipeline[2]])

    class Fixed:
        def __init__(self, ends):
            self.ends = np.asarray(ends, dtype=np.int32)

        def __call__(self, data):
            assert data['target'] is None and data['accident_frame'] is None and data['abnormal_start_frame'] is None
            inds = np.maximum(self.ends[:, None] - np.arange(4, -1, -1) * 3, 0)
            assert (inds <= self.ends[:, None]).all() and (inds < data['total_frames']).all()
            return dict(data, frame_inds=inds.reshape(-1), clip_len=5, num_clips=len(self.ends), frame_interval=3)

    def infer(info, ends, cache=None):
        transforms = copy.deepcopy(pipeline)
        transforms[0] = Fixed(ends)
        if cache is not None:
            def cached(data):
                data.update(imgs=[cache['imgs'][int(i)] for i in data['frame_inds']],
                            original_shape=cache['original_shape'], img_shape=cache['img_shape'])
                return data
            transforms[1] = cached
            # Resize was already applied byte-identically once, to each unique decoded frame.
            del transforms[2]
        sample = Compose(transforms)(copy.deepcopy(info))
        if args.stage == 'preflight':
            infer.last_inputs = sample['inputs'].clone()
        with torch.no_grad():
            prediction = model.test_step({'inputs': [sample['inputs']], 'data_samples': [sample['data_samples']]})[0]
        values = prediction.pred_score.detach().float().cpu().numpy()
        if args.model == 'top':
            values = values.reshape(len(ends), 20).max(axis=1)
        else:
            values = values.reshape(len(ends))
        if not np.array_equal(np.asarray(prediction.frame_inds), ends):
            raise ValueError('Model output/endpoints mismatch.')
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid inference probability.')
        return values

    if args.stage == 'capture':
        preflight = json.loads((directory / 'preflight.json').read_text())
        if preflight['contract'] != fingerprint or not preflight['passed']:
            raise ValueError('Run the full six-checkpoint preflight first.')
        selected = records
    else:
        # Deterministic blind selection across dataset, never selected for high model scores.
        count = args.local_videos if args.local_check else 8
        selected = [records[i] for i in np.linspace(0, len(records) - 1, count, dtype=int)]
    started = time.perf_counter()
    max_delta, max_cache_delta, max_single_delta, max_independent_delta = 0., 0., 0., 0.
    for index, r in enumerate(selected):
        if time.time() >= args.deadline:
            raise DeadlineExceeded('Capture budget reached.')
        n = r['total_frames']
        ends = np.arange((n - 1) % 3, n, 3, dtype=np.int32)
        dest = directory / 'curves' / (r['id'] + '.json')
        if args.stage == 'capture' and dest.exists():
            old = json.loads(dest.read_text())
            values = np.asarray(old['scores'])
            if (old['checkpoint_sha256'] != fingerprint['checkpoint_sha256'] or old['frame_inds'] != ends.tolist()
                    or old['times_s'] != [r['pts'][int(i)] for i in ends] or len(values) != len(ends)
                    or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                    or abs(float(values[-1]) - prior[r['id']]) > 1e-4):
                raise ValueError('Damaged/stale saved curve: ' + r['id'])
            continue
        info = dict(dataset='nexar', filename=r['path'], frame_dir='unused', filename_tmpl='{:06}.jpg',
                    video_id=r['id'], target=None, have_accident=None, accident_frame=None,
                    abnormal_start_frame=None, total_frames=n, fps=30, start_index=0, is_val=False,
                    is_test=True, type=None, modality='RGB', sample_idx=index, text='')
        unique = np.unique(np.maximum(ends[:, None] - np.arange(4, -1, -1) * 3, 0))
        decoded = NativeJpegDecode().transform(dict(info, frame_inds=unique))
        resized = resize(decoded)
        cache = {'imgs': dict(zip(unique.tolist(), resized['imgs'])),
                 'original_shape': resized['original_shape'], 'img_shape': resized['img_shape']}
        # Align complete chunks from the END, so the final 30 snippets have
        # the same batch shape/order as the source evaluator's final window.
        first = len(ends) % 30
        chunks = ([ends[:first]] if first else []) + [ends[k:k + 30] for k in range(first, len(ends), 30)]
        scores = np.concatenate([infer(info, chunk, cache) for chunk in chunks])
        delta = abs(float(scores[-1]) - prior[r['id']])
        if delta > 1e-4:
            raise ValueError('Frozen scalar parity failure id=%s delta=%g; no long-run fallback.' % (r['id'], delta))
        max_delta = max(max_delta, delta)
        if args.stage == 'preflight':
            tail = ends[-30:]
            cached_tail = infer(info, tail, cache)
            cached_pixels = infer.last_inputs
            original = infer(info, tail)
            if not torch.equal(cached_pixels, infer.last_inputs):
                raise ValueError('Cached input pixels differ from the original JPEG/Resize pipeline.')
            cache_delta = float(np.max(np.abs(original - scores[-len(tail):])))
            print('PARITY_DIAGNOSTIC id=%s cached_vs_source=%g same_batch=%g scalar=%g' %
                  (r['id'], cache_delta, float(np.max(np.abs(original - cached_tail))), delta), flush=True)
            max_cache_delta = max(max_cache_delta, cache_delta)
            if cache_delta > 1e-5:
                raise ValueError('Cached decode/chunk inference differs from original source pipeline.')
            # Compare independent repeated snippets at the SAME CUDA batch shape.
            # A batch1/batch10 comparison also changes CUDA numerical kernels,
            # and therefore is a diagnostic, not evidence of temporal dependence.
            repeated = infer(info, [int(tail[-1])] * 10, cache)
            independent_delta = float(np.max(np.abs(repeated - scores[-1])))
            max_independent_delta = max(max_independent_delta, independent_delta)
            if independent_delta > 1e-5:
                raise ValueError('Same-shape independent snippet check failed.')
            single = infer(info, tail[-1:], cache)
            single_delta = abs(float(single[0]) - float(scores[-1]))
            max_single_delta = max(max_single_delta, single_delta)
            print('INDEPENDENCE_DIAGNOSTIC same_batch=%g batch1_vs10=%g' % (independent_delta, single_delta), flush=True)
        else:
            save(dest, {'id': r['id'], 'scores': scores.tolist(), 'times_s': [r['pts'][int(i)] for i in ends],
                        'frame_inds': ends.tolist(), 'coverage': 'full_observed_clip', 'uses_future': False,
                        'checkpoint_sha256': fingerprint['checkpoint_sha256'], 'video_sha256': r['video_sha256'],
                        'scalar_replay_delta': delta, 'prediction_tail_extrapolated': False})
        del decoded, resized, cache
        print('%s %s seed%d %d/%d id=%s delta=%.3g elapsed=%.1fs' %
              (args.stage, args.model, args.seed, index + 1, len(selected), r['id'], delta, time.perf_counter() - started), flush=True)
    if args.stage == 'preflight':
        save(directory / 'preflight.json', {'passed': True, 'contract': fingerprint, 'videos': len(selected),
            'max_scalar_delta': max_delta, 'max_cached_vs_original_delta': max_cache_delta,
            'max_single_vs_chunk_delta': max_single_delta, 'cached_input_pixels_exact': True,
            'max_same_shape_independent_snippet_delta': max_independent_delta,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'seconds_per_video_including_decode_and_checks': (time.perf_counter() - started) / len(selected),
            'warning': 'Preflight performs extra parity passes, not a guaranteed full-run runtime.'})
    else:
        ids = {p.stem for p in (directory / 'curves').glob('*.json')}
        complete = ids == {r['id'] for r in records} and len(ids) == 1344
        save(directory / 'capture_manifest.json', {'complete': complete, 'test_count': len(ids),
             'checkpoint_sha256': fingerprint['checkpoint_sha256'], 'training_changed': False})
        if not complete:
            raise ValueError('Incomplete/extra curve IDs; no final metrics.')


if __name__ == '__main__':
    main()
