"""Frozen full-image/object VGG FC7 features; frame-only causal extraction.

COCO detector is a disclosed substitute, not Suzuki's VOC+dataset fine-tuning.
Cache is shared across seeds, never predictor parameters/adaptive state.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch


def native_jpeg_frames(path, indices, total_frames):
    wanted = sorted(set(int(i) for i in indices))
    if not wanted or wanted[0] < 0 or wanted[-1] >= total_frames:
        raise ValueError('Native frame index out of range.')
    capture = cv2.VideoCapture(str(path))
    frames = {}
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != total_frames:
            raise RuntimeError('Video decode/count mismatch: %s' % path)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0]) or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - wanted[0]) > .01:
            raise RuntimeError('Native-frame seek failed.')
        requested = set(wanted)
        for index in range(wanted[0], wanted[-1] + 1):
            if not capture.grab():
                raise RuntimeError('Video ended early at %d' % index)
            if index in requested:
                ok, bgr = capture.retrieve()
                if not ok:
                    raise RuntimeError('Frame decode failed.')
                ok, jpg = cv2.imencode('.jpg', bgr)
                if not ok:
                    raise RuntimeError('JPEG roundtrip failed.')
                decoded = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                frames[index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return frames


class FrozenExtractor:
    def __init__(self, config, device):
        from torchvision.models import vgg16, VGG16_Weights
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        if device.type != 'cuda':
            raise RuntimeError('Production feature extraction requires CUDA; no hidden CPU fallback.')
        self.config, self.device = config, device
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg.to(device).eval()
        # FC7 activation: classifier[:5] = FC6/ReLU/dropout/FC7/ReLU, eval disables dropout.
        self.vgg.classifier = torch.nn.Sequential(*list(self.vgg.classifier.children())[:5])
        self.detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device).eval()
        for model in (self.vgg, self.detector):
            model.requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
        self.provenance = {
            'global_local': VGG16_Weights.IMAGENET1K_V1.url,
            'detector': FasterRCNN_ResNet50_FPN_Weights.COCO_V1.url,
            'features': 'FP32 extraction, resized 224x224; cache FP16; frozen FC7',
        }
        checkpoint_dir = Path(torch.hub.get_dir()) / 'checkpoints'
        self.provenance['weight_sha256'] = {}
        for url in (self.provenance['global_local'], self.provenance['detector']):
            file = checkpoint_dir / url.rsplit('/', 1)[1]
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(4 << 20), b''):
                    digest.update(block)
            self.provenance['weight_sha256'][file.name] = digest.hexdigest()

    @torch.no_grad()
    def check_max_crop_batch(self):
        """Exercise the configured worst VGG crop count, even with no detections."""
        n = 1+self.config['max_objects']
        images = torch.zeros(n, 3, 224, 224, device=self.device)
        for start in range(0, n, self.config['vgg_chunk']):
            output = self.vgg((images[start:start+self.config['vgg_chunk']] - self.mean) / self.std)
            if output.shape[1] != self.config['feature_dim'] or not torch.isfinite(output).all():
                raise RuntimeError('Max-crop VGG preflight failed.')
        torch.cuda.synchronize()
        return n

    @torch.no_grad()
    def __call__(self, rgb_frames):
        output = []
        for rgb in rgb_frames:
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(self.device).float() / 255.
            prediction = self.detector([tensor])[0]
            valid = prediction['scores'] >= self.config['detector_threshold']
            allowed = torch.tensor(self.config['traffic_classes'], device=self.device)
            valid &= (prediction['labels'][:, None] == allowed).any(1)
            selected = valid.nonzero().flatten()
            selected = selected[prediction['scores'][selected].argsort(descending=True)][:self.config['max_objects']]
            images = [rgb]
            for box in prediction['boxes'][selected].cpu().numpy():
                x1, y1, x2, y2 = np.rint(box).astype(int)
                x1, x2 = np.clip([x1, x2], 0, rgb.shape[1])
                y1, y2 = np.clip([y1, y2], 0, rgb.shape[0])
                if x2 > x1 and y2 > y1:
                    images.append(rgb[y1:y2, x1:x2])
            tensors = torch.from_numpy(np.stack([cv2.resize(image, (224, 224)) for image in images]))
            tensors = tensors.permute(0, 3, 1, 2).to(self.device).float() / 255.
            vectors = []
            for start in range(0, len(tensors), self.config['vgg_chunk']):
                vectors.append(self.vgg((tensors[start:start+self.config['vgg_chunk']] - self.mean) / self.std))
            vectors = torch.cat(vectors).cpu().numpy().astype(np.float16)
            local = np.zeros((self.config['max_objects'], 4096), dtype=np.float16)
            mask = np.zeros(self.config['max_objects'], dtype=bool)
            n = len(vectors) - 1
            local[:n], mask[:n] = vectors[1:], True
            if not np.isfinite(vectors).all():
                raise RuntimeError('Non-finite VGG feature; refusing to cache.')
            output.append((vectors[0], local, mask))
        return output


class FeatureStore:
    def __init__(self, directory, extractor, config, read_only_fallback=None):
        self.directory, self.extractor, self.config = Path(directory), extractor, config
        self.directory.mkdir(parents=True, exist_ok=True)
        self.read_only_fallback = Path(read_only_fallback) if read_only_fallback else None
        self.hits, self.misses, self.extraction_seconds = 0, 0, 0.
        self.cache_read_seconds, self.cache_write_seconds = 0., 0.

    def get(self, record, indices):
        path = self.directory / (record['id'] + '.npz')
        read_path = path
        if not path.exists() and self.read_only_fallback is not None:
            candidate = self.read_only_fallback / path.name
            if candidate.is_file():
                read_path = candidate
        existing = {}
        if read_path.exists():
            started = time.perf_counter()
            with np.load(read_path, allow_pickle=False) as arrays:
                if set(arrays.files) != {'indices', 'global_features', 'local_features', 'mask'}:
                    raise RuntimeError('Invalid feature cache schema: %s' % path)
                # NpzFile.__getitem__ decompresses the whole member EVERY call.
                # Materialize each member exactly once, including validation and lookup.
                cached_indices = arrays['indices']
                global_features = arrays['global_features']
                local_features = arrays['local_features']
                masks = arrays['mask']
                n = len(cached_indices)
                if (cached_indices.ndim != 1 or cached_indices.dtype.kind not in 'iu'
                        or np.any(cached_indices < 0) or np.any(np.diff(cached_indices.astype(np.int64)) <= 0)
                        or global_features.shape != (n, self.config['feature_dim'])
                        or local_features.shape != (n, self.config['max_objects'], self.config['feature_dim'])
                        or masks.shape != (n, self.config['max_objects']) or masks.dtype != np.bool_
                        or not np.isfinite(global_features).all() or not np.isfinite(local_features).all()):
                    raise RuntimeError('Corrupt/incompatible feature cache: %s' % path)
                if 'total_frames' in record and np.any(cached_indices >= record['total_frames']):
                    raise RuntimeError('Cached frame index outside native video.')
                # A warm read needs no per-frame dictionary/recompression.
                requested = np.asarray(indices, dtype=np.int64)
                positions = np.searchsorted(cached_indices, requested)
                present = positions < n
                if n:
                    present &= cached_indices[np.minimum(positions, n-1)] == requested
                if present.all():
                    self.hits += len(set(requested.tolist()))
                    self.cache_read_seconds += time.perf_counter()-started
                    return global_features[positions], local_features[positions], masks[positions]
                existing = {int(index): (global_features[k], local_features[k], masks[k])
                            for k, index in enumerate(cached_indices)}
            self.cache_read_seconds += time.perf_counter()-started
        missing = sorted(set(int(i) for i in indices) - set(existing))
        self.hits += len(set(indices)) - len(missing)
        if missing:
            reserve = self.config['cache_reserve_gib'] * (1 << 30)
            estimated = (len(existing) + len(missing)) * (1 + self.config['max_objects']) * self.config['feature_dim'] * 2
            # Atomic update temporarily holds old and new cache file. Never silently evict/reduce objects.
            if shutil.disk_usage(self.directory).free < reserve + estimated:
                raise RuntimeError('Feature cache disk budget too low; stop before overwriting/checkpoint loss.')
            started = time.perf_counter()
            frames = native_jpeg_frames(record['path'], missing, record['total_frames'])
            extracted = self.extractor([frames[i] for i in missing])
            if len(extracted) != len(missing):
                raise RuntimeError('Extractor returned wrong number of frames.')
            existing.update(zip(missing, extracted))
            self.extraction_seconds += time.perf_counter() - started
            self.misses += len(missing)
            order = sorted(existing)
            started = time.perf_counter()
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as file:
                np.savez_compressed(file, indices=np.asarray(order, dtype=np.int32),
                    global_features=np.stack([existing[i][0] for i in order]),
                    local_features=np.stack([existing[i][1] for i in order]),
                    mask=np.stack([existing[i][2] for i in order]))
            temporary.replace(path)
            self.cache_write_seconds += time.perf_counter()-started
        selected = [existing[int(i)] for i in indices]
        return tuple(np.stack([s[k] for s in selected]) for k in range(3))
