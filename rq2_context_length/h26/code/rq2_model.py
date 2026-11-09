"""Bounded past-only recurrence; original author losses are inherited unchanged."""
from contextlib import contextmanager, nullcontext
from functools import partial

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

WARMUP = 25
_registered = False


def bounded_lstm(rnn, features, history, warmup=WARMUP):
    if (features.ndim != 3 or history not in (1, 26) or warmup < history - 1
            or features.shape[1] <= warmup or rnn.bidirectional):
        raise ValueError('Require B,T,C features, causal history1/26 and enough warm-up.')
    targets = torch.arange(warmup, features.shape[1], device=features.device)
    indices = targets[:, None] - torch.arange(history-1, -1, -1, device=features.device)
    windows = features[:, indices, :]
    batch, count, length, channels = windows.shape
    output, _ = rnn(windows.reshape(batch * count, length, channels))
    return output[:, -1].reshape(batch, count, -1)


@contextmanager
def preserve_bn(module):
    saved = [(buffer, buffer.detach().clone()) for child in module.modules()
             if isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
             for buffer in (child.running_mean, child.running_var, child.num_batches_tracked)
             if buffer is not None]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


def checkpoint_preserving_bn(function, inputs, bn_module=None):
    module = bn_module if bn_module is not None else function
    return checkpoint(function, inputs, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), preserve_bn(module)))


def frame_grid(target_ends, total_frames):
    """Prepend25 past endpoints; each endpoint gets the original five past frames."""
    ends = np.asarray(target_ends)
    if (ends.ndim != 1 or ends.size == 0 or not np.issubdtype(ends.dtype, np.integer)
            or total_frames < 1 or np.any(ends < 0) or np.any(ends >= total_frames)
            or np.any(np.diff(ends) < 0)):
        raise ValueError('Invalid target endpoints.')
    prefix = np.maximum(ends[0] - 3 * np.arange(WARMUP, 0, -1), 0)
    all_ends = np.concatenate([prefix, ends])
    return np.maximum(all_ends[:, None] - 3 * np.arange(4, -1, -1), 0).astype(np.int32)


def register():
    global _registered
    if _registered:
        return
    from mmaction.registry import MODELS, TRANSFORMS
    from mmaction.models.recognizers import Recognizer3D
    from mmaction.models.backbones.resnet3d import Bottleneck3d
    from taa.models import AnticipationHead
    from taa.transforms import SampleFramesBeforeAccident

    @TRANSFORMS.register_module()
    class RQ2SampleFrames(SampleFramesBeforeAccident):
        def transform(self, results):
            results = super().transform(results)
            targets = results['frame_inds'].reshape(self.num_clips, self.clip_len)[:, -1]
            indices = frame_grid(targets, results['total_frames'])
            results.update(frame_inds=indices.reshape(-1), num_clips=len(indices))
            return results

    @MODELS.register_module()
    class RQ2Head(AnticipationHead):
        def __init__(self, history=26, **kwargs):
            if history not in (1, 26):
                raise ValueError('Fixed RQ2 arms use history1 or26.')
            kwargs.update(with_rnn=True, with_decoder=False, rnn_bidirectional=False,
                          two_stream=False, num_clips=30, clip_len=5)
            super().__init__(**kwargs)
            self.history = history

        def score_features(self, features):
            hidden = bounded_lstm(self.rnn, features, self.history)
            return self.fc_cls(self.dropout(hidden)).reshape(-1)

        def forward(self, x, rq2_batch=None, rq2_tokens=None, **kwargs):
            if rq2_batch is None or rq2_tokens is None or x.shape[0] != rq2_batch * rq2_tokens:
                raise ValueError('Missing/invalid batch boundaries for RQ2 recurrence.')
            features = x.mean(dim=tuple(range(2, x.ndim))) if x.ndim > 2 else x
            return self.score_features(features.reshape(rq2_batch, rq2_tokens, self.in_channels))

        def predict_by_feat(self, cls_scores, data_samples):
            count = cls_scores.numel() // len(data_samples)
            for sample in data_samples:
                indices = sample.frame_inds.reshape(-1, sample.clip_len)
                if len(indices) != count + WARMUP:
                    raise ValueError('Warm-up/output alignment mismatch.')
                sample.frame_inds = indices[WARMUP:].reshape(-1)
                sample.num_clips = count
            return super().predict_by_feat(cls_scores, data_samples)

    @MODELS.register_module()
    class RQ2Recognizer(Recognizer3D):
        def __init__(self, activation_checkpoint=True, **kwargs):
            super().__init__(**kwargs)
            if activation_checkpoint:
                for block in self.backbone.modules():
                    if isinstance(block, Bottleneck3d):
                        original = block.forward
                        block.forward = partial(_block_forward, block, original)

        def extract_feat(self, inputs, stage='neck', data_samples=None, test_mode=False):
            batch, tokens = inputs.shape[:2]
            x, kwargs = super().extract_feat(inputs, stage='neck', data_samples=data_samples, test_mode=test_mode)
            kwargs.update(rq2_batch=batch, rq2_tokens=tokens)
            if stage == 'head':
                x = self.cls_head(x, **kwargs)
            return x, kwargs

    _registered = True


def _block_forward(block, original, inputs):
    if block.training and torch.is_grad_enabled():
        return checkpoint_preserving_bn(original, inputs, bn_module=block)
    return original(inputs)
