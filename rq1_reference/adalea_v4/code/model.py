"""Paper-method reimplementation, NOT official Suzuki code.

AdaLEA Eq.4-6: Suzuki et al., CVPR2018 p.4.
DSA pattern: Chan et al. ACCV2016 author accident.py (hidden-conditioned objects).
QRNN fo-pooling: causal 2-step convolution and recurrence, native PyTorch.
All unspecified dimensions/reductions are exposed and documented in PROTOCOL_VI.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AdaLEA(nn.Module):
    def __init__(self, feature_dim=4096, embedding=256, hidden=512, dropout=.5):
        super().__init__()
        self.global_embed = nn.Linear(feature_dim, embedding)
        self.local_embed = nn.Linear(feature_dim, embedding)
        self.att_object = nn.Linear(embedding, embedding)
        self.att_hidden = nn.Linear(hidden, embedding, bias=False)
        self.att_score = nn.Linear(embedding, 1, bias=False)
        # Concatenate current/previous fusion = causal temporal convolution width 2.
        self.qrnn = nn.Linear(4 * embedding, 3 * hidden)
        self.output = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden
        self.embedding = embedding

    def forward(self, global_features, local_features, object_mask):
        if global_features.ndim != 3 or local_features.ndim != 4:
            raise ValueError('Expected B,T,D global and B,T,O,D local features.')
        if local_features.shape[:2] != global_features.shape[:2] or object_mask.shape != local_features.shape[:3]:
            raise ValueError('Feature/mask shapes differ.')
        g = self.global_embed(global_features)
        l = self.local_embed(local_features)
        h = g.new_zeros((g.shape[0], self.hidden))
        c = h.clone()
        prev_x = g.new_zeros((g.shape[0], 2 * self.embedding))
        logits, attention = [], []
        for t in range(g.shape[1]):
            mask = object_mask[:, t].bool()
            e = self.att_score(torch.tanh(self.att_object(l[:, t]) + self.att_hidden(h)[:, None])).squeeze(-1)
            # All-empty frame stays finite; zero attention, not a fake global object.
            # FP16 clamp_min(1e-8) rounds to zero on all-empty frames -> 0/0.
            # Keep mask normalization in FP32 under AMP as well as validation.
            weights = torch.softmax(e.float().masked_fill(~mask, -1e4), -1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            local = (weights[..., None] * l[:, t]).sum(1)
            x = torch.cat((g[:, t], local), -1)
            z, f, o = self.qrnn(torch.cat((x, prev_x), -1)).chunk(3, -1)
            z, f, o = z.tanh(), f.sigmoid(), o.sigmoid()
            c = f * c + (1 - f) * z
            h = o * c
            logits.append(self.output(self.dropout(h)).squeeze(-1))
            attention.append(weights)
            prev_x = x
        return torch.stack(logits, 1), torch.stack(attention, 1)


def adaptive_loss(logits, labels, frames, event_frames, native_fps, previous_attc_s, gamma_frames=5.):
    """Frames/fps MUST use one physical clock. Ignore post-event jitter frames.

    No negative-class penalty on positive videos, FFR or AMC. Stable log-sigmoid
    equivalent to Eq.4/6. Reduction mean time then batch is a disclosed inference.
    """
    if not np.isfinite(previous_attc_s) or previous_attc_s < 0:
        raise ValueError('Previous ATTC must be finite nonnegative seconds.')
    logits = logits.float()
    labels = labels.bool()
    if frames.shape != logits.shape or labels.shape != logits.shape[:1]:
        raise ValueError('Loss metadata shape mismatch.')
    d = event_frames[:, None].float() - frames.float()
    positive_valid = d >= 0
    valid = torch.where(labels[:, None], positive_valid, torch.ones_like(positive_valid))
    if torch.any(valid.sum(1) == 0) or torch.any(labels & (event_frames < 0)) or torch.any(native_fps <= 0):
        raise ValueError('Invalid event anchor/FPS/no pre-event samples.')
    alpha = torch.exp(-(d - native_fps[:, None] * float(previous_attc_s) - gamma_frames).clamp_min(0))
    terms = torch.where(labels[:, None], F.softplus(-logits) * alpha, F.softplus(logits))
    return ((terms * valid).sum(1) / valid.sum(1)).mean()


def validation_attc(records):
    """Detached curriculum statistic, NOT RiskProp low-FAR mTTA.

    Threshold all distinct observed pre-event risks. TTC true positives only.
    Take maximum mean TTC for tied recall, then mean over attained positive recall.
    Discretization is inferred; Chan2016 uses recall deduplication/max TTC, but
    this uses actual native-frame seconds instead of its fixed 5s scaling.
    """
    positive = [r for r in records if r['target'] == 1]
    if not positive:
        raise ValueError('ATTC needs positive validation videos.')
    curves = []
    for r in positive:
        frames, p = np.asarray(r['frame_inds']), np.asarray(r['pred'], dtype=float)
        valid = frames <= r['event_frame']
        if not valid.any() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or r['native_fps'] <= 0:
            raise ValueError('Invalid ATTC risk curve/clock.')
        curves.append((p[valid], (r['event_frame'] - frames[valid]) / r['native_fps']))
    thresholds = np.unique(np.concatenate([p for p, _ in curves]))
    best_by_count = {}
    for start in range(0, len(thresholds), 256):
        threshold = thresholds[start:start + 256]
        count, total = np.zeros(len(threshold), dtype=int), np.zeros(len(threshold))
        for p, ttc in curves:
            hits = p[:, None] >= threshold
            detected = hits.any(0)
            first = hits.argmax(0)
            count += detected
            total += ttc[first] * detected
        for n, s in zip(count, total):
            if n:
                best_by_count[int(n)] = max(best_by_count.get(int(n), 0.), float(s / n))
    return float(np.mean(list(best_by_count.values()))) if best_by_count else 0.
