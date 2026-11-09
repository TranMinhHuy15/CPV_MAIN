"""Trainable residual branch; no labels, event times, or future tokens enter forward."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('m1', 'm1_m2', 'm1_m3', 'full')
SEEDS = (42, 43, 44)
WARMUP = 25


def pack_encoder(feature_map):
    """N,C,T+1,1,1: temporal spatial-means followed by exact original all-axis mean."""
    if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 5:
        raise ValueError('Require encoder map [N,C,T,H,W].')
    temporal = feature_map.mean((-2, -1))
    # Keep original reduction order for the frozen H7 path, not mean(mean(H,W),T).
    static = feature_map.mean(dim=tuple(range(2, feature_map.ndim)))
    return torch.cat((temporal, static[..., None]), 2)[..., None, None]


def window(features, history, warmup=WARMUP):
    if features.ndim != 3 or history not in (7, 13, 26) or warmup < history - 1:
        raise ValueError('Require B,L,C, H7/H13/H26 and enough prefix.')
    if features.shape[1] <= warmup:
        raise ValueError('No target tokens.')
    targets = torch.arange(warmup, features.shape[1], device=features.device)
    indices = targets[:, None] - torch.arange(history - 1, -1, -1, device=features.device)
    return features[:, indices, :]


class ThreeModules(nn.Module):
    def __init__(self, arm='full', channels=2048, width=512, dropout=.1, bound=1.):
        super().__init__()
        if arm not in ARMS or channels < 1 or width < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid branch configuration.')
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('Positive finite correction bound required.')
        self.arm, self.channels, self.width, self.bound = arm, channels, width, bound
        self.multi = arm in ('m1_m2', 'full')
        self.motion = arm in ('m1_m3', 'full')
        self.proj = nn.Linear(channels, width)
        self.norm = nn.LayerNorm(width)
        self.rnn = nn.LSTM(width, width, batch_first=True)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.decay_raw = nn.Parameter(torch.tensor(-2.))
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Linear(width, 1)
        nn.init.constant_(self.residual_gate.bias, -2.)
        if self.multi:
            self.fusion = nn.Linear(3 * width, 3)
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)
        if self.motion:
            self.motion_net = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                            nn.Linear(width, width), nn.LayerNorm(width))
            self.motion_scale = nn.Parameter(torch.tensor(.1))
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def token_features(self, x):
        # B,L,C,T; temporal T is measured, never assumed equal to input frame count.
        if x.ndim != 4 or x.shape[2] != self.channels or x.shape[3] < 1:
            raise ValueError('Require temporal features [B,L,C,T]. Old pooled .npy is unsupported.')
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite encoder features.')
        u = self.norm(self.proj(x.mean(-1)))
        if self.motion:
            if x.shape[-1] < 2:
                raise ValueError('M3 needs at least two actual encoder time positions.')
            diff = (x[..., 1:] - x[..., :-1]).permute(0, 1, 3, 2)
            diff = F.linear(diff, self.proj.weight, None)
            motion = self.motion_net(diff) - self.motion_net(torch.zeros_like(diff))
            u = u + self.motion_scale * motion.mean(2)
        return u

    def readout(self, u, history):
        w = window(u, history)
        batch, count, length, width = w.shape
        # Each call resets hidden/cell; slicing states from H26 would not implement H7.
        states, _ = self.rnn(w.reshape(batch * count, length, width))
        q = self.query(u[:, WARMUP:]).reshape(batch * count, 1, width)
        energy = (q * self.key(states)).sum(-1) / math.sqrt(width)
        age = torch.arange(length - 1, -1, -1, device=u.device, dtype=u.dtype) / 25.
        weights = torch.softmax(energy - F.softplus(self.decay_raw) * age, -1)
        context = (weights[..., None] * states).sum(1).reshape(batch, count, width)
        return context, weights.reshape(batch, count, length)

    def forward(self, features, baseline_logits):
        u = self.token_features(features)
        expected = (u.shape[0], u.shape[1] - WARMUP)
        if baseline_logits.shape != expected or not torch.isfinite(baseline_logits).all():
            raise ValueError('Baseline/target-grid mismatch.')
        histories = (7, 13, 26) if self.multi else (7,)
        contexts, attentions = zip(*(self.readout(u, h) for h in histories))
        if self.multi:
            omega = torch.softmax(self.fusion(torch.cat(contexts, -1)), -1)
            context = (torch.stack(contexts, -2) * omega[..., None]).sum(-2)
        else:
            context, omega = contexts[0], None
        dropped = self.dropout(context)
        raw = self.output(dropped).squeeze(-1)
        gate = torch.sigmoid(self.residual_gate(dropped)).squeeze(-1)
        correction = self.bound * gate * torch.tanh(raw)
        logits = baseline_logits + correction
        diag = {'correction': correction, 'raw': raw, 'gate': gate, 'attention': attentions,
                'scale_weights': omega}
        return logits, diag
