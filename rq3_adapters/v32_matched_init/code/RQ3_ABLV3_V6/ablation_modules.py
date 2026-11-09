"""RQ3 ablation V2 adapters (shared by the frozen-H7 and frozen-V6 packages). No labels/event times/future tokens.

One class, six explicit modes. Layers are created in EXACTLY the order of the source ThreeModules (M1), so with the
same torch seed C7/A7/A26/U3 start bit-identical to M1, and a7_m1 / g3_learned load M1 / M1+M2 state_dicts strictly.

  mode               window(s)             readout                         fusion                trainable
  c7_current_repeat  7 x current token u_t  attention + age (as M1)         -                     3,676,675
  r7_last_state      real H7               last LSTM state (no Q/K/decay)  -                     3,152,386
  a7_m1              real H7               attention + age                 -                     3,676,675  (= M1)
  a26_m1             real H26              attention + age                 -                     3,676,675
  u3_uniform         real H7, H13, H26     attention + age, each window    mean (fixed 1/3)      3,676,675
  g3_learned         real H7, H13, H26     attention + age, each window    softmax(Linear 1536->3) 3,681,286 (= M1+M2)

u_t = LayerNorm(W 2048->512 mean_time(F_t));  every window is a fresh LSTM512 pass (hidden/cell reset per window);
z = z_base + bound * sigmoid(Linear(Dropout c)) * tanh(Linear(Dropout c)),  gate bias -2, output weight/bias 0.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

WARMUP = 25
MODES = ('c7_current_repeat', 'r7_last_state', 'a7_m1', 'a26_m1', 'u3_uniform', 'g3_learned')
HISTORIES = {'c7_current_repeat': (7,), 'r7_last_state': (7,), 'a7_m1': (7,), 'a26_m1': (26,),
             'u3_uniform': (7, 13, 26), 'g3_learned': (7, 13, 26)}
PAST_SNIPPETS = {'c7_current_repeat': 0, 'r7_last_state': 6, 'a7_m1': 6, 'a26_m1': 25, 'u3_uniform': 25, 'g3_learned': 25}
TRAINABLE = {'c7_current_repeat': 3676675, 'r7_last_state': 3152386, 'a7_m1': 3676675, 'a26_m1': 3676675,
             'u3_uniform': 3676675, 'g3_learned': 3681286}
SOURCE_ARM = {'a7_m1': 'm1', 'g3_learned': 'm1_m2'}   # equivalent arm of the source ThreeModules
BASE_PARAMETERS = {'h7': 65207361, 'v6': 31636545}    # frozen base, total parameters (deployed = base + adapter)


def window(u, history, warmup=WARMUP):
    """[B,L,D] -> [B,count,history,D], oldest to newest, ending at each target token (identical to source window)."""
    if u.ndim != 3 or history not in (7, 13, 26) or warmup < history - 1 or u.shape[1] <= warmup:
        raise ValueError('Require B,L,D, H7/H13/H26 and enough warm-up.')
    targets = torch.arange(warmup, u.shape[1], device=u.device)
    indices = targets[:, None] - torch.arange(history - 1, -1, -1, device=u.device)
    return u[:, indices, :]


class AblationAdapter(nn.Module):
    def __init__(self, mode, channels=2048, width=512, dropout=.1, bound=1.):
        super().__init__()
        if mode not in MODES or channels < 1 or width < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid ablation adapter: ' + str(mode))
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('Positive finite correction bound required.')
        self.mode = self.arm = mode
        self.channels, self.width, self.bound = channels, width, bound
        self.histories = HISTORIES[mode]
        self.attention = mode != 'r7_last_state'
        # Creation order mirrors source ThreeModules: proj, norm, rnn, query, key, decay, dropout, gate, fusion, output.
        self.proj = nn.Linear(channels, width)
        self.norm = nn.LayerNorm(width)
        self.rnn = nn.LSTM(width, width, batch_first=True)
        if self.attention:
            self.query = nn.Linear(width, width, bias=False)
            self.key = nn.Linear(width, width, bias=False)
            self.decay_raw = nn.Parameter(torch.tensor(-2.))
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Linear(width, 1)
        nn.init.constant_(self.residual_gate.bias, -2.)
        if mode == 'g3_learned':
            self.fusion = nn.Linear(3 * width, 3)
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def token_features(self, x):
        # x: real temporal features [B,L,C,T] (sentinel static column already removed by the caller).
        if x.ndim != 4 or x.shape[2] != self.channels or x.shape[3] < 1:
            raise ValueError('Require temporal features [B,L,C,T] without the static sentinel column.')
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite encoder features.')
        return self.norm(self.proj(x.mean(-1)))

    def readout(self, u, history):
        current = u[:, WARMUP:]
        if self.mode == 'c7_current_repeat':
            w = current[:, :, None, :].expand(-1, -1, history, -1).contiguous()
        else:
            w = window(u, history)
        batch, count, length, width = w.shape
        states, _ = self.rnn(w.reshape(batch * count, length, width))   # reset per window
        if not self.attention:
            return states[:, -1].reshape(batch, count, width), None
        q = self.query(current).reshape(batch * count, 1, width)
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
        contexts, attentions = zip(*(self.readout(u, h) for h in self.histories))
        omega = None
        if self.mode == 'u3_uniform':
            context = torch.stack(contexts, -2).mean(-2)
        elif self.mode == 'g3_learned':
            omega = torch.softmax(self.fusion(torch.cat(contexts, -1)), -1)
            context = (torch.stack(contexts, -2) * omega[..., None]).sum(-2)
        else:
            context = contexts[0]
        dropped = self.dropout(context)
        raw = self.output(dropped).squeeze(-1)
        gate = torch.sigmoid(self.residual_gate(dropped)).squeeze(-1)
        correction = self.bound * gate * torch.tanh(raw)
        return baseline_logits + correction, {'correction': correction, 'raw': raw, 'gate': gate,
                                              'attention': attentions, 'scale_weights': omega}


def trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
