"""Arm-independent CPU initialization, isolated from sampler/dropout RNG.

The pinned source head constructs a compatibility M1 adapter. It is replaced,
but its advanced RNG no longer chooses the final adapter's weights. All six
final adapters start from one canonical G3 topology inside a forked CPU RNG;
unused modules are removed without further random initialization.
"""
import torch
from ablation_modules import AblationAdapter, HISTORIES, MODES

POLICY = 'canonical-g3-shared-state-cpu-fork-v3'


def build_matched_adapter(mode, channels, width, dropout, seed=None):
    if mode not in MODES:
        raise ValueError('Unknown matched arm: ' + str(mode))
    seed = int(torch.initial_seed()) if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Explicit adapter initialization seed must be an integer in [0,2**63).')
    with torch.random.fork_rng(devices=[]):
        # CPU only: torch.manual_seed would also change CUDA RNG outside this fork.
        torch.default_generator.manual_seed(seed)
        adapter = AblationAdapter('g3_learned', channels=channels, width=width, dropout=dropout)
        if any(p.device.type != 'cpu' for p in adapter.parameters()):
            raise ValueError('Build the model on CPU before moving it to CUDA.')
    adapter.mode = adapter.arm = mode
    adapter.histories = HISTORIES[mode]
    adapter.attention = mode != 'r7_last_state'
    if mode != 'g3_learned':
        del adapter.fusion
    if not adapter.attention:
        del adapter.query
        del adapter.key
        del adapter.decay_raw
    adapter.initialization_seed = seed
    adapter.initialization_policy = POLICY
    return adapter
