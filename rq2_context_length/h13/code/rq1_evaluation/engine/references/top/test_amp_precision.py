"""Check precision wrapper using the real released TOP head; not a CUDA memory benchmark."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


def make_samples(torch, numpy):
    last = 600 - 3 * numpy.arange(29, -1, -1)
    frames = (last[:, None] - 3 * numpy.arange(4, -1, -1)[None, :]).reshape(-1)
    positive = SimpleNamespace(have_accident=True, frame_inds=frames, clip_len=5, frame_interval=3,
                               accident_frame=600, abnormal_start_frame=540)
    negative = SimpleNamespace(have_accident=False, frame_inds=frames, clip_len=5, frame_interval=3,
                               accident_frame=None, abnormal_start_frame=None)
    return [positive, negative]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    import numpy
    import torch
    from accident_anticipation.models.top import AnticipationHead
    head = AnticipationHead(num_clips=30, clip_len=5, with_decoder=True, label_with='annotation', pos_weight=10)
    samples = make_samples(torch, numpy)
    logits = torch.linspace(-2., 2., 60 * 20).reshape(60, 20).requires_grad_()
    expected = head.loss_by_feat(logits, samples)
    expected_grad = torch.autograd.grad(sum(expected.values()), logits)[0]
    import amp_adapter
    amp_adapter.install()  # Idempotence: installing twice must not stack wrappers.
    actual = head.loss_by_feat(logits, samples)
    actual_grad = torch.autograd.grad(sum(actual.values()), logits)[0]
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert torch.equal(expected_grad, actual_grad)
    half_logits = logits.detach().half().requires_grad_()
    half_loss = head.loss_by_feat(half_logits, samples)
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in half_loss.values())
    sum(half_loss.values()).backward()
    assert half_logits.grad is not None and torch.isfinite(half_logits.grad).all()
    print('AMP precision adapter PASS: FP32 formula/gradient parity; half-logit loss FP32. CUDA feasibility not tested here.')


if __name__ == '__main__':
    main()
