"""Check precision wrapper using the real author head; not a CUDA memory benchmark."""
import argparse
import random
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    import torch
    from taa.models import AnticipationHead
    head = AnticipationHead(num_clips=30, clip_len=5, label_with='constraint')
    samples = [SimpleNamespace(target=True), SimpleNamespace(target=False)]
    logits = torch.linspace(-2., 2., 60).reshape(2, 30).requires_grad_()
    random.seed(42)
    expected = head.loss_by_feat(logits, samples)
    expected_grad = torch.autograd.grad(sum(expected.values()), logits)[0]
    import amp_adapter
    amp_adapter.install()  # Idempotence: installing twice must not stack wrappers.
    random.seed(42)
    actual = head.loss_by_feat(logits, samples)
    actual_grad = torch.autograd.grad(sum(actual.values()), logits)[0]
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert torch.equal(expected_grad, actual_grad)
    half_logits = logits.detach().half().requires_grad_()
    random.seed(42)
    half_loss = head.loss_by_feat(half_logits, samples)
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in half_loss.values())
    sum(half_loss.values()).backward()
    assert half_logits.grad is not None and torch.isfinite(half_logits.grad).all()
    print('AMP precision adapter PASS: FP32 formula/gradient parity; half-logit loss FP32. CUDA feasibility not tested here.')


if __name__ == '__main__':
    main()
