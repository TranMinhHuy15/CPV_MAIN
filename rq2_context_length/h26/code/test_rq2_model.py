"""Small real-tensor tests; no Nexar test labels or paid GPU required."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from rq2_model import bounded_lstm, checkpoint_preserving_bn


class WindowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.rnn = torch.nn.LSTM(4, 4, batch_first=True).double()
        self.x = torch.randn(2, 55, 4, dtype=torch.double)

    def test_explicit_windows_and_shape(self):
        for horizon in (1, 26):
            actual = bounded_lstm(self.rnn, self.x, horizon, warmup=25)
            expected = torch.stack([self.rnn(self.x[:, t-horizon+1:t+1])[0][:, -1] for t in range(25, 55)], dim=1)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(tuple(actual.shape), (2, 30, 4))

    def test_no_future_or_too_old_effect(self):
        baseline = bounded_lstm(self.rnn, self.x, 26, warmup=25)
        changed = self.x.clone()
        changed[:, :5] += 123
        changed[:, 31:] -= 123
        replay = bounded_lstm(self.rnn, changed, 26, warmup=25)
        torch.testing.assert_close(baseline[:, 5], replay[:, 5])  # t30 only reads5..30.

    def test_no_cross_video_state(self):
        batch = bounded_lstm(self.rnn, self.x, 26, warmup=25)
        alone = bounded_lstm(self.rnn, self.x[1:], 26, warmup=25)
        torch.testing.assert_close(batch[1:], alone)

    def test_capture_chunk_invariance(self):
        full = bounded_lstm(self.rnn, self.x, 26, warmup=25)
        parts = [bounded_lstm(self.rnn, self.x[:, start:start+35], 26, warmup=25) for start in (0, 10, 20)]
        torch.testing.assert_close(full, torch.cat(parts, dim=1))

    def test_invalid_shapes_and_horizons_fail(self):
        for horizon in (0, 27):
            with self.assertRaises(ValueError):
                bounded_lstm(self.rnn, self.x, horizon, warmup=25)
        with self.assertRaises(ValueError):
            bounded_lstm(self.rnn, self.x[:, :25], 26, warmup=25)

    def test_new_output_creates_shared_index_before_data_audit(self):
        from rq2_capture import prepare_shared_index
        with tempfile.TemporaryDirectory() as folder:
            expected = Path(folder)/'shared_test_index'
            calls = []
            def audit(data,output,deadline):
                self.assertTrue(output.is_dir())
                self.assertEqual(output,expected)
                calls.append(data)
            reference = SimpleNamespace(engine=SimpleNamespace(prepare_data=audit))
            actual = prepare_shared_index(dict(workspace=str(Path(folder)/'long/seed_42'),data_root='data',deadline=12345.),reference)
            self.assertEqual(actual,expected)
            self.assertEqual(calls,['data'])

    def test_both_histories_backpropagate(self):
        for horizon in (1, 26):
            self.rnn.zero_grad()
            x = self.x.clone().requires_grad_(True)
            bounded_lstm(self.rnn, x, horizon, warmup=25).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(self.rnn.weight_ih_l0.grad.abs().sum().item(), 0)
            if horizon == 1:
                self.assertEqual(x.grad[:, :25].abs().sum().item(), 0)


class CheckpointTests(unittest.TestCase):
    def test_gradient_and_bn_buffers_match_noncheckpointed(self):
        torch.manual_seed(6)
        module = torch.nn.Sequential(torch.nn.Conv3d(2, 3, 1), torch.nn.BatchNorm3d(3), torch.nn.ReLU()).double().train()
        reference = copy.deepcopy(module)
        x = torch.randn(4, 2, 2, 4, 4, dtype=torch.double, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        checkpoint_preserving_bn(module, x).square().mean().backward()
        reference(other).square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad)
        for p, q in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        for (name, a), (_, b) in zip(module.named_buffers(), reference.named_buffers()):
            torch.testing.assert_close(a, b, msg=name)


if __name__ == '__main__':
    unittest.main()
