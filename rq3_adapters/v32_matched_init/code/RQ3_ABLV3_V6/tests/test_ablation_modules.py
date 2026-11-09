"""CPU acceptance tests for the shared V2 ablation adapter (handoff V2, section 6)."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ablation_modules as A


def source_modules():
    spec = importlib.util.spec_from_file_location('reference_modules', ROOT / 'tests/reference_modules.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = source_modules()   # byte copy of the source modules.py (ThreeModules = M1 / M1+M2 actually trained)
C, D = 8, 6            # small fixture sizes


def make(mode, seed=0, **kw):
    torch.manual_seed(seed)
    return A.AblationAdapter(mode, channels=kw.get('channels', C), width=kw.get('width', D))


def live(module, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.output.weight.copy_(torch.randn(module.output.weight.shape, generator=g) * .7)
        module.output.bias.fill_(.1)
    return module.eval()


class Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_counts_real_size_and_no_dead_weights(self):
        for mode in A.MODES:
            a = make(mode, channels=2048, width=512)
            self.assertEqual(A.trainable_parameters(a), A.TRAINABLE[mode], mode)
        names = {m: set(make(m).state_dict()) for m in A.MODES}
        self.assertFalse({'query.weight', 'key.weight', 'decay_raw'} & names['r7_last_state'])
        self.assertNotIn('fusion.weight', names['u3_uniform'])
        self.assertIn('fusion.weight', names['g3_learned'])
        self.assertFalse(any(not k.startswith(('proj', 'norm', 'rnn', 'query', 'key', 'decay', 'residual_gate', 'fusion', 'output'))
                             for n in names.values() for k in n))

    def test_same_seed_initial_state_matches_source_m1_and_m1_m2(self):
        for mode, arm in [('c7_current_repeat', 'm1'), ('a7_m1', 'm1'), ('a26_m1', 'm1'), ('u3_uniform', 'm1'),
                          ('g3_learned', 'm1_m2')]:
            torch.manual_seed(5); ref = S.ThreeModules(arm, channels=C, width=D)
            ours = make(mode, seed=5)
            rs, os_ = ref.state_dict(), ours.state_dict()
            self.assertEqual(set(os_), set(rs), mode)
            for k in os_:
                self.assertTrue(torch.equal(os_[k], rs[k]), (mode, k))

    def test_a7_and_g3_reproduce_source_outputs_and_gradients(self):
        for mode, arm in [('a7_m1', 'm1'), ('g3_learned', 'm1_m2')]:
            torch.manual_seed(7); ref = S.ThreeModules(arm, channels=C, width=D)
            with torch.no_grad():
                ref.output.weight.normal_(0, .5); ref.output.bias.fill_(.2)
                if arm == 'm1_m2': ref.fusion.weight.normal_(0, .3)
            ours = A.AblationAdapter(mode, channels=C, width=D); ours.load_state_dict(ref.state_dict(), strict=True)
            ref.eval(); ours.eval()
            x = torch.randn(2, 55, C, 3); z = torch.randn(2, 30)
            a, da = ref(x, z); b, db = ours(x, z)
            self.assertTrue(torch.allclose(a, b, atol=1e-6), mode)
            a.square().mean().backward(); b.square().mean().backward()
            for (n, p), (_, q) in zip(sorted(ref.named_parameters()), sorted(ours.named_parameters())):
                self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-6), (mode, n))

    def test_every_mode_starts_exactly_at_the_base(self):
        x = torch.randn(2, 55, C, 3); z = torch.randn(2, 30)
        for mode in A.MODES:
            out, diag = make(mode).eval()(x, z)
            self.assertTrue(torch.equal(out, z), mode)
            self.assertTrue(torch.isfinite(diag['correction']).all())

    def test_input_validation_rejects_sentinel_and_nonfinite(self):
        a = make('a7_m1'); z = torch.randn(1, 30)
        with self.assertRaises(ValueError): a(torch.randn(1, 55, C + 1, 3), z)      # wrong channels
        with self.assertRaises(ValueError): a(torch.randn(1, 55, C), z)             # missing T axis
        bad = torch.randn(1, 55, C, 3); bad[0, 30, 0, 0] = float('nan')
        with self.assertRaises(ValueError): a(bad, z)
        with self.assertRaises(ValueError): a(torch.randn(1, 55, C, 3), torch.randn(1, 29))

    def test_c7_never_reads_other_snippets(self):
        a = live(make('c7_current_repeat')); x = torch.randn(2, 55, C, 3); z = torch.randn(2, 30); out, _ = a(x, z)
        for t in (0, 9, 29):
            y = torch.randn_like(x); y[:, 25 + t] = x[:, 25 + t]       # baseline logits fixed, current fixed
            self.assertTrue(torch.allclose(a(y, z)[0][:, t], out[:, t], atol=1e-6), t)
        y = x.clone(); y[:, 34] += 1.
        self.assertFalse(torch.allclose(a(y, z)[0][:, 9], out[:, 9]))   # its own snippet matters

    def test_exact_window_lengths_and_causality(self):
        x = torch.randn(1, 55, C, 3); z = torch.randn(1, 30)
        for mode, reach in [('r7_last_state', 7), ('a7_m1', 7), ('a26_m1', 26), ('u3_uniform', 26), ('g3_learned', 26)]:
            a = live(make(mode)); out, _ = a(x, z)
            t = 29; token = 25 + t
            inside = token - (reach - 1); outside = inside - 1
            y = x.clone(); y[:, inside] += 5.
            self.assertGreater(float((a(y, z)[0][:, t] - out[:, t]).abs().max()), 1e-7, (mode, 'inside'))
            if outside >= 0:
                y = x.clone(); y[:, outside] += 3.
                self.assertTrue(torch.allclose(a(y, z)[0][:, t], out[:, t], atol=1e-6), (mode, 'outside'))
        for mode in A.MODES:                                           # no future leakage
            a = live(make(mode)); out, _ = a(x, z)
            y = x.clone(); y[:, 40:] += torch.randn_like(y[:, 40:]) * 5
            self.assertTrue(torch.equal(a(y, z)[0][:, :15], out[:, :15]), mode)

    def test_each_window_is_a_fresh_lstm_pass(self):
        a = live(make('u3_uniform')); x = torch.randn(1, 55, C, 3)
        u = a.token_features(x)
        c7, _ = a.readout(u, 7)
        w = A.window(u, 7); b, n, l, d = w.shape
        states, _ = a.rnn(w.reshape(b * n, l, d))
        self.assertTrue(torch.allclose(c7, a.readout(u, 7)[0]))
        w26 = A.window(u, 26); s26, _ = a.rnn(w26.reshape(b * n, 26, d))
        self.assertFalse(torch.allclose(states, s26[:, -7:]), 'H7 must not be sliced from H26 hidden states')
        self.assertTrue(torch.equal(A.window(u, 7), S.window(u, 7)))

    def test_u3_mean_and_g3_zero_fusion_equals_u3(self):
        g = make('g3_learned', seed=3)
        with torch.no_grad(): g.output.weight.normal_(0, .5)
        u3 = A.AblationAdapter('u3_uniform', channels=C, width=D)
        u3.load_state_dict({k: v for k, v in g.state_dict().items() if not k.startswith('fusion.')}, strict=True)
        g.eval(); u3.eval(); x = torch.randn(2, 55, C, 3); z = torch.randn(2, 30)
        a, da = g(x, z); b, _ = u3(x, z)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.allclose(da['scale_weights'], torch.full_like(da['scale_weights'], 1 / 3)))
        u = u3.token_features(x); ctx = torch.stack([u3.readout(u, h)[0] for h in (7, 13, 26)], -2).mean(-2)
        ref = z + torch.sigmoid(u3.residual_gate(ctx)).squeeze(-1) * torch.tanh(u3.output(ctx).squeeze(-1))
        self.assertTrue(torch.allclose(b, ref, atol=1e-6))

    def test_r7_is_last_lstm_state(self):
        a = live(make('r7_last_state')); x = torch.randn(1, 55, C, 3); z = torch.randn(1, 30)
        u = a.token_features(x); w = A.window(u, 7); b, n, l, d = w.shape
        s, _ = a.rnn(w.reshape(b * n, l, d)); ctx = s[:, -1].reshape(b, n, d)
        ref = z + torch.sigmoid(a.residual_gate(ctx)).squeeze(-1) * torch.tanh(a.output(ctx).squeeze(-1))
        self.assertTrue(torch.allclose(a(x, z)[0], ref, atol=1e-6))

    def test_three_updates_output_first_then_recurrent_gradient(self):
        for mode in A.MODES:
            a = make(mode).train(); opt = torch.optim.AdamW(a.parameters(), lr=1e-2)
            x = torch.randn(2, 55, C, 3); z = torch.randn(2, 30); y = (torch.rand(2, 30) > .5).float()
            seen = []
            for step in range(3):
                opt.zero_grad(); out, _ = a(x, z)
                torch.nn.functional.binary_cross_entropy_with_logits(out, y).backward()
                seen.append(float(a.rnn.weight_hh_l0.grad.abs().sum()) if a.rnn.weight_hh_l0.grad is not None else 0.)
                if step == 0: self.assertGreater(float(a.output.weight.grad.abs().sum()), 0, mode)
                opt.step()
            self.assertEqual(seen[0], 0., mode)              # zero output blocks deep gradient at update 1
            self.assertGreater(max(seen[1:]), 0., mode)      # ... but it reaches the LSTM afterwards


if __name__ == '__main__':
    unittest.main(verbosity=2)
