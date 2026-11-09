"""Actual PyTorch tests of medium recurrence; no synthetic metric is a research result."""
import unittest
import numpy as np
import torch
from rq2_model import bounded_lstm, frame_grid
from rq2_profile import VARIANT, VARIANTS, scientific_profile


class MediumTensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(428)

    def test_release_has_exactly_one_correct_training_arm(self):
        self.assertEqual(len(VARIANTS),1)
        self.assertEqual(VARIANTS[0],VARIANT)
        self.assertIn(scientific_profile()["history_tokens"],(7,13))
        with self.assertRaises(ValueError):
            scientific_profile("unknown")

    def test_target_window_reads_exact_history_and_no_future(self):
        x=torch.arange(55,dtype=torch.float64).reshape(1,55,1)
        class Trace:
            bidirectional=False
            def __call__(self, value):
                self.value=value.clone()
                return value,None
        for h in (7,13):
            trace=Trace()
            result=bounded_lstm(trace,x,h)
            self.assertEqual(tuple(trace.value.shape),(30,h,1))
            for target in range(30):
                expected=torch.arange(25+target-h+1,26+target,dtype=torch.float64)
                torch.testing.assert_close(trace.value[target,:,0],expected)
            torch.testing.assert_close(result[0,:,0],torch.arange(25,55,dtype=torch.float64))

    def test_only_window_features_have_gradient_for_one_score(self):
        for h in (7,13):
            model=torch.nn.LSTM(4,4,batch_first=True).double()
            x=torch.randn(1,55,4,dtype=torch.float64,requires_grad=True)
            bounded_lstm(model,x,h)[:,5].square().sum().backward()
            first=30-h+1
            self.assertEqual(float(x.grad[:,:first].abs().sum()),0.)
            self.assertEqual(float(x.grad[:,31:].abs().sum()),0.)
            self.assertGreater(float(x.grad[:,first:30].abs().sum()),0.)
            self.assertGreater(float(model.weight_hh_l0.grad.abs().sum()),0.)
            self.assertTrue(bool(torch.isfinite(model.weight_hh_l0.grad).all()))

    def test_nominal_lookback_and_target_alignment_on_short_videos(self):
        for n in (1,2,5,20,90,300):
            ends=np.minimum(3*np.arange(30),n-1).astype(np.int64)
            grid=frame_grid(ends,n)
            self.assertEqual(grid.shape,(55,5))
            np.testing.assert_array_equal(grid[25:,-1],ends)
            self.assertTrue(np.all((grid>=0)&(grid<n)))
            self.assertTrue(np.all(grid<=grid[:,-1:]))
        h=scientific_profile()["history_tokens"]
        self.assertEqual(scientific_profile()["maximum_native_frame_lookback"],12+3*(h-1))

    def test_real_2048_width_medium_forward_backward(self):
        model=torch.nn.LSTM(2048,2048,batch_first=True)
        # One prediction; actual production width, but not a CNN/GPU/full224 test.
        h=scientific_profile()["history_tokens"]
        x=torch.randn(1,26,2048,requires_grad=True)
        out=bounded_lstm(model,x,h)
        self.assertEqual(tuple(out.shape),(1,1,2048))
        out.square().mean().backward()
        self.assertGreater(float(model.weight_hh_l0.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))


if __name__=="__main__":
    unittest.main(verbosity=2)
