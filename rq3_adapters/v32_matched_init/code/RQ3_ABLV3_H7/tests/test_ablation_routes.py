"""CPU checks for ablation V2 heads on the frozen H7 (C7, R7, A26, U3) and the paired statistics."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import models

ARMS=('c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned')


def local_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class AblationRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source=local_module('modules',ROOT/'tests/reference_modules.py')  # byte copy of source modules.py
        rq2=local_module('owned_rq2_fixture',ROOT/'tests/reference_rq2_model.py')
        class Registry:
            def __init__(self):self.values={}
            def get(self,name):return self.values.get(name)
            def register_module(self):
                def register(value):self.values[value.__name__]=value;return value
                return register
        registry=Registry();datasets=Registry();source=cls.source

        class RQ3H7Head(nn.Module):
            """Mirrors the real RQ3H7Head: frozen LSTM(C) + fc_cls + dropout, trainable adapter."""
            def __init__(self,arm='m1',width=4,in_channels=8,**kwargs):
                super().__init__();self.in_channels=in_channels
                self.rnn=nn.LSTM(in_channels,in_channels,batch_first=True)
                self.fc_cls=nn.Linear(in_channels,1);self.dropout=nn.Dropout(.4)
                self.adapter=source.ThreeModules(arm,channels=in_channels,width=width)
                self.rnn.requires_grad_(False);self.fc_cls.requires_grad_(False)
                self.baseline_only=False;self.last_diagnostics={}
            def train(self,mode=True):
                super().train(mode);self.rnn.eval();self.fc_cls.eval();self.dropout.eval();return self
            def score_features(self,features):
                z0=self.baseline(features)
                if self.baseline_only:return z0.reshape(-1)
                z,diag=self.adapter(features[..., :-1].float(),z0)
                self.last_diagnostics={k:v.detach() for k,v in diag.items() if k in ('correction','raw')}
                return z.reshape(-1)
            def forward(self,x,rq2_batch=None,rq2_tokens=None,**kwargs):
                temporal=x.mean((-2,-1)).reshape(rq2_batch,rq2_tokens,self.in_channels,x.shape[2])
                return self.score_features(temporal)
        registry.values.update(RQ3H7Head=RQ3H7Head,RQ3CachedRecognizer=nn.Module)
        datasets.values['CachedFeatureDataset']=object
        class Preprocessor(nn.Module):
            def cast_data(self,value):return value
        def extract(model,features,stage):
            b,t=features.shape[:2]
            return features.reshape(-1,*features.shape[2:]),dict(rq2_batch=b,rq2_tokens=t)
        cls.module_map={'modules':source,'mmaction':types.ModuleType('mmaction'),
                        'mmaction.registry':types.SimpleNamespace(MODELS=registry,DATASETS=datasets),
                        'cached_plugin':types.SimpleNamespace(register=lambda:None,cached_extract=extract),
                        'rq2_model':rq2,'mmengine':types.ModuleType('mmengine'),
                        'mmengine.model':types.SimpleNamespace(BaseDataPreprocessor=Preprocessor)}
        with patch.dict(sys.modules,cls.module_map):models.register_ablation()
        cls.Head=registry.get('VastTunedH7AblationHead')

    def head(self,arm,seed=0):
        torch.manual_seed(seed)
        with patch.dict(sys.modules,self.module_map):return self.Head(arm=arm,width=4,adapter_dropout=.1,in_channels=8)

    def test_head_freezes_h7_and_starts_exactly_at_h7(self):
        f=torch.randn(2,55,8,4)
        for arm in ARMS:
            h=self.head(arm);self.assertEqual(h.ablation_arm,arm);self.assertEqual(h.adapter.mode,arm)
            h.train();self.assertTrue(h.adapter.training);self.assertFalse(h.rnn.training)
            self.assertEqual({n.split('.')[0] for n,v in h.named_parameters() if v.requires_grad},{'adapter'})
            h.eval();self.assertTrue(torch.equal(h.score_features(f),h.baseline(f).reshape(-1)),arm)
        self.assertEqual(self.head('m1').adapter.mode,'a7_m1')     # b0/teacher context only
        for bad in ('m1_m2','b1_512','unknown'):
            with self.assertRaises(ValueError):self.head(bad)

    def test_teacher_cache_path_matches_and_only_adapter_gets_gradients(self):
        for arm in ARMS:
            original=self.head(arm,seed=44)
            with torch.no_grad():original.adapter.output.weight.normal_(0,.1)
            fast=copy.deepcopy(original)
            features=torch.randn(2,55,8,3,1,1);temporal=features.squeeze(-1).squeeze(-1)
            frozen=torch.cat([original.baseline(temporal[i:i+1]).detach() for i in range(2)],0)
            flat=features.reshape(-1,8,3,1,1);original.train();fast.train()
            torch.manual_seed(9);a=original(flat,rq2_batch=2,rq2_tokens=55)
            with patch.object(fast.rnn,'forward',side_effect=AssertionError('frozen H7 LSTM repeated during fit')):
                torch.manual_seed(9);b=fast(flat,rq2_batch=2,rq2_tokens=55,baseline_logits=frozen)
            self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6),arm)
            a.square().mean().backward();b.square().mean().backward()
            for (na,pa),(nb,pb) in zip(original.named_parameters(),fast.named_parameters()):
                if na.startswith('adapter.'):
                    if pa.grad is not None:self.assertTrue(torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-6),na)
                else:self.assertIsNone(pa.grad);self.assertIsNone(pb.grad)
            self.assertGreater(float(original.adapter.rnn.weight_hh_l0.grad.abs().sum()),0,arm)  # smoke probe target

    def test_adapter_export_roundtrip_for_every_arm(self):
        for arm in ARMS:
            h=self.head(arm,seed=3)
            state={k:v.detach().clone() for k,v in h.adapter.state_dict().items()}
            torch.manual_seed(99)
            with patch.dict(sys.modules,self.module_map):rebuilt=models.build_adapter(arm,width=4,dropout=.1,channels=8)
            rebuilt.load_state_dict(state,strict=True)
            f=torch.randn(1,55,8,4);z=torch.randn(1,30)
            h.adapter.eval();rebuilt.eval()
            self.assertTrue(torch.equal(h.adapter(f,z)[0],rebuilt(f,z)[0]),arm)

    def test_real_size_counts_and_cost_model(self):
        import ablation_modules as A
        for arm in ARMS:
            a=models.build_adapter(arm,width=512,dropout=.1)
            self.assertEqual(sum(v.numel() for v in a.parameters()),A.TRAINABLE[arm])


class Bootstrap(unittest.TestCase):
    def test_contrast_plan_roles_and_holm(self):
        import bootstrap_paired as b
        import ablation_stats as S
        roles=set(b.NEW)|set(b.PRIOR)
        self.assertEqual(set(S.ARMS6)-roles,set())
        for left,right,_ in S.MAIN+tuple(b.SIDE):self.assertIn(left,roles);self.assertIn(right,roles)
        self.assertEqual({(l,r) for l,r,_ in S.MAIN},{('a7_m1','c7_current_repeat'),('a7_m1','r7_last_state'),
            ('g3_learned','u3_uniform'),('u3_uniform','a26_m1'),('g3_learned','a26_m1')})
        self.assertEqual(b.PRIOR,{});self.assertIn('a7_m1',b.NEW);self.assertIn('g3_learned',b.NEW)
        self.assertEqual(S.holm([.01,.04,.03]),[.03,.06,.06])

    def test_shared_indices_are_identical_and_stratified(self):
        import ablation_stats as S
        labels=np.tile([0,1],300);groups=np.repeat([0,1,2],200)
        a=S.shared_indices(labels,groups,20);b=S.shared_indices(labels,groups,20)
        self.assertTrue(np.array_equal(a,b))
        for row in a:
            for k in (0,1,2):
                for t in (0,1):self.assertEqual(int(((groups[row]==k)&(labels[row]==t)).sum()),int(((groups==k)&(labels==t)).sum()))

    def test_paired_draws_detect_a_shift_and_not_a_null(self):
        import ablation_stats as S
        rng=np.random.default_rng(1);n=600
        labels=np.tile([0,1],n//2);groups=np.repeat([0,1,2],n//3)
        base=1/(1+np.exp(-(rng.normal(size=n)+labels*.8)));better=1/(1+np.exp(-(np.log(base/(1-base))+labels*1.)))
        idx=S.shared_indices(labels,groups,200)
        pb,db=S.role_draws(labels,groups,base,idx);pg,dg=S.role_draws(labels,groups,better,idx)
        s=S.summarize([pg['mAP']-pb['mAP']]*3,np.stack([dg['mAP']-db['mAP']]*3))
        self.assertGreater(s['ci95_low_pp'],0);self.assertEqual(S.verdict(s),'effect present (left > right)')
        s=S.summarize([0.]*3,np.zeros((3,200)));self.assertEqual(S.verdict(s),'no detectable effect (CI includes 0)')

    def test_prediction_loader_rejects_duplicates_range_and_missing(self):
        import ablation_stats as S
        import tempfile,json
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            f=Path(tmp)/'p.json'
            for rows in ([dict(id='a',score=.5),dict(id='a',score=.5)],[dict(id='a',score=1.5),dict(id='b',score=.1)],
                         [dict(id='a',score=float('nan')),dict(id='b',score=.1)],[dict(id='a',score=.5)]):
                f.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):S.load_scores(f,['a','b'])


if __name__=='__main__':unittest.main(verbosity=2)
