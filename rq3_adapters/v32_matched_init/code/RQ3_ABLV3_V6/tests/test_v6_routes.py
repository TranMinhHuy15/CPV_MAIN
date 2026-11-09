"""CPU checks for the frozen-V6 head with the ablation V2 arms and the pre-training validation parity gate."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import models
import protocol as p

ARMS=('c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned')


def local_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class V6Routes(unittest.TestCase):
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
        registry=Registry();datasets=Registry()
        source=cls.source

        class RQ3H7Head(nn.Module):
            """Mirrors the real RQ3H7Head: rnn+fc_cls+dropout frozen, adapter trainable, rnn.eval in train()."""
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
        with patch.dict(sys.modules,cls.module_map):models.register_v6()
        cls.Head=registry.get('VastTunedV6Head')

    def head(self,arm='c7_current_repeat',seed=0):
        torch.manual_seed(seed)
        with patch.dict(sys.modules,self.module_map):return self.Head(arm=arm,width=4,adapter_dropout=.1,in_channels=8)

    def test_v6_head_has_no_base_recurrence_and_builds_the_four_arms(self):
        for arm in ARMS:
            h=self.head(arm)
            self.assertFalse(hasattr(h,'rnn'));self.assertFalse(any(n.startswith('rnn.') for n,_ in h.named_parameters()))
            self.assertEqual(h.v6_arm,arm);self.assertEqual(h.adapter.mode,arm)
            h.train();self.assertTrue(h.adapter.training);self.assertFalse(h.fc_cls.training);self.assertFalse(h.dropout.training)
            self.assertEqual(h.adapter.dropout.p,.1)
            self.assertEqual({n.split('.')[0] for n,v in h.named_parameters() if v.requires_grad},{'adapter'})
        self.assertEqual(self.head('m1').adapter.mode,'a7_m1')   # b0/teacher context only
        for bad in ('m1_m2','diff_h14','unknown'):
            with self.assertRaises(ValueError):self.head(bad)

    def test_baseline_equals_author_v6_head_math(self):
        """V6 AnticipationHead: avg_pool2d(H,W) -> mean(T) -> dropout(eval) -> fc. Our cache keeps the all-axis mean."""
        h=self.head('c7_current_repeat');h.eval()
        fmap=torch.randn(55,8,4,3,3,dtype=torch.float64).float()          # [tokens,C,T,H,W] from the CNN
        packed=self.source.pack_encoder(fmap).squeeze(-1).squeeze(-1)[None]  # [1,55,C,T+1]
        ours=h.baseline(packed)
        x=F.adaptive_avg_pool2d(fmap.reshape(55,8*4,3,3),1).reshape(55,8,4).mean(-1)
        author=h.fc_cls(x).squeeze(-1)[25:][None]
        self.assertEqual(ours.shape,(1,30))
        self.assertTrue(torch.allclose(ours,author,atol=1e-6,rtol=1e-6))
        h.baseline_only=True
        self.assertTrue(torch.equal(h.score_features(packed),ours.reshape(-1)))

    def test_zero_initialised_modules_start_exactly_at_v6(self):
        f=torch.randn(2,55,8,4)
        for arm in ARMS:
            h=self.head(arm);h.eval()
            z0=h.baseline(f)
            self.assertTrue(torch.equal(h.score_features(f),z0.reshape(-1)),arm)

    def test_teacher_cache_path_matches_and_only_adapter_gets_gradients(self):
        for arm in ARMS:
            original=self.head(arm,seed=44)
            with torch.no_grad():original.adapter.output.weight.normal_(0,.1)
            fast=copy.deepcopy(original)
            features=torch.randn(2,55,8,3,1,1);temporal=features.squeeze(-1).squeeze(-1)
            frozen=torch.cat([original.baseline(temporal[i:i+1]).detach() for i in range(2)],0)
            flat=features.reshape(-1,8,3,1,1);original.train();fast.train()
            torch.manual_seed(9);a=original(flat,rq2_batch=2,rq2_tokens=55)
            with patch.object(fast.fc_cls,'forward',side_effect=AssertionError('frozen V6 fc repeated during fit')):
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

    def test_validation_parity_gate(self):
        import worker
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            rng=np.random.default_rng(0);ids=['%05d'%i for i in range(1500)]
            logits=rng.normal(size=(1500,30)).astype(np.float32)
            val=ids[1200:]
            p.save(run/'validation_predictions.json',[dict(id=v,target=0,pred=(1/(1+np.exp(-logits[1200+i].astype(np.float64)))).tolist())
                                                     for i,v in enumerate(val)])
            base=dict(run=str(run),seed=44)
            result=worker.validation_parity(base,ids,logits,root)
            self.assertTrue(result['passed']);self.assertLess(result['max_abs_prob_diff'],1e-6)
            bad=logits.copy();bad[1250,3]+=1.
            with self.assertRaisesRegex(ValueError,'does not reproduce'):worker.validation_parity(base,ids,bad,root)
            with self.assertRaises(ValueError):worker.validation_parity(base,ids[:1300],logits[:1300],root)


class Bootstrap(unittest.TestCase):
    def test_contrast_plan_roles(self):
        import bootstrap_paired as b
        import ablation_stats as S
        roles=set(b.NEW)|set(b.PRIOR)
        self.assertEqual(set(S.ARMS6)-roles,set());self.assertEqual(b.BASE,'v6')
        for left,right,_ in S.MAIN+tuple(b.SIDE):self.assertIn(left,roles);self.assertIn(right,roles)
        self.assertEqual(b.PRIOR,{})
        self.assertIn('a7_m1',b.NEW);self.assertIn('g3_learned',b.NEW)

    def test_real_size_parameter_counts(self):
        import ablation_modules as A
        for arm in ARMS:
            self.assertEqual(sum(v.numel() for v in models.build_adapter(arm,width=512,dropout=.1).parameters()),A.TRAINABLE[arm])


if __name__=='__main__':unittest.main(verbosity=2)
