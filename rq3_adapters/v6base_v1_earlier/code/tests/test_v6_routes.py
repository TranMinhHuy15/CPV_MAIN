"""CPU checks for the frozen-V6 head, the M1+DX module and the pre-training validation parity gate."""
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

    def head(self,arm='m1',seed=0):
        torch.manual_seed(seed)
        with patch.dict(sys.modules,self.module_map):return self.Head(arm=arm,width=4,adapter_dropout=.1,in_channels=8)

    def test_v6_head_has_no_recurrence_and_builds_three_modules(self):
        for arm in ('m1','m1_m2','m1_dx'):
            h=self.head(arm)
            self.assertFalse(hasattr(h,'rnn'));self.assertFalse(any(n.startswith('rnn.') for n,_ in h.named_parameters()))
            self.assertEqual(h.v6_arm,arm);self.assertEqual(h.adapter.arm,arm)
            h.train();self.assertTrue(h.adapter.training);self.assertFalse(h.fc_cls.training);self.assertFalse(h.dropout.training)
            self.assertEqual(h.adapter.dropout.p,.1)
            trainable={n.split('.')[0] for n,v in h.named_parameters() if v.requires_grad}
            self.assertEqual(trainable,{'adapter'})
        with self.assertRaises(ValueError):self.head('m1_m3')

    def test_baseline_equals_author_v6_head_math(self):
        """V6 AnticipationHead: avg_pool2d(H,W) -> mean(T) -> dropout(eval) -> fc. Our cache keeps the all-axis mean."""
        h=self.head('m1');h.eval()
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
        for arm in ('m1','m1_m2','m1_dx'):
            h=self.head(arm);h.eval()
            z0=h.baseline(f)
            self.assertTrue(torch.equal(h.score_features(f),z0.reshape(-1)),arm)

    def test_teacher_cache_path_matches_and_only_adapter_gets_gradients(self):
        for arm in ('m1','m1_m2','m1_dx'):
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
            grads={n for n,v in original.adapter.named_parameters() if v.grad is not None and v.grad.abs().sum()>0}
            self.assertIn('rnn.weight_hh_l0',grads)
            if arm=='m1_dx':self.assertTrue({'diff_scale','diff_net.0.weight'}<=grads)

    def test_dx_is_causal_and_uses_lags_1_3_6(self):
        h=self.head('m1_dx');h.eval()
        with torch.no_grad():
            h.adapter.output.weight.normal_(0,.5);h.adapter.diff_net[0].weight.normal_(0,.5)
        f=torch.randn(1,55,8,4);out=h.score_features(f)
        g=f.clone();g[:,40:]+=torch.randn_like(g[:,40:])*5          # change tokens >= 40 (target index >= 15)
        changed=h.score_features(g)
        self.assertTrue(torch.equal(out[:15],changed[:15]),'earlier outputs must not see future tokens')
        self.assertFalse(torch.equal(out[15:],changed[15:]))
        self.assertEqual(models.DX_LAGS,(1,3,6))
        u=torch.randn(1,10,4);d=h.adapter.diff_net[0].in_features
        self.assertEqual(d,3*4)

    def test_adapter_export_roundtrip_for_every_arm(self):
        for arm in ('m1','m1_m2','m1_dx'):
            h=self.head(arm,seed=3)
            state={k:v.detach().clone() for k,v in h.adapter.state_dict().items()}
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


class BootstrapControl(unittest.TestCase):
    def test_smooth7_is_causal_mean_of_last_points(self):
        import bootstrap_paired as b
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            d=Path(tmp)
            p.save(d/'a.json',dict(scores=list(range(10))));p.save(d/'b.json',dict(scores=[.5,.7]))
            self.assertTrue(np.allclose(b.smoothed_scores(d,['a','b']),[np.mean(range(3,10)),.6]))

    def test_paired_bootstrap_detects_a_real_shift_and_not_a_null(self):
        import bootstrap_paired as b
        rng=np.random.default_rng(1);n=600
        labels=np.tile([0,1],n//2);groups=np.repeat([0,1,2],n//3)
        base=rng.normal(size=n)+labels*.8
        better=base+labels*1.0
        point,draws=b.paired(labels,groups,better,base,np.random.default_rng(2),300)
        s=b.summary(point,draws);self.assertGreater(s['AP@1.5s']['ci95_low_pp'],0)
        point,draws=b.paired(labels,groups,base,base,np.random.default_rng(2),100)
        s=b.summary(point,draws);self.assertEqual(s['mAP']['delta_pp'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
