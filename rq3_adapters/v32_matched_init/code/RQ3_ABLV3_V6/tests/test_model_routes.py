"""Real CPU tensor/gradient equivalence with a bounded registry fixture, no MMCV install."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np  # Initialize the existing Windows NumPy runtime before Torch; no duplicate-OpenMP override.
import torch
from torch import nn
import models


def local_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ModelRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        reference=ROOT.parent/'RQ3_VAL_TEST_42_43_v1/reference_source'
        # Tests in release ZIP use a code-only copy of these two original definitions.
        module_path=ROOT/'tests/reference_modules.py' if (ROOT/'tests/reference_modules.py').is_file() else reference/'modules.py'
        rq2_path=ROOT/'tests/reference_rq2_model.py' if (ROOT/'tests/reference_rq2_model.py').is_file() else reference/'base_h7/rq2_model.py'
        original=local_module('owned_module_fixture',module_path)
        rq2=local_module('owned_rq2_fixture',rq2_path)
        cls.bounded=staticmethod(rq2.bounded_lstm)
        class Registry:
            def __init__(self):self.values={}
            def get(self,name):return self.values.get(name)
            def register_module(self):
                def register(value):self.values[value.__name__]=value;return value
                return register
        registry=Registry();datasets=Registry()
        class BaseHead(nn.Module):
            def __init__(self,arm='m1',width=4,channels=8,**kwargs):
                super().__init__();self.in_channels=channels
                self.rnn=nn.LSTM(channels,channels,batch_first=True).requires_grad_(False)
                self.fc_cls=nn.Linear(channels,1).requires_grad_(False)
                self.adapter=original.ThreeModules(arm=arm,channels=channels,width=width)
                self.baseline_only=False;self.last_diagnostics={}
            def forward(self,x,rq2_batch=None,rq2_tokens=None,**kwargs):
                f=x.mean((-2,-1)).reshape(rq2_batch,rq2_tokens,self.in_channels,x.shape[2])
                z=self.baseline(f)
                if self.baseline_only:return z.reshape(-1)
                return self.adapter(f[..., :-1],z)[0].reshape(-1)
        registry.values.update(RQ3H7Head=BaseHead,RQ3CachedRecognizer=nn.Module)
        datasets.values['CachedFeatureDataset']=object
        class Preprocessor(nn.Module):
            def cast_data(self,value):return value
        def extract(model,features,stage):
            b,t=features.shape[:2]
            return features.reshape(-1,*features.shape[2:]),dict(rq2_batch=b,rq2_tokens=t)
        module_map={'mmaction':types.ModuleType('mmaction'),'mmaction.registry':types.SimpleNamespace(MODELS=registry,DATASETS=datasets),
                    'cached_plugin':types.SimpleNamespace(register=lambda:None,cached_extract=extract),
                    'rq2_model':rq2,'mmengine':types.ModuleType('mmengine'),'mmengine.model':types.SimpleNamespace(BaseDataPreprocessor=Preprocessor)}
        with patch.dict(sys.modules,module_map):models.register()
        cls.Head=registry.get('VastTunedH7Head');cls.Recognizer=registry.get('VastTeacherCachedRecognizer');cls.Preprocessor=registry.get('VastTeacherPreprocessor')

    def test_cached_frozen_logits_match_forward_and_adapter_gradients(self):
        torch.manual_seed(44)
        original=self.Head(arm='m1',width=4,adapter_dropout=.2)
        with torch.no_grad():original.adapter.output.weight.normal_(0,.1)
        fast=copy.deepcopy(original)
        features=torch.randn(2,55,8,3,1,1)
        temporal=features.squeeze(-1).squeeze(-1)
        # The real exporter computes one video at a time, while fitting uses microbatch2.
        frozen=torch.cat([original.baseline(temporal[i:i+1]).detach() for i in range(2)],0)
        flat=features.reshape(-1,8,3,1,1)
        original.train();fast.train()
        torch.manual_seed(99);a=original(flat,rq2_batch=2,rq2_tokens=55)
        with patch.object(fast.rnn,'forward',side_effect=AssertionError('Frozen recurrence repeated during fit')):
            torch.manual_seed(99);b=fast(flat,rq2_batch=2,rq2_tokens=55,baseline_logits=frozen)
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))
        a.square().mean().backward();b.square().mean().backward()
        for (na,pa),(nb,pb) in zip(original.named_parameters(),fast.named_parameters()):
            self.assertEqual(na,nb)
            if na.startswith('adapter.'):
                self.assertIsNotNone(pa.grad);self.assertTrue(torch.allclose(pa.grad,pb.grad,atol=1e-7,rtol=1e-6),na)
            else:self.assertIsNone(pa.grad);self.assertIsNone(pb.grad)

    def test_reference_h13_uses_actual_thirteen_past_tokens(self):
        torch.manual_seed(1);head=self.Head(arm='m1',width=4,reference_history=13)
        f=torch.randn(1,55,8,3)
        actual=head.baseline(f)
        with torch.no_grad():expected=head.fc_cls(self.bounded(head.rnn,f[..., -1],13)).squeeze(-1)
        self.assertTrue(torch.equal(actual,expected))
        with self.assertRaises(ValueError):self.Head(reference_history=16)
        with self.assertRaises(ValueError):head(torch.rand(55,8,3,1,1),rq2_batch=1,rq2_tokens=55,baseline_logits=torch.zeros(1,30))

    def test_preprocessor_accepts_pseudo_collate_dict_and_list(self):
        pre=self.Preprocessor()
        samples=[dict(features=torch.ones(55,8,3,1,1),baseline_logits=torch.zeros(30)) for _ in range(2)]
        a=pre(dict(inputs=samples));b=pre(dict(inputs={k:[r[k] for r in samples] for k in samples[0]}))
        self.assertEqual(a['inputs']['features'].shape,(2,55,8,3,1,1))
        self.assertTrue(torch.equal(a['inputs']['baseline_logits'],b['inputs']['baseline_logits']))
        with self.assertRaises(ValueError):pre(dict(inputs={'unexpected':[]}))

    def test_teacher_cache_rejects_wrong_baseline_missing_batch_parity_and_corruption(self):
        import tempfile
        import protocol as policy
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp);path=root/'logits.npz'
            with path.open('wb') as stream:np.savez(stream,logits=np.ones((1500,30),dtype=np.float32))
            manifest=dict(complete=True,cache_manifest_sha256='cache',checkpoint_sha256='baseline',array_sha256=policy.sha(path),
                          video_ids=['%05d'%i for i in range(1500)],training_microbatch2_checked=True,original_frozen_path_checked=True)
            policy.save(root/'teacher_manifest.json',manifest)
            _,values=models.checked_teacher(root,'cache','baseline');self.assertEqual(len(values),1500)
            with self.assertRaises(ValueError):models.checked_teacher(root,'cache','other-seed-baseline')
            policy.save(root/'teacher_manifest.json',dict(manifest,training_microbatch2_checked=False))
            with self.assertRaises(ValueError):models.checked_teacher(root,'cache','baseline')
            policy.save(root/'teacher_manifest.json',manifest);path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):models.checked_teacher(root,'cache','baseline')


if __name__=='__main__':unittest.main(verbosity=2)
