"""Adapter-only fitting on a frozen RiskProp V6 base with cached frozen logits.

V6 (author code, with_rnn=False) scores each snippet as fc_cls(mean CNN feature). The adapters are the
same M1/M1+M2 code used on H7 in v1.1, plus M1+DX (causal lag differences between snippets)."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

WARMUP=25
DX_LAGS=(1,3,6)


def checked_teacher(root,cache_manifest_sha,expected_checkpoint=None):
    from protocol import load,sha
    root=Path(root);m=load(root/'teacher_manifest.json')
    if (m.get('complete') is not True or m.get('cache_manifest_sha256')!=cache_manifest_sha
            or m.get('training_microbatch2_checked') is not True or m.get('original_frozen_path_checked') is not True
            or (expected_checkpoint is not None and m.get('checkpoint_sha256')!=expected_checkpoint)
            or sha(root/'logits.npz')!=m.get('array_sha256')):raise ValueError('Wrong/corrupt frozen teacher logits.')
    with np.load(root/'logits.npz',allow_pickle=False) as archive:
        if set(archive.files)!={'logits'}:raise ValueError('Unexpected teacher array fields.')
        logits=archive['logits']
    if (logits.dtype!=np.float32 or logits.shape!=(1500,30) or not np.isfinite(logits).all()
            or len(set(m['video_ids']))!=1500 or len(m['video_ids'])!=1500):raise ValueError('Teacher shape/membership mismatch.')
    return m,{video:logits[i].copy() for i,video in enumerate(m['video_ids'])}


def register():
    from mmaction.registry import MODELS,DATASETS
    if MODELS.get('VastTunedH7Head') is not None:return
    from cached_plugin import register as cached_register,cached_extract
    cached_register()
    from rq2_model import bounded_lstm
    from mmengine.model import BaseDataPreprocessor
    BaseHead=MODELS.get('RQ3H7Head');BaseRecognizer=MODELS.get('RQ3CachedRecognizer')
    BaseDataset=DATASETS.get('CachedFeatureDataset')

    @MODELS.register_module()
    class VastTunedH7Head(BaseHead):
        def __init__(self,adapter_dropout=.1,reference_history=7,**kwargs):
            if reference_history not in (7,13):raise ValueError('Only the confirmed H7/H13 comparison is supported.')
            super().__init__(**kwargs)
            self.adapter.dropout.p=adapter_dropout;self.reference_history=reference_history

        def baseline(self,temporal):
            with torch.no_grad():
                hidden=bounded_lstm(self.rnn,temporal[..., -1].float(),self.reference_history)
                return self.fc_cls(hidden).squeeze(-1)

        def forward(self,x,rq2_batch=None,rq2_tokens=None,baseline_logits=None,**kwargs):
            if baseline_logits is None:return super().forward(x,rq2_batch=rq2_batch,rq2_tokens=rq2_tokens,**kwargs)
            if self.reference_history!=7 or baseline_logits.shape!=(rq2_batch,30) or not torch.isfinite(baseline_logits).all():
                raise ValueError('Invalid cached frozen H7 logits.')
            temporal=x.mean((-2,-1)).reshape(rq2_batch,rq2_tokens,self.in_channels,x.shape[2])
            if self.baseline_only:return baseline_logits.reshape(-1)
            logits,diag=self.adapter(temporal[..., :-1].float(),baseline_logits)
            self.last_diagnostics={k:v.detach() for k,v in diag.items() if k in ('correction','raw')}
            return logits.reshape(-1)

    @MODELS.register_module()
    class VastTeacherCachedRecognizer(BaseRecognizer):
        def extract_feat(self,inputs,stage='neck',data_samples=None,test_mode=False):
            if not isinstance(inputs,dict) or set(inputs)!={'features','baseline_logits'}:raise ValueError('Require features and frozen logits.')
            x,kwargs=cached_extract(self,inputs['features'],stage='neck')
            kwargs['baseline_logits']=inputs['baseline_logits']
            return (self.cls_head(x,**kwargs) if stage=='head' else x),kwargs

    @MODELS.register_module()
    class VastTeacherPreprocessor(BaseDataPreprocessor):
        def forward(self,data,training=False):
            data=self.cast_data(data);inputs=data['inputs']
            if isinstance(inputs,(tuple,list)):
                inputs={k:[row[k] for row in inputs] for k in ('features','baseline_logits')}
            if not isinstance(inputs,dict) or set(inputs)!={'features','baseline_logits'}:raise ValueError('Unexpected teacher collate schema.')
            data['inputs']={k:(torch.stack(v) if isinstance(v,(list,tuple)) else v).float() for k,v in inputs.items()}
            return data

    @DATASETS.register_module()
    class VastTeacherDataset(BaseDataset):
        def __init__(self,teacher_root,**kwargs):
            super().__init__(**kwargs)
            from protocol import load
            cache=load(self.root/'cache_manifest.json')
            self.teacher_manifest,self.logits=checked_teacher(teacher_root,kwargs['manifest_sha256'],cache['header']['checkpoint_sha256'])
            if set(self.logits)!={r['video_id'] for r in cache['records']}:raise ValueError('Teacher/cache membership differs.')

        def __getitem__(self,index):
            item=super().__getitem__(index);video=self.records[index]['video_id']
            item['inputs']=dict(features=item['inputs'],baseline_logits=torch.from_numpy(self.logits[video].copy()))
            return item


def lag_difference_class():
    """Built lazily: the base ThreeModules lives in the hash-checked source package."""
    from modules import ThreeModules

    class LagDiffModules(ThreeModules):
        """M1 + DX. For each token t: d_k = u_t - u_(t-k), k in DX_LAGS (clamped at the first token).

        u' = u + eta * (phi([d_1,d_3,d_6]) - phi(0)); phi = Linear(3W,W) -> ReLU -> Linear(W,W) -> LayerNorm,
        eta initialised to 0.1. Only past tokens enter; the zero-initialised output keeps the start at V6."""
        def __init__(self,channels=2048,width=512,dropout=.1,bound=1.):
            super().__init__('m1',channels=channels,width=width,dropout=dropout,bound=bound)
            self.arm='m1_dx'
            self.diff_net=nn.Sequential(nn.Linear(len(DX_LAGS)*width,width),nn.ReLU(),
                                        nn.Linear(width,width),nn.LayerNorm(width))
            self.diff_scale=nn.Parameter(torch.tensor(.1))

        def token_features(self,x):
            u=super().token_features(x)
            index=torch.arange(u.shape[1],device=u.device)
            d=torch.cat([u-u[:,(index-k).clamp(min=0)] for k in DX_LAGS],-1)
            return u+self.diff_scale*(self.diff_net(d)-self.diff_net(torch.zeros_like(d)))

    return LagDiffModules


def build_adapter(arm,width,dropout,channels=2048):
    if arm=='m1_dx':return lag_difference_class()(channels=channels,width=width,dropout=dropout)
    if arm not in ('m1','m1_m2'):raise ValueError('Unknown V6-base module: '+str(arm))
    from modules import ThreeModules
    return ThreeModules(arm=arm,channels=channels,width=width,dropout=dropout)


def v6_baseline(fc_cls,temporal):
    """Frozen V6 logits for the 30 target tokens. temporal=[B,55,C,T+1]; last column is the exact
    all-axis mean, identical to V6 AnticipationHead (avg-pool H,W then mean T) up to float order."""
    if temporal.ndim!=4 or temporal.shape[1]<=WARMUP:raise ValueError('Require [B,55,C,T+1] packed features.')
    with torch.no_grad():
        return fc_cls(temporal[:,WARMUP:,:,-1].float()).squeeze(-1)


def register_v6():
    from mmaction.registry import MODELS
    register()
    if MODELS.get('VastTunedV6Head') is not None:return
    Base=MODELS.get('VastTunedH7Head')

    @MODELS.register_module()
    class VastTunedV6Head(Base):
        def __init__(self,arm='m1',width=512,adapter_dropout=.1,**kwargs):
            if arm not in ('m1','m1_m2','m1_dx'):raise ValueError('V6-base modules are m1, m1_m2, m1_dx.')
            super().__init__(arm='m1' if arm=='m1_dx' else arm,width=width,adapter_dropout=adapter_dropout,**kwargs)
            if arm=='m1_dx':self.adapter=build_adapter(arm,width,adapter_dropout,self.in_channels)
            del self.rnn  # V6 has no recurrence; its checkpoint carries no rnn weights.
            self.v6_arm=arm

        def train(self,mode=True):
            nn.Module.train(self,mode)
            self.fc_cls.eval();self.dropout.eval()  # frozen V6 path stays deterministic
            return self

        def baseline(self,temporal):
            return v6_baseline(self.fc_cls,temporal)
