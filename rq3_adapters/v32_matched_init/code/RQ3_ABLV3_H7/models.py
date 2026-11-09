"""Adapter-only fitting on the frozen H7 of the v1.1 study, with cached frozen logits.

Ablation V2 arms (ablation_modules.py): C7 current-repeat, R7 last-state, A26, U3 uniform fusion."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

WARMUP=25


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

        def score_features(self,features):
            z0=self.baseline(features)
            if self.baseline_only:
                self.last_diagnostics={}
                return z0.reshape(-1)
            logits,diag=self.adapter(features[..., :-1].float(),z0)
            from diagnostics import detached
            self.last_diagnostics=detached(diag)
            return logits.reshape(-1)

        def forward(self,x,rq2_batch=None,rq2_tokens=None,baseline_logits=None,**kwargs):
            if baseline_logits is None:return super().forward(x,rq2_batch=rq2_batch,rq2_tokens=rq2_tokens,**kwargs)
            if self.reference_history!=7 or baseline_logits.shape!=(rq2_batch,30) or not torch.isfinite(baseline_logits).all():
                raise ValueError('Invalid cached frozen H7 logits.')
            temporal=x.mean((-2,-1)).reshape(rq2_batch,rq2_tokens,self.in_channels,x.shape[2])
            if self.baseline_only:
                self.last_diagnostics={}
                return baseline_logits.reshape(-1)
            logits,diag=self.adapter(temporal[..., :-1].float(),baseline_logits)
            from diagnostics import detached
            self.last_diagnostics=detached(diag)
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


from ablation_modules import AblationAdapter,TRAINABLE

ABLATION_ARMS=('c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned')


def build_adapter(arm,width,dropout,channels=2048,init_seed=None):
    """arm 'm1' only appears in b0/teacher contexts (the source config maps b0 -> m1); it builds the A7 (= M1) adapter."""
    mode='a7_m1' if arm=='m1' else arm
    if mode not in ABLATION_ARMS+('a7_m1',):raise ValueError('Unknown ablation arm: '+str(arm))
    from matched_initialization import build_matched_adapter
    return build_matched_adapter(mode,channels,width,dropout,seed=init_seed)


def register_ablation():
    from mmaction.registry import MODELS
    register()
    if MODELS.get('VastTunedH7AblationHead') is not None:return
    Base=MODELS.get('VastTunedH7Head')

    @MODELS.register_module()
    class VastTunedH7AblationHead(Base):
        """Frozen H7 (bounded LSTM2048 over 7 snippets, unchanged) + one V2 ablation adapter."""
        def __init__(self,arm='c7_current_repeat',width=512,adapter_dropout=.1,adapter_init_seed=None,**kwargs):
            if arm not in ABLATION_ARMS+('m1',):raise ValueError('Ablation V2 arms: '+', '.join(ABLATION_ARMS))
            super().__init__(arm='m1',width=width,adapter_dropout=adapter_dropout,**kwargs)
            self.adapter=build_adapter(arm,width,adapter_dropout,self.in_channels,init_seed=adapter_init_seed)
            self.ablation_arm=arm
