"""Mocked native-video I/O validates the 4 paths (V6 + 3 modules) and joint resume commits."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import protocol as p
import test_backend as entry


class PairCapture(unittest.TestCase):
    def test_one_v6_encoder_four_heads_blind_resume_corruption(self):
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp);count=8
            records=[dict(id='%05d'%i,total_frames=5,pts=[0,.1,.2,.3,.4],video_sha256='sha',path='blind.mp4') for i in range(count)]
            shared=root/'index';shared.mkdir();p.save(shared/'blind_index.json',records);p.save(shared/'blind_index_contract.json',[{k:v for k,v in r.items() if k!='path'} for r in records])
            job=root/'job';job.mkdir();lock=p.bind(job/'selection_frozen_before_test.json',dict(frozen=True))
            settings=dict(workspace=str(job/'test/runs/m1/seed_44'),seed=44,deadline=10**12,arm='m1',h7_checkpoint_sha256='')
            b0=dict(settings,workspace=str(job/'test/runs/b0/seed_44'),validation_threshold=.6,arm='b0')
            more={arm:dict(settings,workspace=str(job/('test/runs/'+arm+'/seed_44')),arm=arm) for arm in ('m1_m2','m1_dx')}
            paths={role:Path(s['workspace'])/'run' for role,s in [('module',settings),('b0',b0),*more.items()]}
            for role,path in paths.items():
                path.mkdir(parents=True);(path/'best.pth').write_bytes(role.encode());p.save(path/'training_status.json',dict(best_checkpoint='best.pth'))
                p.save(path/'run_manifest.json',dict(contract=dict(arm=role)))
            frozen=dict(checkpoint_sha256=p.sha(paths['module']/'best.pth'),h7_checkpoint_sha256=p.sha(paths['b0']/'best.pth'),validation_threshold=.6)
            settings['h7_checkpoint_sha256']=frozen['h7_checkpoint_sha256']
            def model(name):return types.SimpleNamespace(name=name,cls_head=types.SimpleNamespace(baseline_only=False,adapter=types.SimpleNamespace(name='m1')),
                                                        parameters=lambda:iter([types.SimpleNamespace(device='cpu')]))
            v6=model('V6')
            class Tensor:
                def __init__(self,a):self.a=a
                def to(self,**kwargs):return self
                def detach(self):return self
                def cpu(self):return self
                def numpy(self):return self.a
            torch=types.SimpleNamespace(__version__='fake',version=types.SimpleNamespace(cuda='fake'),
                cuda=types.SimpleNamespace(get_device_name=lambda:'fixture'),backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(allow_tf32=False)),from_numpy=Tensor)
            calls=[];reports=[];profiles=[];slots={};pixels=[];parities=[]
            def encode(model,info,ends,pipeline,decoded,settings):
                self.assertIsNone(info['target']);self.assertIsNone(info['accident_frame']);self.assertIsNone(info['abnormal_start_frame'])
                calls.append((model.name,info['video_id']));return Tensor(np.ones((len(ends),2048,3),dtype=np.float32))
            reference=types.SimpleNamespace(dataset_contract=lambda path:(None,None,{r['id']:i%2 for i,r in enumerate(records)},None))
            def report(current,shared,directory,fingerprint,checkpoint):
                self.assertEqual(len(list((Path(settings['workspace'])/'paired_commits').glob('*.json'))),count)
                role=Path(current['workspace']).parent.name;profiles.append((role,capture.scientific_profile()['maximum_native_frame_lookback']))
                slots[role]=capture.scientific_profile()['nominal_frame_slots']
                reports.append(current['workspace']);return dict(summary_metrics=dict(mAP=.8,mAUC_grouped=.8,**{'AP@0.5s':.8,'AP@1.0s':.8,'AP@1.5s':.8}))
            def probabilities(active,ends,*a):
                value=.2 if active.cls_head.baseline_only else {'m1':.4,'m1_m2':.6,'m1_dx':.8}[active.cls_head.adapter.name]
                return np.full(len(ends),value)
            def parity(active,*args):parities.append(active.name);return dict(passed=True)
            capture=types.SimpleNamespace(evaluation_reference=lambda:reference,decode_cache=lambda *a:(pixels.append(1) or {}),
                curve_from_features=probabilities,technical_capture_check=parity,
                validate_curve=lambda c,r,f:self.assertEqual(c['capture_fingerprint'],f),create_report=report)
            capture.scientific_profile=lambda:dict(maximum_native_frame_lookback=30)
            package=root/'package';package.mkdir();(package/'plugin.py').write_bytes(b'fixture');(package/'PACKAGE_SHA256.json').write_text('{}')
            pipeline=[dict(type='RQ2SampleFrames')]+[dict(type=x) for x in ('NativeJpegDecode','Resize','Flow','FormatShape','PackActionInputs')]
            integration=types.SimpleNamespace(ROOT=package,encode_temporal=encode,config=lambda *a:types.SimpleNamespace(val_pipeline_frame=pipeline))
            modules={'torch':torch,'integration':integration,'rq2_capture':capture,'mmengine':types.ModuleType('mmengine'),
                     'mmengine.dataset':types.SimpleNamespace(Compose=lambda v:v),'rq2_profile':types.SimpleNamespace(scientific_profile=lambda variant=None:dict(history_tokens=13 if variant=='medium_h13' else 7,maximum_native_frame_lookback=48 if variant=='medium_h13' else 30)),
                     'run_rq2':types.SimpleNamespace(seal_evaluation=lambda *a:None)}
            extras={role:dict(settings=s,adapter=types.SimpleNamespace(name=role),frozen=dict(validation_threshold=.6,checkpoint_sha256=p.sha(paths[role]/'best.pth'))) for role,s in more.items()}
            cfg=dict(job_root=str(job),test_feature_input_root=None,export_test_features=False)
            with patch.dict(sys.modules,modules),patch.object(entry,'TEST_COUNT',count),contextlib.redirect_stdout(io.StringIO()):
                entry.paired_capture(settings,b0,v6,shared,lock,frozen,cfg,extras)
                self.assertEqual(len(calls),count,'one shared V6 CNN pass per video');self.assertEqual(len(pixels),count)
                self.assertEqual(len(reports),4);self.assertEqual(len(parities),count*4)
                self.assertEqual(dict(profiles),{'m1':30,'b0':12,'m1_m2':87,'m1_dx':48})
                self.assertEqual(slots,{'m1':11,'b0':5,'m1_m2':30,'m1_dx':17})
                for role,value in [('module',.4),('b0',.2),('m1_m2',.6),('m1_dx',.8)]:
                    self.assertEqual(p.load(paths[role]/'test_capture/curves/00000.json')['scores'],[value,value])
                entry.paired_capture(settings,b0,v6,shared,lock,frozen,cfg,extras)
                self.assertEqual(len(calls),count,'completed joint commits must not re-encode the CNN')
                (paths['m1_dx']/'test_capture/curves/00000.json').write_text('{}')
                with self.assertRaises(ValueError):entry.paired_capture(settings,b0,v6,shared,lock,frozen,cfg,extras)


if __name__=='__main__':unittest.main(verbosity=2)
