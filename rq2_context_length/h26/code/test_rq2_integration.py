"""Integration checks against the REAL author source; fixtures are not research scores."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--gpu',action='store_true')
parser.add_argument('--pretrained',help='Optional trusted official Kinetics710 file for local GPU verification.')
parser.add_argument('--report')
parser.add_argument('--eval-size',type=int,choices=(32,224),default=32,help='Local codec/capture inference resolution; training fixture stays32px.')
args = parser.parse_args()
SOURCE = Path(args.source).resolve()
sys.path.insert(0,str(SOURCE))

import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from repro import config
from rq2_model import register, frame_grid
from rq2_profile import scientific_profile

init_default_scope('mmaction')
import taa
import kaggle_adapter
import amp_adapter
register()
from mmaction.registry import MODELS, TRANSFORMS
from mmaction.structures import ActionDataSample
from taa.models import AnticipationHead


def settings(root):
    return dict(workspace=str(root),seed=42,world_size=1,batch_per_gpu=2,effective_batch_size=16,
                workers_per_gpu=0,epochs=50,amp=True,deadline=time.time()+3600)


class SourceIntegration(unittest.TestCase):
    def test_sampler_retains_exact_original_target_grid(self):
        from taa.transforms import SampleFramesBeforeAccident
        for target in (True,False,None):
            original = SampleFramesBeforeAccident(clip_len=5,num_clips=30,test_mode=True)
            extended = TRANSFORMS.build(dict(type='RQ2SampleFrames',clip_len=5,num_clips=30,test_mode=True))
            info = dict(total_frames=1200,fps=30,start_index=0,target=target,accident_frame=600 if target is True else None)
            a = original.transform(dict(info))['frame_inds'].reshape(30,5)
            b = extended.transform(dict(info))['frame_inds'].reshape(55,5)
            np.testing.assert_array_equal(a,b[-30:])
            self.assertTrue(np.all(b <= b[:,-1:]))

    def test_config_reaches_all_dataset_pipelines(self):
        cfg = config(settings(Path(tempfile.gettempdir())/'rq2_config_fixture'),SOURCE,Path('fixture_data'))
        for loader in (cfg.train_dataloader,cfg.val_dataloader,cfg.test_dataloader):
            for name in ('pipeline_frame','pipeline_video'):
                self.assertEqual(loader.dataset[name][0]['type'],'RQ2SampleFrames')
        self.assertEqual(cfg.model.type,'RQ2Recognizer')
        self.assertEqual(cfg.model.cls_head.history,scientific_profile()['history_tokens'])
        self.assertEqual(cfg.optim_wrapper.accumulative_counts,8)
        self.assertEqual(cfg.train_dataloader.batch_size,2)

    def test_same_parameter_count_and_original_losses(self):
        heads = [MODELS.build(dict(type='RQ2Head',history=h,in_channels=4,label_with='constraint')) for h in (1,26)]
        self.assertEqual(sum(p.numel() for p in heads[0].parameters()),sum(p.numel() for p in heads[1].parameters()))
        original = AnticipationHead(in_channels=4,label_with='constraint',clip_len=5,num_clips=30)
        logits = torch.randn(60,requires_grad=True)
        samples = [SimpleNamespace(target=True),SimpleNamespace(target=False)]
        random.seed(18)
        expected = original.loss_by_feat(logits,samples)
        random.seed(18)
        actual = heads[1].loss_by_feat(logits,samples)
        self.assertEqual(set(expected),set(actual))
        for key in expected:
            torch.testing.assert_close(expected[key],actual[key])

    def test_prediction_warmup_alignment_variable_targets(self):
        head = MODELS.build(dict(type='RQ2Head',history=26,in_channels=4,label_with='constraint'))
        for count in (1,10,30):
            ends = np.arange(100,100+count*3,3)
            sample = ActionDataSample()
            sample.frame_inds = frame_grid(ends,1200).reshape(-1)
            sample.clip_len = 5
            x = torch.randn(25+count,4,1,1,1)
            logits = head(x,rq2_batch=1,rq2_tokens=25+count)
            result = head.predict_by_feat(logits,[sample])[0]
            self.assertEqual(result.pred_score.numel(),count)
            np.testing.assert_array_equal(result.frame_inds,ends)


class RealBackboneGPU(unittest.TestCase):
    def test_real_slowonly_backward_checkpoint_roundtrip_and_capture(self):
        from rq2_capture import curve_from_features
        if not torch.cuda.is_available():
            self.fail('--gpu requested but CUDA is unavailable.')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cfg = config(settings(Path(tempfile.gettempdir())/'rq2_gpu_fixture'),SOURCE,Path('fixture_data'))
        cfg.model.backbone.pretrained = None
        model = MODELS.build(cfg.model).cuda()
        model.init_weights()
        if args.pretrained:
            from mmengine.runner.checkpoint import load_checkpoint
            loaded = load_checkpoint(model,args.pretrained,map_location='cpu',strict=False)
            torch.testing.assert_close(model.backbone.conv1.conv.weight.detach().cpu(),loaded['state_dict']['backbone.conv1.conv.weight'])
        model.train()
        optimizer = torch.optim.SGD(model.parameters(),lr=.002,momentum=.9)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
        inputs = torch.randn(2,55,3,5,32,32,device='cuda')
        samples = [SimpleNamespace(target=True),SimpleNamespace(target=False)]
        before = model.cls_head.rnn.weight_ih_l0.detach().clone()
        with torch.cuda.amp.autocast():
            losses = model.loss(inputs,samples)
            total = sum(losses.values())
        self.assertTrue(torch.isfinite(total).item())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        self.assertGreater(model.backbone.conv1.conv.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.cls_head.rnn.weight_ih_l0.grad.abs().sum().item(),0)
        torch.nn.utils.clip_grad_norm_(model.parameters(),40.)
        scaler.step(optimizer)
        scaler.update()
        self.assertFalse(torch.equal(before,model.cls_head.rnn.weight_ih_l0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.pth'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict()),path)
            saved = torch.load(path,map_location='cpu')
            model.load_state_dict(saved['model'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
        model.eval()
        with torch.no_grad():
            first = model.extract_feat(inputs[:1],stage='head',test_mode=True)[0]
            changed = inputs[:1].clone()
            changed[:,50:] += 30
            replay = model.extract_feat(changed,stage='head',test_mode=True)[0]
            torch.testing.assert_close(first[:20],replay[:20],atol=1e-5,rtol=1e-5)
            features = torch.randn(1,80,2048,device='cuda')
            ends = np.arange(0,80*3,3,dtype=np.int32)
            values = curve_from_features(model,ends,ends,features[0])
            self.assertEqual(len(values),80)
            self.assertTrue(np.isfinite(values).all())
        # Real codec -> JPEG -> resize -> normalizer -> backbone -> bounded head parity.
        from rq2_capture import decode_cache, encode_features, technical_capture_check
        from mmengine.dataset import Compose
        import cv2
        model.zero_grad(set_to_none=True)
        del optimizer,scaler,inputs,before
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'synthetic_codec_fixture.mp4'
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(64,48))
            self.assertTrue(writer.isOpened())
            for i in range(65):
                image = np.zeros((48,64,3),dtype=np.uint8)
                image[:,:,0] = i*3
                image[:,i%48:i%48+8,1] = 240
                writer.write(image)
            writer.release()
            pipeline = copy.deepcopy(cfg.val_pipeline_frame)
            pipeline[2]['scale'] = (args.eval_size,args.eval_size)
            info = dict(dataset='nexar',video_id='fixture',sample_idx=0,type='frame',filename=str(video),
                frame_dir=str(video.with_suffix('')),filename_tmpl='{:06}.jpg',start_index=0,total_frames=65,
                fps=30,target=None,accident_frame=None,abnormal_start_frame=None,modality='RGB',
                is_val=False,is_test=True,text='')
            ends = np.arange(1,65,3,dtype=np.int32)
            encoded_ends = np.unique(np.concatenate([np.array([0],dtype=np.int32),ends]))
            cache = decode_cache(info,Compose([pipeline[2]]))
            features = encode_features(model,info,encoded_ends,pipeline,cache,dict(deadline=time.time()+600))
            values = curve_from_features(model,ends,encoded_ends,features)
            checked = technical_capture_check(model,info,ends,pipeline,cache,encoded_ends,features,values)
            self.assertTrue(checked['passed'])
        self.__class__.gpu_report = dict(gpu=torch.cuda.get_device_name(),fixture_input=[2,55,3,5,32,32],
            loss=float(total.detach()),peak_memory_bytes=torch.cuda.max_memory_allocated(),
            actual_backbone_gradient=True,actual_lstm_update=True,checkpoint_roundtrip=True,
            production224_verified=False,pretrained_checkpoint_loaded=bool(args.pretrained))
        self.__class__.gpu_report['codec_capture_parity'] = checked
        self.__class__.gpu_report['codec_inference_resolution'] = args.eval_size
        self.__class__.gpu_report['full224_inference_verified'] = args.eval_size==224


if __name__=='__main__':
    suites = [unittest.defaultTestLoader.loadTestsFromTestCase(SourceIntegration)]
    if args.gpu:
        suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(RealBackboneGPU))
    suite = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,
            skipped=len(result.skipped),gpu=getattr(RealBackboneGPU,'gpu_report',None)),indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
