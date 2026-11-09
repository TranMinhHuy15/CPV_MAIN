"""Full1344 pooled test curves; same RQ1 scalar and inferred-author-index mTTA conventions."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from rq2_model import WARMUP, frame_grid
from rq2_profile import scientific_profile, validate_metrics
from repro import sha256, write_json, run_contract

ROOT = Path(__file__).resolve().parent


def evaluation_reference():
    reference = ROOT/'rq1_evaluation'
    sys.path.insert(0,str(reference))
    import author_report
    author_report.engine.verify_package()
    # Check the outer manifest as well: metric arithmetic must not drift from RQ1.
    for name, expected in author_report.load(reference/'PACKAGE_SHA256.json').items():
        if sha256(reference/name) != expected:
            raise ValueError('RQ1 reference changed: '+name)
    return author_report


def check_budget(settings):
    if time.time() >= settings['deadline']:
        raise RuntimeError('Evaluation deadline reached. Existing curves are resumable; instance billing is still active.')


def prepare_shared_index(settings, reference):
    shared = Path(settings['workspace']).parent.parent/'shared_test_index'
    shared.mkdir(parents=True,exist_ok=True)
    reference.engine.prepare_data(settings['data_root'],shared,settings['deadline'])
    return shared


class FixedIndices:
    def __init__(self, ends, context):
        self.ends, self.context = np.asarray(ends,dtype=np.int32), context

    def __call__(self, data):
        if data['target'] is not None or data['accident_frame'] is not None or data['abnormal_start_frame'] is not None:
            raise ValueError('Test labels/anchors must not enter the model.')
        if self.context:
            indices = frame_grid(self.ends, data['total_frames'])
        else:
            indices = np.maximum(self.ends[:,None] - 3*np.arange(4,-1,-1),0).astype(np.int32)
        return dict(data,frame_inds=indices.reshape(-1),clip_len=5,num_clips=len(indices),frame_interval=3)


def decode_cache(info, resize):
    """Decode/roundtrip/resize each unique required frame once, not55 full-size duplicate images."""
    import cv2
    import mmcv
    n = info['total_frames']
    phase = (n-1)%3
    wanted = set(range(phase,n,3)) | {0}
    capture, images = cv2.VideoCapture(info['filename']), {}
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n:
            raise ValueError('Cannot decode video with exact frame count.')
        original_shape = None
        for index in range(n):
            if not capture.grab():
                raise ValueError('Video ended early.')
            if index not in wanted:
                continue
            ok, bgr = capture.retrieve()
            if not ok:
                raise ValueError('Cannot retrieve frame.')
            ok, encoded = cv2.imencode('.jpg',bgr)
            if not ok:
                raise ValueError('JPEG roundtrip failed.')
            rgb = mmcv.imfrombytes(encoded.tobytes(),channel_order='rgb',backend='cv2')
            original_shape = rgb.shape[:2]
            resized = resize(dict(imgs=[rgb],img_shape=original_shape,original_shape=original_shape))
            images[index] = resized['imgs'][0]
    finally:
        capture.release()
    return dict(imgs=images,original_shape=original_shape,img_shape=next(iter(images.values())).shape[:2])


def sample_for(info, ends, pipeline, cache=None, context=True):
    from mmengine.dataset import Compose
    transforms = copy.deepcopy(pipeline)
    transforms[0] = FixedIndices(ends,context)
    if cache is not None:
        def cached(data):
            data.update(imgs=[cache['imgs'][int(i)] for i in data['frame_inds']],
                        original_shape=cache['original_shape'],img_shape=cache['img_shape'])
            return data
        transforms[1] = cached
        del transforms[2]  # Already resized identically once per native frame.
    return Compose(transforms)(copy.deepcopy(info))


def encode_features(model, info, ends, pipeline, cache, settings):
    chunks = []
    for start in range(0,len(ends),10):
        check_budget(settings)
        sample = sample_for(info,ends[start:start+10],pipeline,cache,context=False)
        batch = model.data_preprocessor(dict(inputs=[sample['inputs']],data_samples=[sample['data_samples']]),False)
        tensor = batch['inputs'][0]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            features = model.backbone(tensor)
            if not isinstance(features,torch.Tensor):
                raise ValueError('Only the single-output V6 backbone is supported.')
            chunks.append(features.mean(dim=tuple(range(2,features.ndim))))
    return torch.cat(chunks)


def curve_from_features(model, ends, encoded_ends, features):
    positions = {int(end):i for i,end in enumerate(encoded_ends)}
    scores = []
    for start in range(0,len(ends),30):
        current = ends[start:start+30]
        prefix = np.maximum(int(current[0])-3*np.arange(WARMUP,0,-1),0)
        indices = [positions[int(end)] for end in np.concatenate([prefix,current])]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            logits = model.cls_head.score_features(features[indices][None])
            scores.extend(logits.sigmoid().float().cpu().tolist())
    return np.asarray(scores)


def technical_capture_check(model, info, ends, pipeline, cache, encoded_ends, features, values):
    # Full pipeline versus cached feature path, including a history window at video end.
    worst, pixel_equal = 0., True
    for current in (ends[:1], ends[-10:]):
        sample = sample_for(info,current,pipeline,cache,context=True)
        direct = sample_for(info,current,pipeline,None,context=True)
        pixel_equal &= torch.equal(sample['inputs'],direct['inputs'])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            expected = model.test_step(dict(inputs=[direct['inputs']],data_samples=[direct['data_samples']]))[0]
        actual = expected.pred_score.detach().float().cpu().numpy().reshape(-1)
        select = [int(np.searchsorted(ends,end)) for end in current]
        worst = max(worst,float(np.max(np.abs(actual-values[select]))))
        if not np.array_equal(expected.frame_inds,np.asarray(current)):
            raise ValueError('Capture endpoints/output alignment failed.')
    if not pixel_equal or worst > 1e-4:
        raise ValueError('Full-pipeline/cached capture parity failed: %g'%worst)
    return dict(passed=True,decoded_pixel_parity=True,max_probability_delta=worst)


def validate_curve(curve, record, fingerprint):
    n = record['total_frames']
    ends = list(range((n-1)%3,n,3))
    p = np.asarray(curve.get('scores',[]))
    if (curve.get('id') != record['id'] or curve.get('frame_inds') != ends
            or curve.get('times_s') != [record['pts'][i] for i in ends]
            or p.shape != (len(ends),) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or curve.get('capture_fingerprint') != fingerprint
            or curve.get('video_sha256') != record['video_sha256']
            or curve.get('uses_future') is not False or curve.get('coverage') != 'full_observed_clip'):
        raise ValueError('Stale/corrupt/noncausal curve: '+record['id'])


def create_report(settings, shared, directory, fingerprint, selected_checkpoint):
    reference = evaluation_reference()
    records,solution,labels,anchors = reference.dataset_contract(shared)
    curves = []
    actual_ids = {p.stem for p in (directory/'curves').glob('*.json')}
    if actual_ids != set(labels):
        raise ValueError('Final test requires exactly all1344 curves.')
    for record in records:
        curve = json.loads((directory/'curves'/(record['id']+'.json')).read_text())
        validate_curve(curve,record,fingerprint)
        curves.append(curve)
    from metrics import scalar_metrics
    predictions = [{'id':c['id'],'score':float(c['scores'][-1])} for c in curves]
    scalar = scalar_metrics(solution,predictions)
    temporal = reference.temporal(curves,labels,anchors)
    combined = scalar['Combined']
    values = {key:combined[key] for key in ('mAP','mAUC_grouped','mPAUC_sklearn_standardized','mPAUC_raw_divided_by_0.1')}
    values['mTTA_test_author_index10_inferred_s'] = temporal['values']['mtta@0.1']
    values['maximum_actual_negative_frame_FAR'] = temporal['maximum_actual_negative_frame_FAR']
    combined['mTTA'] = values['mTTA_test_author_index10_inferred_s']
    combined['mTTA_reason'] = 'Computed from full curves below; author-index10/inferred-anchor convention, not verified Table1 equivalence.'
    for group in combined['groups']:
        lead = group['lead_time_s']
        for name,source in (('AP','AP'),('AUC_full','AUC'),('pAUC_raw_div_0.1','pAUC_raw_divided_by_0.1')):
            if source in group:
                values['%s@%.1fs'%(name,lead)] = group[source]
        positive_ids = {r['id'] for r in solution if int(r['target'])==1 and int(r['group'])==int(round(lead*2-1))}
        subset_ids = positive_ids | {video_id for video_id,y in labels.items() if y==0}
        group_tta = reference.temporal([c for c in curves if c['id'] in subset_ids],
            {v:labels[v] for v in subset_ids},{v:anchors[v] for v in positive_ids})
        values['mTTA_author_index10_inferred@%.1fs'%lead] = group_tta['values']['mtta@0.1']
    for usage in ('Public','Private'):
        for key in ('mAP','mAUC_grouped','mPAUC_sklearn_standardized','mPAUC_raw_divided_by_0.1'):
            values['%s/%s'%(usage,key)] = scalar[usage][key]
        subset = {r['id'] for r in solution if r['Usage']==usage}
        subcurves = [c for c in curves if c['id'] in subset]
        temporal_subset = reference.temporal(subcurves,{v:labels[v] for v in subset},{v:a for v,a in anchors.items() if v in subset})
        values['mTTA_%s_author_index10_inferred_s'%usage] = temporal_subset['values']['mtta@0.1']
        scalar[usage]['mTTA'] = temporal_subset['values']['mtta@0.1']
        scalar[usage]['mTTA_reason'] = combined['mTTA_reason']
    validate_metrics(values)
    result = dict(complete=True,test_count=1344,seed=settings['seed'],rq2=scientific_profile(),
        selected_checkpoint=selected_checkpoint,checkpoint_sha256=fingerprint['checkpoint_sha256'],
        summary_metrics=values,scalar_metrics=scalar,temporal_metrics=temporal,
        capture_fingerprint=fingerprint,paper_equivalent=False,
        strict_frame_far_0_1_satisfied=temporal['maximum_actual_negative_frame_FAR']<=.1+1e-12,
        curve_file_sha256={c['id']:sha256(directory/'curves'/(c['id']+'.json')) for c in curves},
        notes=['Pooled1344 curve-level evaluation, NOT averaging Public/Private metrics.',
               'mAP/AUC are macro means across original lead cohorts, as in RQ1.',
               'mTTA source arithmetic unchanged; index-clock/anchors inferred, not Table1 parity.',
               'Test-negative empirical ranks describe ROC performance, not calibrated deployment alarms.',
               'All positive misses remain zero in the denominator; no prediction-tail extrapolation.'])
    work = Path(settings['workspace'])/'run'
    write_json(work/'test_predictions.json',predictions)
    write_json(work/'rq2_test_report.json',result)
    with (work/'rq2_test_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric','value','paper_equivalent'])
        writer.writerows([key,value,False] for key,value in values.items())
    return result


def capture_and_report(settings, model):
    reference = evaluation_reference()
    work = Path(settings['workspace'])/'run'
    status = json.loads((work/'training_status.json').read_text())
    if not status['complete'] or status['completed_epochs'] != 50:
        raise ValueError('Test stays locked until50 epochs.')
    if not shutil.which('ffprobe'):
        raise RuntimeError('Install ffmpeg/ffprobe BEFORE paid training.')
    # One shared data/clock audit for both arms and all seeds.
    shared = prepare_shared_index(settings,reference)
    records = reference.load(shared/'blind_index.json')
    directory = work/'test_capture'
    (directory/'curves').mkdir(parents=True,exist_ok=True)
    model.eval().requires_grad_(False)
    fingerprint = dict(checkpoint_sha256=sha256(work/status['best_checkpoint']),
        training_contract=run_contract(settings),blind_index_sha256=sha256(shared/'blind_index_contract.json'),
        torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),
        precision='FP32, no autocast',sampling='native_end_anchored_stride3_bounded_past',
        cudnn_tf32=torch.backends.cudnn.allow_tf32,capture_code_sha256=sha256(__file__))
    contract = directory/'capture_contract.json'
    if contract.exists() and reference.load(contract) != fingerprint:
        raise ValueError('Capture runtime/code/checkpoint changed; do not mix curves.')
    write_json(contract,fingerprint)
    from repro import config
    from mmengine.dataset import Compose
    source = Path(settings['workspace'])/'author_source'
    cfg = config(settings,source,Path(settings['workspace'])/'prepared_nexar',phase='evaluate')
    pipeline = cfg.val_pipeline_frame
    if [p['type'] for p in pipeline[1:]] != ['NativeJpegDecode','Resize','Flow','FormatShape','PackActionInputs']:
        raise ValueError('Unexpected transform pipeline.')
    resize = Compose([copy.deepcopy(pipeline[2])])
    torch.cuda.reset_peak_memory_stats()
    checks = []
    for index,record in enumerate(records):
        check_budget(settings)
        path = directory/'curves'/(record['id']+'.json')
        if path.exists():
            validate_curve(reference.load(path),record,fingerprint)
            continue
        info = dict(dataset='nexar',video_id=record['id'],sample_idx=index,type='frame',
            filename=record['path'],frame_dir=str(Path(record['path']).with_suffix('')),
            filename_tmpl='{:06}.jpg',start_index=0,total_frames=record['total_frames'],fps=30,
            target=None,accident_frame=None,abnormal_start_frame=None,modality='RGB',
            is_val=False,is_test=True,text='')
        ends = np.arange((record['total_frames']-1)%3,record['total_frames'],3,dtype=np.int32)
        encoded_ends = np.unique(np.concatenate([np.array([0],dtype=np.int32),ends]))
        cache = decode_cache(info,resize)
        features = encode_features(model,info,encoded_ends,pipeline,cache,settings)
        values = curve_from_features(model,ends,encoded_ends,features)
        if len(checks)<8:
            checks.append(technical_capture_check(model,info,ends,pipeline,cache,encoded_ends,features,values))
            write_json(directory/'capture_preflight.json',dict(videos=len(checks),checks=checks))
        curve = dict(id=record['id'],scores=values.tolist(),frame_inds=ends.tolist(),
            times_s=[record['pts'][int(i)] for i in ends],video_sha256=record['video_sha256'],
            observed_history_span_s=[record['pts'][int(i)]-record['pts'][max(0,int(i)-scientific_profile()['maximum_native_frame_lookback'])] for i in ends],
            coverage='full_observed_clip',uses_future=False,prediction_tail_extrapolated=False,
            capture_fingerprint=fingerprint)
        validate_curve(curve,record,fingerprint)
        write_json(path,curve)
        del cache,features
        if (index+1)%25==0:
            print('RQ2_TEST %d/1344'% (index+1),flush=True)
    report = create_report(settings,shared,directory,fingerprint,status['best_checkpoint'])
    write_json(directory/'capture_manifest.json',dict(complete=True,test_count=1344,
        checkpoint_sha256=fingerprint['checkpoint_sha256'],peak_memory_bytes=torch.cuda.max_memory_allocated()))
    from run_rq2 import seal_evaluation
    seal_evaluation(work,fingerprint['checkpoint_sha256'])
    print('FULL_TEST_REPORT',work/'rq2_test_report.json',flush=True)
    return report
