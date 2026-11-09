"""Paired test: frozen H7 (b0) and the four ablation arms share one H7 CNN pass per video."""
import copy
import gc
import json
from recovery import stage_file
import time
from pathlib import Path
from protocol import load,save,sha,digest,bind,ARMS,CANDIDATES,SEEDS,candidate
from capture_helpers import budget,feature_array,operating_point, shared_inputs_ready
TEST_COUNT=1344


def link(target,source):
    target,source=Path(target),Path(source).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve()!=source:raise ValueError('Staged artifact link changed: '+str(target))
    else:target.symlink_to(source,target_is_directory=source.is_dir())


def test_seed(cfg,root,baselines,seed,runtime):
    import torch
    import run_modules as run
    import integration
    import rq2_capture as capture
    from artifacts import load_adapter,verify_trial
    out=Path(cfg['output_root']);plan=load(out/'selection_frozen_before_test.json');lock_sha=digest(plan)
    marker=out/('test_complete_seed%d.json'%seed)
    if marker.exists():marker.replace(marker.with_name('test_complete_seed%d.previous_%d.json'%(seed,time.time_ns())))
    overall=out/'comparison/TEST_COMPARISON_COMPLETE.json'
    if overall.exists():overall.replace(overall.with_name('TEST_COMPARISON_COMPLETE.previous_%d.json'%time.time_ns()))
    if plan.get('seeds')!=list(SEEDS) or plan.get('study_contract_sha256')!=sha(out/'study_contract.json'):
        raise ValueError('Freeze all common hyperparameters/seeds before test.')
    if plan.get('runtime_identity_sha256')!=sha(out/'runtime_identity.json'):raise ValueError('Runtime changed after freeze.')
    base=baselines['h7'][str(seed)]
    if sha(base['checkpoint'])!=base['checkpoint_sha256']:raise ValueError('H7 base changed after freeze.')
    inp=dict(out_root=str(out/'test/runs'),data_root=cfg['data_root'],encode_chunk=cfg['encode_chunk'])
    shared=out/'extraction/shared'
    if not shared_inputs_ready(shared):
        from cached_driver import prepare_shared
        # Extraction may have been skipped because CPV_1 caches were supplied.
        prepare_shared(dict(inp,out_root=str(out/'extraction')),float('1e300'))
    settings={};frozen={};workspace_by_role={}
    for arm in ARMS:
        selection=plan['chosen'][arm]
        if {r['seed'] for r in selection['selected']}!=set(SEEDS) or {r['candidate'] for r in selection['selected']}!={selection['candidate']}:
            raise ValueError('Each module must use the SAME chosen candidate for all3seeds.')
        row=next(r for r in selection['selected'] if r['seed']==seed)
        source=Path(row['workspace']);verify_trial(source,row)
        s=run.settings_for(inp,{str(seed):base},shared,runtime,arm,seed,float('1e300'))
        s['hyperparameters']=candidate(selection['candidate']);settings[arm]=s
        work=Path(s['workspace'])/'run';work.mkdir(exist_ok=True);workspace_by_role[arm]=work
        contract=load(source/'run/run_manifest.json')['contract']
        bind(work/'run_manifest.json',dict(contract=contract,original_selected_full_checkpoint_sha256=row['checkpoint_sha256'],
             checkpoint_representation='selected adapter NPZ plus original frozen H7 weights'))
        bind(work/'training_status.json',dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint='selected_adapter.npz'))
        target=work/'validation_history.jsonl'
        stage_file(source / 'run/validation_history.jsonl', target)
        link(work/'selected_adapter.npz',source/'selected_adapter.npz')
        frozen[arm]=dict(checkpoint_sha256=row['adapter_sha256'],h7_checkpoint_sha256=base['checkpoint_sha256'],
                         validation_threshold=row['validation_threshold'])
    s=run.settings_for(inp,{str(seed):base},shared,runtime,'b0',seed,float('1e300'))
    s.update(reference_history=7,validation_threshold=base['validation_threshold'],hyperparameters=candidate(CANDIDATES[0]['id']))
    b0_settings=s
    work=Path(s['workspace'])/'run';work.mkdir(exist_ok=True)
    bind(work/'run_manifest.json',dict(contract=dict(arm='b0',seed=seed,base_model='H7 frozen',original_contract_sha256=base['contract_sha256'],
         checkpoint_sha256=base['checkpoint_sha256'],study_contract_sha256=sha(out/'study_contract.json')),baseline_reused=True))
    bind(work/'training_status.json',dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint=Path(base['checkpoint']).name))
    link(work/Path(base['checkpoint']).name,base['checkpoint'])
    primary=ARMS[0]
    runner=run.build(settings[primary],'evaluate');run.load_weights(runner,base['checkpoint'])
    model=runner.model
    model.cls_head.adapter=load_adapter(next(r['workspace'] for r in plan['chosen'][primary]['selected'] if r['seed']==seed),'cuda')
    model.eval().requires_grad_(False)
    extra={arm:dict(settings=settings[arm],frozen=frozen[arm],
                   adapter=load_adapter(next(r['workspace'] for r in plan['chosen'][arm]['selected'] if r['seed']==seed),'cuda')) for arm in ARMS[1:]}
    reference=capture.evaluation_reference();index=out/'test/shared_test_index';index.mkdir(parents=True,exist_ok=True)
    reference.engine.prepare_data(cfg['data_root'],index,float('1e300'))
    for arm in ARMS:print('FREEZE module=%s seed=%d shared_candidate=%s hyperparameters=%s'%(arm,seed,plan['chosen'][arm]['candidate'],plan['chosen'][arm]['hyperparameters']),flush=True)
    paired_capture(settings[primary],b0_settings,model,index,lock_sha,frozen[primary],
                   dict(job_root=str(out),export_test_features=False,test_feature_input_root=None),extra)
    save(out/('test_complete_seed%d.json'%seed),dict(complete=True,seed=seed,selection_lock_sha256=lock_sha,roles=['b0',*ARMS]))
    del runner,model,extra;gc.collect();torch.cuda.empty_cache()

def paired_capture(settings, baseline_settings, model, shared, lock_sha, frozen, cfg, extra=None):
    import numpy as np
    import torch
    import integration
    import rq2_capture as capture
    from mmengine.dataset import Compose
    from run_rq2 import seal_evaluation
    reference = capture.evaluation_reference()
    records = load(shared / 'blind_index.json')
    if len(records) != TEST_COUNT or len({r['id'] for r in records}) != TEST_COUNT:
        raise ValueError('Need complete1344 blind test records.')
    pipeline = integration.config(settings, Path(settings['workspace']) / 'author_source',
                                  Path(settings['workspace']) / 'prepared_nexar', 'evaluate').val_pipeline_frame
    if [p['type'] for p in pipeline[1:]] != ['NativeJpegDecode','Resize','Flow','FormatShape','PackActionInputs']:
        raise ValueError('Test preprocessing differs from the original audited pipeline.')
    resize = Compose([copy.deepcopy(pipeline[2])])
    extra=extra or {}
    contexts={'module':settings,'b0':baseline_settings}
    contexts.update({role:item['settings'] for role,item in extra.items()})
    runs={role:Path(current['workspace'])/'run' for role,current in contexts.items()}
    primary_adapter=model.cls_head.adapter if extra else None
    fingerprints = {}
    for role, run_dir in runs.items():
        contract = load(run_dir / 'run_manifest.json')['contract']
        fingerprints[role] = dict(checkpoint_sha256=sha(run_dir / load(run_dir / 'training_status.json')['best_checkpoint']),
            training_contract=contract, blind_index_sha256=sha(shared / 'blind_index_contract.json'),
            torch=torch.__version__, cuda=torch.version.cuda, GPU=torch.cuda.get_device_name(),
            precision='FP32, no autocast', sampling='native_end_anchored_stride3_bounded_past',
            cudnn_tf32=torch.backends.cudnn.allow_tf32, capture_code_sha256=sha(__file__), selection_lock_sha256=lock_sha)
        (run_dir / 'test_capture/curves').mkdir(parents=True, exist_ok=True)
        bind(run_dir / 'test_capture/capture_contract.json', fingerprints[role])
    if (fingerprints['module']['checkpoint_sha256'] != frozen['checkpoint_sha256']
            or fingerprints['b0']['checkpoint_sha256'] != frozen['h7_checkpoint_sha256']):
        raise ValueError('Checkpoint bytes differ from the pre-test freeze lock.')
    for role,item in extra.items():
        if fingerprints[role]['checkpoint_sha256']!=item['frozen']['checkpoint_sha256']:
            raise ValueError('Extra arm changed after six-run freeze.')
    commits = Path(settings['workspace']) / 'paired_commits'
    commits.mkdir(exist_ok=True)
    elapsed, new = 0., 0
    for index, record in enumerate(records):
        budget(settings['deadline'], 60)
        if digest(load(Path(cfg['job_root'])/'selection_frozen_before_test.json')) != lock_sha:
            raise ValueError('Pre-test freeze lock changed during capture.')
        paths = {role: run_dir / 'test_capture/curves' / (record['id'] + '.json') for role, run_dir in runs.items()}
        marker = commits / (record['id'] + '.json')
        if marker.exists():
            saved = load(marker)
            if saved.get('lock_sha256') != lock_sha:
                raise ValueError('Wrong paired commit selection lock.')
            if set(saved.get('files',{}))!=set(runs):
                raise ValueError('Paired commit has a different arm matrix.')
            for role, path in paths.items():
                if sha(path) != saved.get('files', {}).get(role):
                    raise ValueError('Committed paired curve changed: ' + str(path))
                capture.validate_curve(load(path), record, fingerprints[role])
            continue
        started = time.monotonic()
        info = dict(dataset='nexar', video_id=record['id'], sample_idx=index, type='frame', filename=record['path'],
            frame_dir=str(Path(record['path']).with_suffix('')), filename_tmpl='{:06}.jpg', start_index=0,
            total_frames=record['total_frames'], fps=30, target=None, accident_frame=None,
            abnormal_start_frame=None, modality='RGB', is_val=False, is_test=True, text='')
        ends = np.arange((record['total_frames'] - 1) % 3, record['total_frames'], 3, dtype=np.int32)
        encoded_ends = np.unique(np.concatenate([np.array([0], dtype=np.int32), ends]))
        identity = dict(seed=settings['seed'], id=record['id'], video_sha256=record['video_sha256'],
            encoded_ends=encoded_ends.tolist(), checkpoint_sha256=settings['h7_checkpoint_sha256'],
            blind_index_sha256=fingerprints['module']['blind_index_sha256'],
            preprocessing='original native JPEG/Resize/Flow/FormatShape', encoder_source_sha256=sha(Path(integration.ROOT) / 'plugin.py'),
            source_package_sha256=sha(Path(integration.ROOT)/'PACKAGE_SHA256.json'),
            pipeline=json.loads(json.dumps(pipeline)),
            runtime={k:fingerprints['module'][k] for k in ('torch','cuda','GPU','precision','cudnn_tf32')})
        pixel_cache = [None]
        def decode():
            if pixel_cache[0] is None:
                pixel_cache[0] = capture.decode_cache(info, resize)
            return pixel_cache[0]
        def encode():
            return integration.encode_temporal(model, info, encoded_ends, pipeline, decode(), settings).detach().cpu().numpy()
        local = Path(cfg['job_root']) / 'test_features' / ('seed_%d' % settings['seed']) / (record['id'] + '.npz')
        borrowed_root = cfg.get('test_feature_input_root')
        borrowed = Path(borrowed_root) / ('seed_%d' % settings['seed']) / local.name if borrowed_root else None
        array = feature_array(local, borrowed, identity, encode, cfg['export_test_features'])
        features = torch.from_numpy(array).to(device=next(model.parameters()).device)
        checks = {}
        for role in ('b0', 'module', *extra):
            active=model
            active_features=features
            active.cls_head.baseline_only = role=='b0'
            if role!='b0':
                active.cls_head.adapter=primary_adapter if role=='module' else extra[role]['adapter']
            values = capture.curve_from_features(active, ends, encoded_ends, active_features)
            if index < 8:
                checks[role] = capture.technical_capture_check(active, info, ends, pipeline, decode(), encoded_ends, active_features, values)
            curve = dict(id=record['id'], scores=values.tolist(), frame_inds=ends.tolist(),
                times_s=[record['pts'][int(i)] for i in ends], video_sha256=record['video_sha256'],
                coverage='full_observed_clip', uses_future=False, prediction_tail_extrapolated=False,
                capture_fingerprint=fingerprints[role])
            capture.validate_curve(curve, record, fingerprints[role])
            save(paths[role], curve)
        save(marker, dict(lock_sha256=lock_sha, files={role:sha(path) for role,path in paths.items()}, technical_checks=checks))
        del pixel_cache, features, array
        new += 1; elapsed += time.monotonic() - started
        if new == 1 or (index + 1) % 10 == 0:
            speed = elapsed / new
            save(Path(cfg['job_root'])/'test'/('timing_seed%d.json'%settings['seed']),
                 dict(completed_paired_videos=index+1,total=TEST_COUNT,seconds_per_paired_video=speed,
                      remaining_hours=(TEST_COUNT-index-1)*speed/3600,scope='one frozen H7 CNN pass shared by b0 and the four ablation heads, same decoded pixels'))
            print('TEST seed%d %d/%d | %.2fs/video | ETA %.2fh | H7+%s heads share the frozen H7 CNN' %
                  (settings['seed'], index + 1, TEST_COUNT, speed, (TEST_COUNT-index-1)*speed/3600, '+'.join([settings['arm']]+list(extra))), flush=True)
    ids = {r['id'] for r in records}
    if {p.stem for p in commits.glob('*.json')} != ids:
        raise ValueError('Missing/extra paired commit IDs.')
    for r in records[:8]:
        check = load(commits / (r['id'] + '.json'))['technical_checks']
        if set(check) != set(runs) or any(v.get('passed') is not True for v in check.values()):
            raise ValueError('Need eight-video parity for EVERY path before publishing test.')
    # First scalar/temporal scoring occurs only after ALL paired curves are committed.
    _, _, labels, _ = reference.dataset_contract(shared)
    for role, run_dir in runs.items():
        current = contexts[role]
        status = load(run_dir / 'training_status.json')
        from rq2_profile import scientific_profile
        profile=scientific_profile('medium_h7')
        profile['base_model']='H7 frozen';profile['baseline_history_tokens']=profile['history_tokens']
        # C7 sees only the current snippet (repeated); R7 the real H7 window; A26/U3 up to 26 snippets.
        profile['ablation_arm']=current['arm']
        from ablation_modules import HISTORIES,PAST_SNIPPETS
        arm=current['arm']
        profile['adapter_window_lengths']=[] if arm=='b0' else list(HISTORIES[arm])
        profile['adapter_repeat_current']=arm=='c7_current_repeat'
        profile['adapter_past_snippets_seen']=0 if arm=='b0' else PAST_SNIPPETS[arm]
        profile['adapter_histories']=[] if arm=='b0' else list(HISTORIES[arm])
        profile['maximum_history_tokens']=max(profile['history_tokens'],1+profile['adapter_past_snippets_seen'])
        profile['nominal_frame_slots']=profile['maximum_history_tokens']+4
        profile['maximum_native_frame_lookback']=12+3*(profile['maximum_history_tokens']-1)
        previous_profile=capture.scientific_profile
        capture.scientific_profile=lambda:dict(profile)
        try:
            result=capture.create_report(current,shared,run_dir/'test_capture',fingerprints[role],status['best_checkpoint'])
        finally:
            capture.scientific_profile=previous_profile
        curves = [load(run_dir / 'test_capture/curves' / (r['id'] + '.json')) for r in records]
        threshold = (frozen['validation_threshold'] if role=='module' else
                     extra[role]['frozen']['validation_threshold'] if role in extra else current['validation_threshold'])
        save(run_dir / 'validation_threshold_operating_point.json', operating_point(curves, labels, threshold))
        save(run_dir / 'test_capture/capture_manifest.json', dict(complete=True,test_count=1344,
             checkpoint_sha256=fingerprints[role]['checkpoint_sha256'], selection_lock_sha256=lock_sha))
        seal_evaluation(run_dir, fingerprints[role]['checkpoint_sha256'])
        m = result['summary_metrics']
        label=settings['arm'] if role=='module' else 'H7' if role=='b0' else role
        print('RESULT TEST seed=%d model=%s mAP=%.4f%% mAUC=%.4f%% AP0.5=%.4f%% AP1.0=%.4f%% AP1.5=%.4f%%' %
              (settings['seed'],label,100*m['mAP'],100*m['mAUC_grouped'],100*m['AP@0.5s'],100*m['AP@1.0s'],100*m['AP@1.5s']),flush=True)
