"""Verified same-seed comparisons. No test metric is fed back to selection."""
from pathlib import Path
import math
import statistics
import time
from protocol import load,save,sha,digest,csv_write,SEEDS,ARMS,CANDIDATES,measured_eta

REPORT_FILES=('rq2_test_report.json','rq2_test_metrics.csv','test_predictions.json',
              'test_capture/capture_manifest.json','test_capture/capture_contract.json')
TEST_COUNT=1344


def checked_report(run,seed,role,lock_sha):
    run=Path(run);report=load(run/'rq2_test_report.json');seal=load(run/'evaluation_complete.json')
    status=load(run/'training_status.json');fingerprint=load(run/'test_capture/capture_contract.json')
    checkpoint=run/status['best_checkpoint']
    if (report.get('complete') is not True or report.get('test_count')!=TEST_COUNT or report.get('seed')!=seed
            or seal.get('complete') is not True or seal.get('test_count')!=TEST_COUNT
            or seal.get('metric_schema')!='rq2-28-v1' or report.get('capture_fingerprint')!=fingerprint
            or fingerprint.get('selection_lock_sha256')!=lock_sha
            or fingerprint.get('checkpoint_sha256')!=sha(checkpoint)
            or report.get('checkpoint_sha256')!=sha(checkpoint)
            or seal.get('checkpoint_sha256')!=sha(checkpoint)
            or fingerprint.get('training_contract')!=load(run/'run_manifest.json')['contract']
            or fingerprint['training_contract'].get('arm')!=role or fingerprint['training_contract'].get('seed')!=seed
            or report.get('selected_checkpoint')!=status['best_checkpoint']):raise ValueError('Wrong/incomplete sealed report: '+str(run))
    if set(seal.get('files',{}))!=set(REPORT_FILES):raise ValueError('Wrong evaluation seal inventory.')
    for name in REPORT_FILES:
        if sha(run/name)!=seal['files'][name]:raise ValueError('Changed sealed report artifact: '+name)
    hashes=report.get('curve_file_sha256',{})
    if len(hashes)!=TEST_COUNT or {p.stem for p in (run/'test_capture/curves').glob('*.json')}!=set(hashes):
        raise ValueError('Need all1344 curves before comparison.')
    for ident,expected in hashes.items():
        if sha(run/'test_capture/curves'/(ident+'.json'))!=expected:raise ValueError('Changed scored curve: '+ident)
    values=report['summary_metrics']
    if len(values)!=28 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):raise ValueError('Invalid test metric schema.')
    return report


def prior_parity(out,seed):
    """Diagnostic: H7 (b0) re-scored here vs the same H7 scored in the v1.1 study (same checkpoints, same videos)."""
    import numpy as np
    value=load(out/'prepared.json')['prior_study']['root']
    if value is None:
        return dict(seed=seed,videos=0,max_abs_score_diff=None,mean_abs_score_diff=None,share_within_1e_3=None,note='Not performed: prior is optional and never used for main comparisons.')
    root=Path(value)
    original={r['id']:float(r['score']) for r in load(root/'test/runs/b0'/('seed_%d'%seed)/'run/test_predictions.json')}
    ours={r['id']:float(r['score']) for r in load(out/'test/runs/b0'/('seed_%d'%seed)/'run/test_predictions.json')}
    if set(original)!=set(ours):raise ValueError('Prior and re-scored H7 test IDs differ.')
    d=np.abs(np.array([ours[k]-original[k] for k in sorted(ours)]))
    return dict(seed=seed,videos=len(d),max_abs_score_diff=float(d.max()),mean_abs_score_diff=float(d.mean()),
                share_within_1e_3=float(np.mean(d<=1e-3)),
                note='Same H7 checkpoint, same pipeline: ~0 expected. Required for comparing the new arms with v1.1 A7 (=M1) / G3 (=M1+M2).')


def report(out):
    out=Path(out);completion=out/'comparison/TEST_COMPARISON_COMPLETE.json'
    if completion.exists():completion.replace(completion.with_name('TEST_COMPARISON_COMPLETE.previous_%d.json'%time.time_ns()))
    plan=load(out/'selection_frozen_before_test.json');lock_sha=digest(plan)
    if sha(out/'study_contract.json')!=plan['study_contract_sha256']:raise ValueError('Study changed after selection freeze.')
    if sha(out/'runtime_identity.json')!=plan['runtime_identity_sha256']:raise ValueError('Runtime identity changed after selection freeze.')
    all_reports={};paired=[];wide=[];ids=None;clocks=set();parity=[]
    for seed in SEEDS:
        marker=load(out/('test_complete_seed%d.json'%seed))
        if marker.get('complete') is not True or marker.get('seed')!=seed or marker.get('selection_lock_sha256')!=lock_sha:
            raise ValueError('All3seeds must finish same frozen study.')
        for role in ('b0',*ARMS):
            r=checked_report(out/'test/runs'/role/('seed_%d'%seed)/'run',seed,role,lock_sha)
            if role in ARMS:
                selected=next(row for row in plan['chosen'][role]['selected'] if row['seed']==seed)
                expected=selected['adapter_sha256']
                if digest(r['capture_fingerprint']['training_contract'])!=selected['contract_sha256']:
                    raise ValueError('Test used another selected training contract.')
            else:
                base=plan['baselines']['h7'][str(seed)];expected=base['checkpoint_sha256']
                if r['capture_fingerprint']['training_contract'].get('original_contract_sha256')!=base['contract_sha256']:
                    raise ValueError('Reference baseline contract differs from frozen plan.')
            if r['checkpoint_sha256']!=expected:raise ValueError('Test checkpoint differs from frozen validation selection.')
            current=set(r['curve_file_sha256'])
            if ids is not None and current!=ids:raise ValueError('Test case IDs differ between models/seeds.')
            ids=current;clocks.add(r['capture_fingerprint']['blind_index_sha256']);all_reports[(role,seed)]=r
        parity.append(prior_parity(out,seed))
        b=all_reports[('b0',seed)]['summary_metrics']
        print('RESULT FINAL seed=%d model=H7(b0) mAP=%.4f%% AP1.5=%.4f%% | vs H7 scored in v1.1: max|d|=%.2e mean|d|=%.2e'%
              (seed,b['mAP']*100,b['AP@1.5s']*100,parity[-1]['max_abs_score_diff'],parity[-1]['mean_abs_score_diff']),flush=True)
        for arm in ARMS:
            selected=plan['chosen'][arm];metrics=all_reports[(arm,seed)]['summary_metrics']
            row=dict(seed=seed,module=arm,shared_candidate=selected['candidate'],lr=selected['hyperparameters']['lr'],dropout=selected['hyperparameters']['dropout'])
            for metric in ('mAP','mAUC_grouped','AP@0.5s','AP@1.0s','AP@1.5s'):
                row[metric+'_percent']=metrics[metric]*100
                row['H7_'+metric+'_percent']=b[metric]*100;row['delta_vs_H7_'+metric+'_pp']=(metrics[metric]-b[metric])*100
            wide.append(row)
            for metric,value in metrics.items():
                bv=b[metric];seconds='mTTA' in metric
                paired.append(dict(seed=seed,module=arm,shared_candidate=selected['candidate'],baseline='H7',baseline_seed=seed,
                    metric=metric,module_value=value,baseline_value=bv,signed_delta=(value-bv)*(1 if seconds else 100),
                    delta_unit='seconds' if seconds else 'percentage_points',
                    desired_direction='lower' if 'FAR' in metric else 'higher',paper_equivalent=False))
            print('RESULT FINAL seed=%d module=%s config=%s mAP=%.4f%% delta_H7=%+.4fpp AP1.5=%.4f%% delta_H7_AP1.5=%+.4fpp'%
                  (seed,arm,selected['candidate'],metrics['mAP']*100,row['delta_vs_H7_mAP_pp'],metrics['AP@1.5s']*100,row['delta_vs_H7_AP@1.5s_pp']),flush=True)
    if len(clocks)!=1:raise ValueError('All paths must use identical blind native clocks/index.')
    csv_write(out/'comparison/per_seed_summary.csv',wide)
    csv_write(out/'comparison/per_seed_all28_metrics_vs_H7.csv',paired)
    csv_write(out/'comparison/h7_rescore_vs_v11_parity.csv',parity)
    base_rows=[]
    for seed in SEEDS:
        for role in ('b0',*ARMS):
            m=all_reports[(role,seed)]['summary_metrics']
            base_rows.append(dict(seed=seed,model='H7' if role=='b0' else role,**{k:m[k] for k in sorted(m)}))
    csv_write(out/'comparison/all_models_all28_metrics.csv',base_rows)
    lines=['# Test: ablation V2 (C7, R7, A26, U3) trên H7 đóng băng, so với H7 cùng seed','',
           'Một cấu hình định trước (c2_lr_double, giống M1/M1+M2 ở v1.1). Delta=nhánh−H7, đơn vị percentage points.','',
           '| Seed | Arm | mAP% | H7 mAP% | ΔmAP pp | AP@1.5s% | H7 AP@1.5s% | ΔAP@1.5s pp |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in wide:
        lines.append('| %d | %s | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |'%
            (r['seed'],r['module'],r['mAP_percent'],r['H7_mAP_percent'],r['delta_vs_H7_mAP_pp'],
             r['AP@1.5s_percent'],r['H7_AP@1.5s_percent'],r['delta_vs_H7_AP@1.5s_pp']))
    lines+=['','Kiểm tra H7 (b0) chấm lại so với H7 đã chấm ở đợt v1.1:','',
            '| Seed | max abs | mean abs | tỉ lệ ≤1e-3 |','|---:|---:|---:|---:|']
    for r in parity:
        if r['max_abs_score_diff'] is None:lines.append('| %d | N/A | N/A | N/A |'%r['seed'])
        else:lines.append('| %d | %.2e | %.2e | %.3f |'%(r['seed'],r['max_abs_score_diff'],r['mean_abs_score_diff'],r['share_within_1e_3']))
    lines+=['','Đủ28metrics trong CSV. Kết luận thống kê: chạy bootstrap_paired.py (luật trong PREREGISTERED_ANALYSIS.md).']
    (out/'comparison/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(completion,dict(complete=True,seeds=list(SEEDS),selection_lock_sha256=lock_sha,
        rows_per_seed_summary=len(wide),rows_all_metrics=len(paired),test_count=1344,
        files={name:sha(out/'comparison'/name) for name in ('per_seed_summary.csv','per_seed_all28_metrics_vs_H7.csv',
               'h7_rescore_vs_v11_parity.csv','all_models_all28_metrics.csv','RESULTS.md')}))


def estimate(out):
    out=Path(out);epoch={};coverage={}
    for arm in ARMS:
        values=[]
        for c in CANDIDATES:
            for seed in SEEDS:
                p=out/'trials'/c['id']/arm/('seed_%d'%seed)/'run/named_epochs.json'
                if p.exists():values.extend(r['seconds'] for r in load(p)[-3:])
        if values:epoch[arm]=statistics.median(values);coverage[arm]=len(values)
    result=dict(measured_epoch_seconds=epoch,samples=coverage,
                training_runs=len(ARMS)*len(CANDIDATES)*len(SEEDS),epochs=50*len(ARMS)*len(CANDIDATES)*len(SEEDS),
                extraction_videos=4500,test_paired_videos=4032,
                runtime_and_data_audit='separate, not hidden in training estimate',
                scenario_training_hours_at_uniform_epoch_seconds={str(s):len(ARMS)*len(CANDIDATES)*len(SEEDS)*50*s/3600 for s in (15,20,30,60)},
                scenario_note='Arithmetic examples, NOT a measured RTX4080 prediction.')
    remaining={arm:0 for arm in ARMS};completed=0
    for arm in ARMS:
        for c in CANDIDATES:
            for seed in SEEDS:
                path=out/'trials'/c['id']/arm/('seed_%d'%seed)/'run/training_status.json'
                count=load(path).get('completed_epochs',0) if path.exists() else 0
                if type(count) is not int or not 0<=count<=50:raise ValueError('Invalid training progress for ETA.')
                remaining[arm]+=50-count;completed+=int(count==50)
    result['remaining_epochs_by_module']=remaining;result['completed_runs']=completed
    if set(epoch)==set(ARMS):
        result['current_GPU_projection']=measured_eta(epoch,completed_runs=completed)
        result['estimated_remaining_training_hours']=sum(epoch[arm]*remaining[arm] for arm in ARMS)/3600
    else:result['current_GPU_projection']='Await pilots for all declared modules; do not extrapolate from GPU peak TFLOPS.'
    save(out/'runtime_estimate.json',result);print('ETA '+str(result),flush=True)
    return result
