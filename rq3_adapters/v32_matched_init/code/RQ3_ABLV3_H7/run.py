"""CPU supervisor. Running all/setup installs/executes only on the user's chosen host."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy'))
from protocol import SEEDS,ARMS,CANDIDATES,load,save,sha,bind,validate_config,validate_baseline,validate_prior_study,rank_common,csv_write
from console import Log,command


def prepare(cfg):
    from module_inputs import materialize_source,verify_materialized
    out=Path(cfg['output_root']).resolve()
    from study_guard import ensure_output,code_hashes,plan_identity,archive_plan,VERSION
    ensure_output(out)
    # Catch missing dataset mounts before creating the output directory.
    for split in ('train', 'test-public', 'test-private'):
        if not (Path(cfg['data_root']) / split).is_dir():
            raise FileNotFoundError(Path(cfg['data_root']) / split)
    out.mkdir(parents=True, exist_ok=True)
    root=materialize_source(cfg['source_zip'],out/'checked_source')
    verify_materialized(root)
    sys.path.insert(0,str(root/'base_h7'));sys.path.insert(0,str(root))
    from training_state import assert_committed
    from h7_selected_export import verified_baseline
    rows={str(seed):validate_baseline(cfg['h7_runs'][str(seed)],seed,7,assert_committed,verified_baseline) for seed in SEEDS}
    if len({r['recipe_sha256'] for r in rows.values()})!=1:raise ValueError('Baseline H7 scientific recipe differs across seeds.')
    for seed,row in rows.items():
        contract=load(Path(row['run'])/'run_manifest.json')['contract']
        for name in ('rq2_model.py','kaggle_adapter.py'):
            if contract.get('launcher_sha256',{}).get(name)!=sha(root/'base_h7'/name):
                raise ValueError('Incompatible baseline forward/preprocessing source: H7 seed%s %s'%(seed,name))
    prior=dict(root=cfg['prior_study_root'],used_for_main=False)
    baselines={'h7':rows}
    code=code_hashes(HERE)
    contract=dict(source_root=str(root.resolve()),study_version=VERSION,analysis_plan=plan_identity(HERE),solution_sha256=sha(root/'base_h7/metadata_snapshot/solution.csv'),config=cfg,baselines=baselines,prior_study=prior,candidates=list(CANDIDATES),code=code,
                  selection='single predeclared candidate (c2_lr_double); epoch per run by earliest max validation mAUC; no test selection',
                  study='exploratory matched ablation V3 on the frozen H7 of the v1.1 study: all six adapters trained anew; no prior prediction reuse; test set previously inspected')
    bind(out/'study_contract.json',contract)
    archive_plan(out,HERE)
    save(out/'prepared.json',dict(source_root=str(root),baselines=baselines,prior_study=prior))
    return out,root,baselines


def environment():
    env=dict(os.environ)
    for key in ('PYTHONHOME','PYTHONPATH','RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',MPLBACKEND='Agg',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def summaries(out):
    rows=[]
    for c in CANDIDATES:
        for arm in ARMS:
            for seed in SEEDS:
                marker=out/'trials'/c['id']/arm/('seed_%d'%seed)/'trial_complete.json'
                if not marker.is_file():raise FileNotFoundError('Complete all trials before freeze/test: '+str(marker))
                row=load(marker)
                from artifacts import verify_trial
                verify_trial(marker.parent,row)
                rows.append(row)
    return rows


def freeze_selection(out,baselines):
    selected,rankings=rank_common(summaries(out))
    plan=dict(seeds=list(SEEDS),arms=list(ARMS),chosen=selected,baselines=baselines,
              study_contract_sha256=sha(out/'study_contract.json'),primary_selection='mean validation full mAUC across all3seeds')
    plan['runtime_identity_sha256']=sha(out/'runtime_identity.json')
    lock=bind(out/'selection_frozen_before_test.json',plan)
    csv_write(out/'comparison'/'validation_config_ranking.csv',rankings)
    save(out/'comparison'/'chosen_hyperparameters.json',selected)
    lines=['# Validation (một cấu hình duy nhất c2_lr_double, định trước)','',
           'Mỗi run chọn epoch bằng earliest max validation mAUC@. Không dùng test.','',
           '| Module | Rank | Config | Seed42 mAUC% | Seed43 mAUC% | Seed44 mAUC% | Mean% | SD pp | Chọn |',
           '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rankings:
        lines.append('| %s | %d | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %s |'%
            (r['arm'],r['rank'],r['candidate'],r['seed42_validation_mAUC']*100,r['seed43_validation_mAUC']*100,
             r['seed44_validation_mAUC']*100,r['mean_validation_mAUC']*100,r['sd_validation_mAUC']*100,'yes' if r['rank']==1 else ''))
    lines+=['','Hyperparameters và lý do từng cấu hình: chosen_hyperparameters.json; rationale định trước trong protocol.py.']
    (out/'comparison'/'VALIDATION_SELECTION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('phase',choices=('preflight','setup','extract','tune','freeze','test','report','estimate','pilot','all'))
    args=parser.parse_args();cfg=validate_config(load(args.config))
    out,root,baselines=prepare(cfg);log=Log(out/'logs');env=environment()
    log.event('INPUT READY',seeds=list(SEEDS),modules=list(ARMS),candidates=len(CANDIDATES),runs=len(SEEDS)*len(ARMS)*len(CANDIDATES),epochs_per_run=50,
              module_parent='H7 frozen',comparison_baselines='same-study b0 and all six newly trained adapters',disk_free_GiB=round(shutil.disk_usage(out).free/2**30,2))
    runtime=Path(cfg['runtime_root']);py=runtime/'venv/bin/python'
    setup=args.phase in ('setup','all')
    if setup:
        missing=[n for n in ('ffmpeg','ffprobe','nvidia-smi') if shutil.which(n) is None]
        if missing:raise RuntimeError('Install host prerequisites before setup: '+str(missing))
        if shutil.disk_usage(out).free<15*2**30:raise RuntimeError('Need15GiB reserve for isolated runtime and work after attaching inputs.')
        command([sys.executable,root/'base_h7/setup_environment.py','--launcher',root/'base_h7','--runtime-root',runtime,'--world-size','1'],'runtime_setup',log,env)
    if args.phase in ('preflight','setup'):return
    if args.phase=='freeze':freeze_selection(out,baselines);return
    if args.phase=='report':
        from reporting import report
        report(out);return
    if args.phase=='estimate':
        from reporting import estimate
        estimate(out);return
    if not py.is_file():raise FileNotFoundError('Run setup first: '+str(py))
    canonical=out/'inputs.resolved.json';save(canonical,cfg)
    def worker(phase,seed,arm=None,c=None,stop_after=None):
        args=[py,HERE/'worker.py','--config',canonical,'--phase',phase,'--seed',seed]
        if arm:args+=['--arm',arm]
        if c:args+=['--candidate',c]
        if stop_after is not None:args+=['--stop-after-epoch',str(stop_after)]
        command(args,'%s_seed%d%s%s'%(phase,seed,'_'+arm if arm else '','_'+c if c else ''),log,env,accepted=(75,) if stop_after is not None else (0,))
    def pilot():
        seed=SEEDS[0];arm=ARMS[0];c=CANDIDATES[0]['id']
        trial=out/'trials'/c/arm/('seed_%d'%seed)/'run'
        worker('teacher', seed)
        from training_state import assert_committed
        # The locked worker restores the last snapshot before consulting progress.
        worker('train', seed, arm, c, stop_after=2)
        status2 = load(trial / 'training_status.json')
        if status2['completed_epochs'] not in (2, 3):
            raise ValueError('Pilot has not reached committed epoch 2 or 3.')
        assert_committed(trial, status2)
        worker('train', seed, arm, c, stop_after=3)
        status=load(trial/'training_status.json');assert_committed(trial,status)
        history=[json.loads(s) for s in (trial/'validation_history.jsonl').read_text().splitlines() if s.strip()]
        if status['completed_epochs']!=3 or [r['epoch'] for r in history]!=[1,2,3]:raise ValueError('Pilot recovery did not reach exactly three unique committed epochs.')
        save(out/'PILOT_RESUME_RECEIPT.json',dict(complete=True,epochs=[1,2,3],study_contract_sha256=sha(out/'study_contract.json'),
             scope='snapshot verified at pilot boundaries 2 and 3; completed pilot stages may be reused after interruption'))
        log.event('PILOT DONE',next_action='tune to finish all 18 runs; first trial resumes from epoch3')
        return
    if args.phase=='pilot':pilot();return
    if args.phase in ('extract','all'):
        # Teacher right after each extraction (cheap; checks cached vs direct frozen-H7 logits).
        for seed in SEEDS:worker('extract',seed);worker('teacher',seed)
    if args.phase in ('tune','all'):
        from recovery import check_training_space
        check_training_space(out, baselines, cfg['compact_completed_trials'])
        receipt=out/'PILOT_RESUME_RECEIPT.json'
        if not receipt.exists():pilot()
        pilot_receipt=load(receipt)
        if pilot_receipt.get('complete') is not True or pilot_receipt.get('epochs')!=[1,2,3] or pilot_receipt.get('study_contract_sha256')!=sha(out/'study_contract.json'):raise ValueError('Pilot is incomplete or belongs to another study.')
        for seed in SEEDS:worker('teacher',seed)
        for c in CANDIDATES:
            for arm in ARMS:
                for seed in SEEDS:worker('train',seed,arm,c['id'])
                from reporting import estimate
                estimate(out)
        freeze_selection(out,baselines)
    if args.phase in ('test','all'):
        freeze_selection(out,baselines)
        for seed in SEEDS:worker('test',seed)
        from reporting import report
        report(out)
    log.event('DONE',completed_phase=args.phase,output_root=str(out))


if __name__=='__main__':main()
