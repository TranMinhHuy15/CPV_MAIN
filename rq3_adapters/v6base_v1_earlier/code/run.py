"""CPU supervisor. Running all/setup installs/executes only on the user's chosen host."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy'))
from protocol import SEEDS,ARMS,CANDIDATES,load,save,sha,bind,validate_config,validate_v6_baseline,rank_common,csv_write
from console import Log,command


def prepare(cfg):
    from module_inputs import materialize_source,verify_materialized
    out=Path(cfg['output_root']).resolve();out.mkdir(parents=True,exist_ok=True)
    root=materialize_source(cfg['source_zip'],out/'checked_source')
    verify_materialized(root)
    sys.path.insert(0,str(root/'base_h7'));sys.path.insert(0,str(root))
    adapter_sha=sha(root/'base_h7'/'kaggle_adapter.py')
    rows={str(seed):validate_v6_baseline(cfg['v6_runs'][str(seed)],seed,adapter_sha) for seed in SEEDS}
    if len({r['recipe_sha256'] for r in rows.values()})!=1:raise ValueError('V6 scientific recipe differs across the three base runs.')
    if len({r['checkpoint_sha256'] for r in rows.values()})!=3:raise ValueError('The three V6 bases must be three different checkpoints.')
    baselines={'v6':rows}
    data=Path(cfg['data_root'])
    for split in ('train','test-public','test-private'):
        if not (data/split).is_dir():raise FileNotFoundError(data/split)
    code={str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in list(HERE.glob('*.py'))+list((HERE/'legacy').glob('*.py'))}
    contract=dict(config=cfg,baselines=baselines,candidates=list(CANDIDATES),code=code,
                  selection='single predeclared candidate (c2_lr_double); epoch per run by earliest max validation mAUC; no test selection',
                  study='exploratory: history adapters on frozen RiskProp V6 (seeds 42/43 original, 44 rerun); test set previously inspected')
    bind(out/'study_contract.json',contract)
    save(out/'prepared.json',dict(source_root=str(root),baselines=baselines))
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
    parser.add_argument('phase',choices=('preflight','setup','extract','tune','freeze','test','report','estimate','all'))
    args=parser.parse_args();cfg=validate_config(load(args.config))
    out,root,baselines=prepare(cfg);log=Log(out/'logs');env=environment()
    log.event('INPUT READY',seeds=list(SEEDS),modules=list(ARMS),candidates=len(CANDIDATES),runs=len(SEEDS)*len(ARMS)*len(CANDIDATES),epochs_per_run=50,
              module_parent='RiskProp V6 frozen',comparison_baselines='V6 same seed (b0) + zero-parameter smoothing control',disk_free_GiB=round(shutil.disk_usage(out).free/2**30,2))
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
    def worker(phase,seed,arm=None,c=None):
        args=[py,HERE/'worker.py','--config',canonical,'--phase',phase,'--seed',seed]
        if arm:args+=['--arm',arm]
        if c:args+=['--candidate',c]
        command(args,'%s_seed%d%s%s'%(phase,seed,'_'+arm if arm else '','_'+c if c else ''),log,env)
    if args.phase in ('extract','all'):
        # Teacher right after each extraction: its V6 validation-parity gate stops a wrong base early.
        for seed in SEEDS:worker('extract',seed);worker('teacher',seed)
    if args.phase in ('tune','all'):
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
