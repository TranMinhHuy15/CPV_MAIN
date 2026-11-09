"""Study policy, artifact identities and selection. Standard library only."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

SEEDS = (42, 43, 44)
ARMS = ('c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned')
METRICS = ('mAP@', 'mAUC@', 'AP@0.5s', 'AP@1.0s', 'AP@1.5s')
PROTOCOL = 'rq3-ablation-v3-base-h7-matched-six-arms-c2-three-seeds'
DATA_REVISION = '7535d0656dac31d7da2846913bb69c6331d2a70a'
SOURCE_COMMIT = '579376fa1d879f28a9d68c2565d1e449de24b666'
# One configuration only: identical to the configuration the v1.1 H7 study selected for M1 and M1+M2 (c2),
# so the controls differ from those modules ONLY in the information they receive.
CANDIDATES = (
    dict(id='c2_lr_double', lr=2e-4, weight_decay=1e-4, dropout=.1, width=512,
         rationale='Configuration selected for M1 and M1+M2 in the v1.1 H7 study; reused unchanged so the ablation is matched.'),
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def bind(path,value):
    path=Path(path)
    if path.exists() and load(path)!=value:raise ValueError('Identity changed; preserve outputs and use a NEW output_root: '+str(path))
    if not path.exists():save(path,value)
    return digest(value)


def csv_write(path,rows):
    if not rows:raise ValueError('Refuse empty result table.')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def candidate(name):
    return dict(next(c for c in CANDIDATES if c['id']==name))


def validate_candidates():
    identities=[digest({k:v for k,v in c.items() if k not in ('id','rationale')}) for c in CANDIDATES]
    if len(set(c['id'] for c in CANDIDATES))!=len(CANDIDATES) or len(set(identities))!=len(CANDIDATES):
        raise ValueError('Duplicate candidate ID or actual hyperparameters.')
    return True


def plain_name(name):
    if not isinstance(name,str) or not name or name in ('.','..') or any(c in name for c in '/\\:'):
        raise ValueError('Unsafe checkpoint filename.')
    return name


def validate_config(cfg):
    validate_candidates()
    required={'source_zip','data_root','output_root','runtime_root','h7_runs','prior_study_root','cache_roots',
              'encode_chunk','log_every_batches','compact_completed_trials'}
    if set(cfg)!=required:raise ValueError('Use exact inputs.example.json fields; unexpected/missing: '+str(set(cfg)^required))
    for key in ('h7_runs','cache_roots'):
        if set(cfg[key])!={str(s) for s in SEEDS}:raise ValueError(key+' must explicitly include42/43/44.')
    for key in ('source_zip','data_root','output_root','runtime_root'):
        if not isinstance(cfg[key],str) or not cfg[key].strip():raise ValueError('Fill '+key)
    if type(cfg['encode_chunk']) is not int or not 1<=cfg['encode_chunk']<=16:raise ValueError('encode_chunk1..16')
    if type(cfg['log_every_batches']) is not int or cfg['log_every_batches']<1:raise ValueError('Positive log interval.')
    if type(cfg['compact_completed_trials']) is not bool:raise ValueError('compact_completed_trials must be bool.')
    inputs=[Path(cfg['data_root']).resolve(),Path(cfg['source_zip']).resolve()]
    if cfg['prior_study_root'] is not None:
        if not isinstance(cfg['prior_study_root'],str) or not cfg['prior_study_root'].strip():raise ValueError('prior_study_root must be null or a path.')
        inputs.append(Path(cfg['prior_study_root']).resolve())
    for key in ('h7_runs','cache_roots'):
        for value in cfg[key].values():
            if value is not None:inputs.append(Path(value).resolve())
    outputs=[Path(cfg[key]).resolve() for key in ('output_root','runtime_root')]
    for out in outputs:
        for source in inputs:
            if out==source or out in source.parents or source in out.parents:raise ValueError('Output/runtime must not overlap original inputs.')
    if outputs[0]==outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise ValueError('Keep runtime_root and output_root separate.')
    return cfg


def history_selection(run):
    run=Path(run);status=load(run/'training_status.json')
    history=[json.loads(s) for s in (run/'validation_history.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if (status.get('complete') is not True or status.get('completed_epochs')!=50 or status.get('target_epochs')!=50
            or [r.get('epoch') for r in history]!=list(range(1,51))):raise ValueError('Need exactly50 committed epochs: '+str(run))
    for row in history:
        for key in METRICS:
            v=row.get('metrics',{}).get(key)
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid validation metric '+key)
    best=max(history,key=lambda r:r['metrics']['mAUC@'])
    checkpoint=plain_name(status.get('best_checkpoint'))
    if checkpoint!='best_mAUC@_epoch_%d.pth'%best['epoch']:raise ValueError('Best checkpoint does not match earliest maximum validation mAUC.')
    threshold=best['metrics'].get('threshold@0.10')
    if type(threshold) not in (int,float) or not math.isfinite(threshold):raise ValueError('Missing frozen validation threshold.')
    return dict(epoch=best['epoch'],checkpoint=checkpoint,metrics=best['metrics'],validation_threshold=threshold)


# Frozen base for this ablation: the same H7 checkpoints used in the v1.1 module study (approved selected export).
def validate_baseline(run,seed,history,assert_committed=None,verified_export=None):
    run=Path(run).resolve();manifest=load(run/'run_manifest.json');c=manifest['contract'];p=c.get('rq2',{})
    expected=dict(history_tokens=history,warmup_tokens=25,target_tokens=30,input_tokens=55,snippet_frames=5,native_frame_stride=3)
    if (type(c.get('seed')) is not int or c['seed']!=seed or c.get('source_commit')!=SOURCE_COMMIT
            or c.get('data_revision')!=DATA_REVISION or any(p.get(k)!=v for k,v in expected.items())):
        raise ValueError('Wrong seed/history/source/data/sampling baseline: '+str(run))
    if (run/'selected_export.json').is_file():
        if history!=7 or verified_export is None:raise ValueError('Only the original approved H7 selected export is supported; attach full H13 run.')
        verified_export(run,seed);kind='selected_frozen_H7_export'
    elif assert_committed is not None:
        assert_committed(run,load(run/'training_status.json'),require_complete=True);kind='full_committed_run'
    else:kind='unchecked'
    selected=history_selection(run);cp=run/selected['checkpoint']
    if not cp.is_file() or cp.stat().st_size==0:raise FileNotFoundError(cp)
    return dict(run=str(run),checkpoint=str(cp),checkpoint_sha256=sha(cp),contract_sha256=digest(c),
                recipe_sha256=digest({k:c.get(k) for k in ('protocol','source_commit','data_revision','epochs','batch_per_gpu',
                    'effective_batch_size','world_size','amp','precision','loss','preprocessing','rq2')}),
                history=history,seed=seed,baseline_input_kind=kind,selected_epoch=selected['epoch'],validation_metrics={k:selected['metrics'][k] for k in METRICS},
                validation_threshold=selected['validation_threshold'],manifest_sha256=sha(run/'run_manifest.json'),
                validation_history_sha256=sha(run/'validation_history.jsonl'))


def rank_common(rows):
    """ONE candidate per module across all THREE seeds; never use held-out test."""
    expected={(a,c['id'],s) for a in ARMS for c in CANDIDATES for s in SEEDS}
    keys=[(r['arm'],r['candidate'],r['seed']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Need the complete unique candidate x module x seed matrix before selection.')
    rankings=[];chosen={}
    for arm in ARMS:
        arm_rows=[]
        for order,c in enumerate(CANDIDATES):
            subset=sorted((r for r in rows if r['arm']==arm and r['candidate']==c['id']),key=lambda r:r['seed'])
            values=[r['validation_metrics']['mAUC@'] for r in subset]
            aps=[r['validation_metrics']['AP@1.5s'] for r in subset]
            if any(type(x) not in (int,float) or not math.isfinite(x) for x in values+aps):raise ValueError('Nonfinite validation rank.')
            row=dict(arm=arm,candidate=c['id'],n=3,mean_validation_mAUC=statistics.mean(values),
                     sd_validation_mAUC=statistics.stdev(values),mean_validation_AP15=statistics.mean(aps),
                     **{'seed%d_validation_mAUC'%r['seed']:r['validation_metrics']['mAUC@'] for r in subset})
            arm_rows.append((row,order,subset))
        arm_rows.sort(key=lambda x:(-x[0]['mean_validation_mAUC'],-x[0]['mean_validation_AP15'],x[1]))
        winner=arm_rows[0]
        chosen[arm]=dict(candidate=winner[0]['candidate'],hyperparameters=candidate(winner[0]['candidate']),
                         selected=winner[2],reason='Maximum mean full validation mAUC across42/43/44; ties: mean AP@1.5s then predeclared candidate order. Same candidate for all three seeds.')
        for rank,(row,_,_) in enumerate(arm_rows,1):rankings.append(dict(rank=rank,**row))
    return chosen,rankings


def measured_eta(train_seconds_per_epoch,extract_seconds_per_video=None,test_seconds_per_video=None,completed_runs=0):
    if set(train_seconds_per_epoch)!=set(ARMS):raise ValueError('Measure all declared module pilots.')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in train_seconds_per_epoch.values()):raise ValueError('Positive measured epoch seconds required.')
    runs_per_arm=len(CANDIDATES)*len(SEEDS)
    seconds=sum(train_seconds_per_epoch[a]*50*runs_per_arm for a in ARMS)
    return dict(training_runs=runs_per_arm*len(ARMS),epochs=50*runs_per_arm*len(ARMS),
                training_hours=seconds/3600,extraction_hours=None if extract_seconds_per_video is None else extract_seconds_per_video*1500*3/3600,
                test_hours=None if test_seconds_per_video is None else test_seconds_per_video*1344*3/3600,
                scope='Measured on current GPU; one H7 encoder per test video shared by b0 and the four ablation arms. Setup, data audit, report I/O listed separately.',
                completed_runs=completed_runs)


PRIOR_ROLES = ('b0', 'm1', 'm1_m2', 'm1_m3')  # from the v1.1 H7 study (same H7 checkpoints, same 1,344 test videos)


def validate_prior_study(root, h7_rows):
    """Read-only check that the v1.1 study outputs are present and used the same frozen H7 checkpoints."""
    root = Path(root).resolve()
    if not (root/'comparison/TEST_COMPARISON_COMPLETE.json').is_file():
        raise FileNotFoundError('Prior v1.1 study is incomplete: '+str(root))
    plan = load(root/'selection_frozen_before_test.json')
    out = {}
    for seed in SEEDS:
        prior = plan['baselines']['h7'][str(seed)]['checkpoint_sha256']
        if prior != h7_rows[str(seed)]['checkpoint_sha256']:
            raise ValueError('Prior study used another H7 checkpoint for seed %d.' % seed)
        for role in PRIOR_ROLES:
            f = root/'test/runs'/role/('seed_%d' % seed)/'run/test_predictions.json'
            rows = load(f)
            if len(rows) != 1344 or len({r['id'] for r in rows}) != 1344:
                raise ValueError('Prior predictions must cover 1344 test videos: '+str(f))
            out['%s_%d' % (role, seed)] = sha(f)
    chosen = {a: plan['chosen'][a]['candidate'] for a in ('m1', 'm1_m2', 'm1_m3')}
    if set(chosen.values()) != {'c2_lr_double'}:
        raise ValueError('Ablation is matched to c2_lr_double; prior study chose '+str(chosen))
    return dict(root=str(root), prediction_sha256=out, prior_chosen=chosen,
                selection_sha256=sha(root/'selection_frozen_before_test.json'))
