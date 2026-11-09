"""Strict artifact gates around unchanged V6 code. Does not change model/loss/RNG."""
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import tempfile
import zipfile

from training_state import assert_committed, basename, atomic_json

ROOT=Path(__file__).resolve().parent
LEGACY=ROOT/'legacy_v6'
PROTOCOL='riskprop-v6-vast-batch2-eff16-amp-single-gpu-3seed-terminal-v3'
EVAL_FILES=('validation_metrics.json','validation_predictions.json','official_test_metrics.json',
            'evaluation_notes.json','test_predictions.json','submission.csv')
LAUNCHERS={'repro.py','worker.py','kaggle_adapter.py','evaluate.py','amp_adapter.py',
           'run_vast.py','run_three_seeds.py','process_control.py','checkpoint_safety.py'}


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def csv_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def ap_auc(labels,scores):
    if len(labels)!=len(scores) or not labels or set(labels)!={0,1}:
        raise ValueError('Require both labels and matched predictions.')
    if not all(math.isfinite(p) and 0<=p<=1 for p in scores):raise ValueError('Invalid probabilities')
    npos=sum(labels);nneg=len(labels)-npos
    ranked=sorted(zip(scores,labels),reverse=True)
    tp=fp=0;ap=0.;wins=0.;i=0
    while i<len(ranked):
        j=i+1
        while j<len(ranked) and ranked[j][0]==ranked[i][0]:j+=1
        positives=sum(y for _,y in ranked[i:j]);negatives=j-i-positives
        # A positive beats all lower negatives; ties count one half.
        wins+=positives*(nneg-fp-negatives)+.5*positives*negatives
        tp+=positives;fp+=negatives
        ap+=positives/npos*tp/(tp+fp)
        i=j
    return ap,wins/(npos*nneg)


def training(run,complete=False):
    run=Path(run)
    status=load(run/'training_status.json');contract=load(run/'run_manifest.json')['contract']
    if (contract.get('protocol')!=PROTOCOL or type(contract.get('seed')) is not int
            or contract.get('seed') not in (42,43,44) or contract.get('world_size')!=1):
        raise ValueError('Expected V6 RiskProp seed42/43/44, not H7/H1 LSTM package.')
    if set(contract.get('launcher_sha256',{}))!=LAUNCHERS:raise ValueError('Incomplete launcher provenance')
    for name,value in contract.get('launcher_sha256',{}).items():
        basename(name)
        if digest(LEGACY/name)!=value:raise ValueError('Legacy code does not match saved run: '+name)
    if not contract.get('launcher_sha256'):raise ValueError('Missing launcher provenance')
    assert_committed(run,status,require_complete=complete)
    history=[json.loads(s) for s in (run/'validation_history.jsonl').read_text().splitlines() if s.strip()]
    selected=max(history,key=lambda r:r['metrics']['mAUC@'])
    expected='best_mAUC@_epoch_%d.pth'%selected['epoch']
    if status['best_checkpoint']!=expected:raise ValueError('Best checkpoint does not match validation full mAUC@ argmax')
    return status,contract,history


def evaluation(run):
    run=Path(run);status,contract,history=training(run,complete=True)
    for name in EVAL_FILES:
        if not (run/name).is_file() or not (run/name).stat().st_size:raise ValueError('Incomplete evaluation: '+name)
    notes=load(run/'evaluation_notes.json')
    if (notes.get('selected_checkpoint')!=status['best_checkpoint'] or notes.get('completed_epochs')!=50
            or notes.get('test_was_not_used_for_selection') is not True):
        raise ValueError('Evaluation notes do not identify the selected checkpoint.')
    solution=csv_rows(LEGACY/'metadata_snapshot/solution.csv')
    values=load(run/'test_predictions.json');pred={r['id']:float(r['score']) for r in values}
    expected={r['id'] for r in solution}
    if len(values)!=1344 or len(pred)!=1344 or set(pred)!=expected:raise ValueError('Require exactly1344 unique official test IDs.')
    if not all(math.isfinite(p) and 0<=p<=1 for p in pred.values()):raise ValueError('Nonfinite/out-of-range test score')
    submission=csv_rows(run/'submission.csv')
    if len(submission)!=1344 or len({r['id'] for r in submission})!=1344 or {r['id'] for r in submission}!=expected:
        raise ValueError('Submission IDs differ from predictions.')
    if any(float(r['score'])!=pred[r['id']] for r in submission):raise ValueError('Submission/prediction mismatch')
    saved=load(run/'official_test_metrics.json');measured={}
    for usage in ('Public','Private','Combined'):
        aps=[];aucs=[]
        for group in range(3):
            sub=[r for r in solution if int(r['group'])==group and (usage=='Combined' or r['Usage']==usage)]
            ap,auc=ap_auc([int(r['target']) for r in sub],[pred[r['id']] for r in sub])
            aps.append(ap);aucs.append(auc)
            if usage!='Combined':
                rows={int(r['group']):r for r in saved[usage]['groups']}
                if set(rows)!={0,1,2} or rows[group]['n']!=len(sub):raise ValueError('Saved metric group counts differ')
                if abs(rows[group]['AP']-ap)>1e-12 or abs(rows[group]['AUC']-auc)>1e-12:
                    raise ValueError('Saved AP/AUC does not match raw predictions')
        measured[usage]=dict(mAP=sum(aps)/3,mAUC=sum(aucs)/3)
        if usage!='Combined' and (abs(saved[usage]['mAP']-sum(aps)/3)>1e-12 or
                                 abs(saved[usage]['mAUC_grouped']-sum(aucs)/3)>1e-12):
            raise ValueError('Saved aggregate metric does not match predictions')
    val=load(run/'validation_predictions.json');ids=[r['id'] for r in val]
    data=load(run.parent/'prepared_nexar/data_manifest.json')
    if len(ids)!=300 or len(set(ids))!=300 or set(ids)!=set(data['validation_ids']):
        raise ValueError('Validation prediction IDs do not match the 300-video split')
    for row in val:
        if len(row['pred'])!=30 or not all(math.isfinite(float(p)) and 0<=p<=1 for p in row['pred']):
            raise ValueError('Invalid validation risk sequence')
    val_metrics=load(run/'validation_metrics.json')
    selected=max(history,key=lambda row:row['metrics']['mAUC@'])['metrics']
    for name in ('mAUC@','mAP@'):
        a=val_metrics.get(name);b=selected.get(name)
        if isinstance(a,bool) or not isinstance(a,(float,int)) or not math.isfinite(a) or b is None or abs(a-b)>1e-10:
            raise ValueError('Validation metrics differ from selected epoch: '+name)
    return measured


def seal(run,inference_rerun=False):
    run=Path(run);measured=evaluation(run)
    status=load(run/'training_status.json');epoch=status['completed_epochs']
    names=list(EVAL_FILES)+['run_manifest.json','training_status.json','validation_history.jsonl',
                           status['best_checkpoint'],'resume_latest.pth','rng_epoch%d_rank0.pth'%epoch,
                           'safe_resume_snapshot.zip']
    value=dict(schema='v6-verified-artifact-seal-v1',seed=load(run/'run_manifest.json')['contract']['seed'],
               files={name:digest(run/name) for name in names},measured=measured,
               inference_rerun=bool(inference_rerun),
               scope='stored prediction arithmetic and artifact integrity; GPU inference only if explicitly rerun')
    atomic_json(run/'verified_artifacts.json',value)
    return value


def preserve_evaluation(run):
    """Keep prior predictions and metrics before an explicitly requested GPU reevaluation."""
    run=Path(run)
    names=[n for n in EVAL_FILES+('verified_artifacts.json',) if (run/n).is_file()]
    if not names:return None
    folder=Path(tempfile.mkdtemp(prefix='evaluation_before_rerun_',dir=str(run.parent)))
    for name in names:shutil.copy2(run/name,folder/name)
    atomic_json(folder/'preserved_sha256.json',{n:digest(folder/n) for n in names})
    return folder


def is_evaluated(run):
    run=Path(run)
    try:
        training(run,complete=True)
        saved=load(run/'verified_artifacts.json')
        if saved.get('schema')!='v6-verified-artifact-seal-v1':return False
        expected=set(EVAL_FILES)|{'run_manifest.json','training_status.json','validation_history.jsonl',
                load(run/'training_status.json')['best_checkpoint'],'resume_latest.pth',
                'rng_epoch50_rank0.pth','safe_resume_snapshot.zip'}
        if set(saved.get('files',{}))!=expected:return False
        return all(digest(run/name)==value for name,value in saved['files'].items())
    except (OSError,ValueError,KeyError,zipfile.BadZipFile):return False


def restore_snapshot(run,expected_protocol,expected_seed):
    """Validate and stage every member BEFORE publishing any recovered file; status last."""
    run=Path(run);snapshot=run/'safe_resume_snapshot.zip'
    if not snapshot.exists():return False
    with zipfile.ZipFile(snapshot) as archive:
        if archive.testzip() is not None:raise ValueError('Snapshot CRC failure')
        status=json.loads(archive.read('training_status.json'));contract=json.loads(archive.read('run_manifest.json'))['contract']
        epoch=status['completed_epochs'];best=basename(status['best_checkpoint'])
        if type(epoch) is not int or not 1<=epoch<=50:raise ValueError('Invalid snapshot epoch')
        if contract.get('protocol')!=expected_protocol or contract.get('seed')!=expected_seed:
            raise ValueError('Snapshot protocol/seed mismatch')
        expected={'resume_latest.pth',best,'rng_epoch%d_rank0.pth'%epoch,
                  'training_status.json','run_manifest.json','validation_history.jsonl'}
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=expected:
            raise ValueError('Unsafe/missing/duplicate snapshot members')
        for name in expected:basename(name)
        with tempfile.TemporaryDirectory(prefix='.verified_restore_',dir=str(run)) as temporary:
            stage=Path(temporary)
            for name in sorted(expected):
                with archive.open(name) as src,(stage/name).open('wb') as dst:
                    for b in iter(lambda:src.read(4<<20),b''):dst.write(b)
            for name in sorted(expected-{'training_status.json'}):os.replace(stage/name,run/name)
            os.replace(stage/'training_status.json',run/'training_status.json')
    return True
