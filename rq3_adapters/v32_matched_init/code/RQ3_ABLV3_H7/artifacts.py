"""Compact completed owned trials; originals remain untouched and inference reproducible."""
from pathlib import Path
from protocol import load,save,sha,digest,history_selection,METRICS


def verify_trial(workspace,row=None):
    workspace=Path(workspace);row=load(workspace/'trial_complete.json') if row is None else row
    seal=load(workspace/'compact_manifest.json')
    if seal.get('complete') is not True or row.get('complete') is not True:raise ValueError('Incomplete candidate export.')
    required={'selected_adapter.npz','run/run_manifest.json','run/training_status.json','run/validation_history.jsonl'}
    from diagnostics import FILES
    allowed=required|{'run/'+n for n in FILES}
    if (not required.issubset(set(seal.get('files',{}))) or not set(seal.get('files',{})).issubset(allowed) or seal.get('original_full_commit_crc_checked') is not True
            or seal.get('frozen_weights_equal_original') is not True or Path(row['workspace']).resolve()!=workspace.resolve()):
        raise ValueError('Wrong candidate seal inventory/provenance.')
    for name,expected in seal['files'].items():
        path=(workspace/name).resolve()
        if workspace.resolve() not in path.parents or not path.is_file() or sha(path)!=expected:raise ValueError('Candidate artifact changed: '+name)
    selected=history_selection(workspace/'run')
    contract=load(workspace/'run/run_manifest.json')['contract']
    if contract.get('protocol','').startswith('rq3-ablation-v3'):
        from diagnostics import verify_validation
        verify_validation(workspace/'run')
        if not allowed.issubset(set(seal['files'])):raise ValueError('V3 diagnostic hashes missing.')
    owner=load(workspace/'owned_trial.json')
    if owner.get('study_contract_sha256')!=sha(workspace.resolve().parents[3]/'study_contract.json'):
        raise ValueError('Candidate belongs to another study.')
    if (digest(contract)!=row['contract_sha256'] or selected['epoch']!=row['selected_epoch']
            or {k:selected['metrics'][k] for k in METRICS}!=row['validation_metrics']
            or selected['validation_threshold']!=row['validation_threshold']
            or contract.get('seed')!=row['seed'] or contract.get('arm')!=row['arm']
            or contract.get('hyperparameters',{}).get('id')!=row['candidate']
            or seal['selected_full_checkpoint_sha256']!=row['checkpoint_sha256']
            or seal['files']['selected_adapter.npz']!=row['adapter_sha256']):raise ValueError('Candidate completion metadata differs.')
    return row


def compact(workspace,base,cfg):
    import numpy as np
    import torch
    from training_state import assert_committed
    workspace=Path(workspace).resolve();run=workspace/'run'
    out=Path(cfg['output_root']).resolve()
    relative=workspace.relative_to(out)
    if len(relative.parts)!=4 or relative.parts[0]!='trials' or run.is_symlink():raise ValueError('Refuse compacting outside an owned trial.')
    status=load(run/'training_status.json');assert_committed(run,status,require_complete=True)
    selected=history_selection(run);contract=load(run/'run_manifest.json')['contract']
    if load(workspace/'owned_trial.json')['study_contract_sha256']!=sha(out/'study_contract.json'):
        raise ValueError('Unowned trial; no compaction.')
    checkpoint=run/selected['checkpoint']
    def state(path):
        value=torch.load(str(path),map_location='cpu',weights_only=False);value=value.get('state_dict',value)
        return {(k[7:] if k.startswith('module.') else k):v for k,v in value.items()}
    saved,baseline=state(checkpoint),state(base['checkpoint'])
    prefix='cls_head.adapter.'
    frozen={k:v for k,v in saved.items() if not k.startswith(prefix)}
    if set(frozen)!=set(baseline) or any(not torch.equal(v,baseline[k]) for k,v in frozen.items()):raise ValueError('Frozen baseline changed during tuning.')
    adapter={k[len(prefix):]:v.detach().cpu().numpy() for k,v in saved.items() if k.startswith(prefix)}
    if not adapter or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in adapter.values()):raise ValueError('Invalid adapter export.')
    path=workspace/'selected_adapter.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**adapter)
    tmp.replace(path)
    row=dict(complete=True,seed=contract['seed'],arm=contract['arm'],candidate=contract['hyperparameters']['id'],
             selected_epoch=selected['epoch'],checkpoint_sha256=sha(checkpoint),
             validation_metrics={k:selected['metrics'][k] for k in METRICS},validation_threshold=selected['validation_threshold'],
             contract_sha256=digest(contract),baseline_checkpoint_sha256=base['checkpoint_sha256'],
             workspace=str(workspace),adapter_sha256=sha(path))
    seal=dict(complete=True,selected_full_checkpoint_sha256=row['checkpoint_sha256'],
              original_snapshot_sha256=sha(run/'safe_resume_snapshot.zip'),original_full_commit_crc_checked=True,
              frozen_weights_equal_original=True,files={n:sha(workspace/n) for n in
              ('selected_adapter.npz','run/run_manifest.json','run/training_status.json','run/validation_history.jsonl')})
    if contract.get('protocol','').startswith('rq3-ablation-v3'):
        from diagnostics import verify_validation,FILES
        verify_validation(run)
        seal['files'].update({'run/'+n:sha(run/n) for n in FILES})
    save(workspace/'compact_manifest.json',seal);save(workspace/'trial_complete.json',row)
    verify_trial(workspace,row)
    # Explicitly configured retention policy. Only generated files in this checked trial/run.
    if cfg['compact_completed_trials']:
        for p in list(run.glob('*.pth'))+[run/'safe_resume_snapshot.zip']:
            if p.is_symlink() or p.resolve().parent!=run.resolve():raise ValueError('Refuse pruning a linked/outside artifact.')
            if p.is_file():p.unlink()
    print('COMPACT module=%s seed=%d config=%s selected_epoch=%d adapter_MiB=%.2f full_original_snapshot_verified=True'%
          (row['arm'],row['seed'],row['candidate'],row['selected_epoch'],path.stat().st_size/2**20),flush=True)
    return row


def load_adapter(workspace,device):
    import numpy as np
    import torch
    from models import build_adapter
    workspace=Path(workspace);row=verify_trial(workspace)
    c=load(workspace/'run/run_manifest.json')['contract']['hyperparameters']
    adapter=build_adapter(row['arm'],width=c['width'],dropout=c['dropout'])
    with np.load(workspace/'selected_adapter.npz',allow_pickle=False) as z:
        state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    adapter.load_state_dict(state,strict=True)
    return adapter.to(device=device).eval().requires_grad_(False)
