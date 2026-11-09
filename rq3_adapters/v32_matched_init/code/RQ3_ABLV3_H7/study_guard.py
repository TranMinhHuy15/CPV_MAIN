"""Bind runtime code and the analysis plan before training; reject stale/cross-study inputs."""
import json
from pathlib import Path
from protocol import load,save,sha,digest,ARMS,SEEDS

VERSION='rq3-ablation-v3.2-matched-six-arms'
PLAN='PREREGISTERED_ANALYSIS.md'


def code_hashes(root):
    root=Path(root)
    paths=list(root.glob('*.py'))+list((root/'legacy').glob('*.py'))+[root/PLAN]
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def ensure_output(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        path=out/'study_contract.json'
        if not path.exists():
            # Only pinned-source staging may exist before the study is bound.
            source = out / 'checked_source'
            entries = [p for p in out.iterdir() if p != source]
            if (source.is_dir() and not source.is_symlink()
                    and all(p.is_file() and not p.is_symlink()
                            and p.match('.study_contract.json.*.tmp') for p in entries)):
                return
        elif load(path).get('study_version') == VERSION:
            return
        raise ValueError('Use a NEW empty V3 output_root; keep old V2 results unchanged.')


def plan_identity(root):return dict(file=PLAN,sha256=sha(Path(root)/PLAN))


def shared_analysis_hashes(root):
    root=Path(root)
    return {name:sha(root/name) for name in ('ablation_stats.py','analyze_cross_base.py','study_guard.py')}


def require_study(out,root):
    out,root=Path(out),Path(root)
    contract=load(out/'study_contract.json')
    if (contract.get('study_version')!=VERSION or contract.get('code')!=code_hashes(root)
            or contract.get('analysis_plan')!=plan_identity(root)):
        raise ValueError('Study code/analysis plan changed. Preserve outputs and use a NEW output_root.')
    if sha(out/'analysis_plan.md')!=contract['analysis_plan']['sha256']:
        raise ValueError('Archived analysis plan changed.')
    return contract


def archive_plan(out,root):
    out,root=Path(out),Path(root);data=(root/PLAN).read_bytes();target=out/'analysis_plan.md'
    if target.exists() and target.read_bytes()!=data:raise ValueError('Archived plan identity changed.')
    if not target.exists():
        from recovery import atomic_bytes
        atomic_bytes(target, data)


def require_worker_inputs(out,root,cfg,prepared):
    contract=require_study(out,root)
    if (cfg!=contract.get('config') or prepared.get('baselines')!=contract.get('baselines')
            or str(Path(prepared['source_root']).resolve())!=contract.get('source_root')):
        raise ValueError('Worker config/prepared inputs changed after preflight; preserve the bound study.')
    return contract


def require_evaluation(out,root):
    from reporting import checked_report
    out=Path(out);contract=require_study(out,root)
    selection=load(out/'selection_frozen_before_test.json');lock=digest(selection)
    complete=load(out/'comparison/TEST_COMPARISON_COMPLETE.json')
    if (selection['study_contract_sha256']!=sha(out/'study_contract.json')
            or selection['runtime_identity_sha256']!=sha(out/'runtime_identity.json')
            or complete.get('complete') is not True or complete.get('selection_lock_sha256')!=lock
            or complete.get('seeds')!=list(SEEDS) or set(selection.get('arms',[]))!=set(ARMS)):
        raise ValueError('Incomplete or stale six-arm evaluation.')
    for name,expected in complete['files'].items():
        if sha(out/'comparison'/name)!=expected:raise ValueError('Comparison artifact changed: '+name)
    for seed in SEEDS:
        for arm in ('b0',)+ARMS:
            report=checked_report(out/'test/runs'/arm/('seed_%d'%seed)/'run',seed,arm,lock)
            if arm=='b0':
                base=next(iter(selection['baselines']))
                expected=selection['baselines'][base][str(seed)]['checkpoint_sha256']
            else:
                expected=next(r for r in selection['chosen'][arm]['selected'] if r['seed']==seed)['adapter_sha256']
            if report['checkpoint_sha256']!=expected:raise ValueError('Scored weights differ from frozen selection.')
    return contract


def finish_bootstrap(out,metadata):
    out=Path(out);comparison=out/'comparison'
    save(comparison/'BOOTSTRAP_COMPLETE.json',dict(complete=True,metadata=metadata,
         files={n:sha(comparison/n) for n in ('bootstrap_draws.npz','paired_bootstrap.csv','paired_bootstrap_decisions.json')}))


def require_bootstrap(path,expected_base,plan_sha):
    path=Path(path);comparison=path if path.name=='comparison' else path/'comparison'
    if path.suffix=='.npz':comparison=path.parent
    marker=load(comparison/'BOOTSTRAP_COMPLETE.json');metadata=marker.get('metadata',{})
    if (marker.get('complete') is not True or metadata.get('study_version')!=VERSION
            or metadata.get('base')!=expected_base or metadata.get('analysis_plan_sha256')!=plan_sha
            or metadata.get('analysis_code_sha256')!=shared_analysis_hashes(Path(__file__).resolve().parent)):
        raise ValueError('Bootstrap is from another base/version/plan.')
    for name,expected in marker['files'].items():
        if sha(comparison/name)!=expected:raise ValueError('Bootstrap artifact changed: '+name)
    import numpy as np
    with np.load(comparison/'bootstrap_draws.npz',allow_pickle=False) as z:
        if json.loads(str(z['metadata_json']))!=metadata:raise ValueError('Draw metadata differs from completion marker.')
        for key in z.files:
            if key.startswith('draw|') and (z[key].shape!=(metadata['resamples'],) or not np.isfinite(z[key]).all()):
                raise ValueError('Wrong/nonfinite bootstrap draws.')
    return comparison/'bootstrap_draws.npz',metadata
