"""Harden original V6 launch operations while preserving the original scientific contract."""
import json
import os
from pathlib import Path
import sys
import time

from integrity import (ROOT,LEGACY,training,seal,restore_snapshot,digest,atomic_json,load,preserve_evaluation)
sys.path.insert(0,str(LEGACY))
import repro
import run_vast
from process_control import exclusive_job
_installed=False


def validate_settings_before_write(path,value):
    path=Path(path)
    existing=path.parent/'run/run_manifest.json'
    if existing.exists() and load(existing).get('contract')!=repro.run_contract(value):
        raise ValueError('New settings differ from saved contract; original settings preserved. Use NEW workspace.')


def install():
    global _installed
    if _installed:return
    old_write=Path.write_text
    def safe_write(path,data,*args,**kwargs):
        if path.name=='settings.json':
            value=json.loads(data)
            if isinstance(value,dict) and {'workspace','seed','epochs'}.issubset(value):
                validate_settings_before_write(path,value)
                atomic_json(path,value)
                return len(data)
        return old_write(path,data,*args,**kwargs)
    Path.write_text=safe_write
    run_vast.restore_snapshot=restore_snapshot
    old_execute=run_vast.execute
    def checked_execute(command,log_path,env=None):
        evaluating=len(command)>2 and Path(command[1]).name=='repro.py' and command[2]=='evaluate'
        if evaluating:
            settings=load(command[command.index('--settings')+1]);run=Path(settings['workspace'])/'run'
            training(run,complete=True)
            preserve_evaluation(run)
        elif len(command)>2 and Path(command[1]).name=='repro.py' and command[2]=='train':
            settings=load(command[command.index('--settings')+1]);run=Path(settings['workspace'])/'run'
            if (run/'run_manifest.json').exists():training(run,complete=False)
        result=old_execute(command,log_path,env)
        if evaluating:seal(run,inference_rerun=True)
        return result
    run_vast.execute=checked_execute
    import checkpoint_safety
    checkpoint_safety.restore_snapshot=restore_snapshot
    _installed=True


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--workspace',default='/workspace/riskprop_v6_seed44_rerun')
    parser.add_argument('--runtime-root',default='/workspace/riskprop_v6_vast_runtime')
    parser.add_argument('--data-root',required=True)
    args,_=parser.parse_known_args(argv)
    workspace=Path(args.workspace).resolve();runtime=Path(args.runtime_root).resolve();data=Path(args.data_root).resolve()
    if any(p==data or data in p.parents for p in (workspace,runtime)):
        raise ValueError('Workspace/runtime cannot be inside the dataset')
    if workspace==ROOT or ROOT in workspace.parents or workspace in ROOT.parents:
        raise ValueError('Workspace must be separate from package directory')
    if workspace==runtime or workspace in runtime.parents or runtime in workspace.parents:
        raise ValueError('Workspace and runtime must be separate directories')
    from verify_package import verify
    verify()
    with exclusive_job(workspace/'.safe_job.lock'),exclusive_job(runtime/'.safe_runtime.lock'),exclusive_job(ROOT/'.package_job.lock'):
        if sys.platform!='linux':raise RuntimeError('Training/evaluation requires Linux; Windows supports audit/tests only')
        verify()  # re-check hashes while holding the locks (closes the start-up race)
        from linux_preflight import check
        check(workspace)
        install()
        receipt={'wrapper_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},
                 'scope':'Operational wrapper only; legacy scientific contract unchanged'}
        atomic_json(workspace/'safety_wrapper_receipt.json',receipt)
        run_vast.main(argv)

if __name__=='__main__':main()
