"""Explicit preservation of a failed attempt before its first committed epoch."""
import argparse
from pathlib import Path
from integrity import PROTOCOL,load
from training_state import archive_uncommitted
import sys
from integrity import LEGACY
sys.path.insert(0,str(LEGACY))
from process_control import exclusive_job

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run).resolve()
    contract=load(run/'run_manifest.json')['contract']
    if contract.get('protocol')!=PROTOCOL or contract.get('seed') not in (42,43,44):
        raise ValueError('Wrong experiment; no files changed')
    with exclusive_job(run.parent/'.safe_job.lock'):
        print('Preserved failed attempt:',archive_uncommitted(run))
