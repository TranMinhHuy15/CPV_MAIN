"""Seed44 rerun with explicit fresh/resume/evaluate modes and unchanged V6 recipe."""
import argparse,os
from pathlib import Path
import safe_entry
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','train','resume','evaluate'])
    p.add_argument('--data-root',default=os.environ.get('DATA','/workspace/Nexar_HF'))
    p.add_argument('--workspace',default=os.environ.get('WS','/workspace/riskprop_v6_seed44_rerun'))
    p.add_argument('--runtime-root',default=os.environ.get('RUNTIME','/workspace/riskprop_v6_vast_runtime'))
    p.add_argument('--hours',type=float,default=float(os.environ.get('HOURS','40')))
    a=p.parse_args(argv)
    command=['--data-root',a.data_root,'--workspace',a.workspace,'--runtime-root',a.runtime_root,'--seed','44','--hours',str(a.hours)]
    if a.mode=='train':command+=['--train']
    elif a.mode=='resume':command+=['--train','--resume-root',str(Path(a.workspace).resolve()/'run')]
    elif a.mode=='evaluate':command+=['--evaluate-only']
    safe_entry.main(command)
if __name__=='__main__':main()
