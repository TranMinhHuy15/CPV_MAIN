"""Linux/NVIDIA runner: preflight -> setup if needed -> extract -> pilot+18runs -> test -> analysis."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from protocol import load
from console import Log,command

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('phase',choices=('preflight','setup','extract','pilot','tune','test','analyze','all'))
    args=parser.parse_args();config=Path(args.config).resolve()
    subprocess.run([sys.executable,str(HERE/'run.py'),'--config',str(config),'preflight'],check=True)
    if args.phase=='preflight':return
    cfg=load(config);out=Path(cfg['output_root']);log=Log(out/'launcher_logs')
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    def phase(name):command([sys.executable,HERE/'run.py','--config',config,name],name,log,env)
    runtime=Path(cfg['runtime_root'])/'venv/bin/python'
    def analyze():
        if not runtime.is_file():raise FileNotFoundError('Run setup first: '+str(runtime))
        source=Path(load(out/'prepared.json')['source_root'])
        command([runtime,HERE/'bootstrap_paired.py','--output-root',out,'--solution',source/'base_h7/metadata_snapshot/solution.csv'],'bootstrap',log,env)
    if args.phase=='all':
        if not runtime.is_file():phase('setup')
        phase('extract');phase('tune');phase('test');analyze()
    elif args.phase=='analyze':analyze()
    else:phase(args.phase)
    print('DONE '+str(out),flush=True)


if __name__=='__main__':main()
