"""Run reproducible CPU checks and retain real logs; no source modification."""
import argparse
import ast
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,help='Directory outside source package')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=Path(args.output).resolve()
    if output==root or root in output.parents:
        raise ValueError('Verification logs must be outside the packaged source.')
    output.mkdir(parents=True,exist_ok=True)
    parsed=[]
    for path in root.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
        parsed.append(path.name)
    checks=[]
    for script in ('test_adalea.py','test_full_eval.py','test_process_safety.py'):
        start=time.monotonic()
        result=subprocess.run([sys.executable,'-X','utf8','-u',str(root/script)],cwd=str(root),
            text=True,capture_output=True,encoding='utf-8',errors='replace',timeout=600)
        text=result.stdout+'\n'+result.stderr
        log=output/(script+'.log')
        log.write_text(text,encoding='utf-8')
        checks.append(dict(script=script,exit_code=result.returncode,elapsed_s=time.monotonic()-start,log=str(log)))
        print(script,'PASS' if result.returncode==0 else 'FAIL',flush=True)
        if result.returncode:
            print(text[-6000:],flush=True)
            break
    report=dict(passed=len(checks)==3 and all(c['exit_code']==0 for c in checks),
        python=sys.version,platform=platform.platform(),parsed_python_files=parsed,checks=checks,
        gpu_rtx4080_verified=False,real_dataset_end_to_end_verified=False,real_legacy_checkpoint_verified=False,
        purpose='CPU fixtures only. No paper-performance claim. No paid Vast run.')
    (output/'LOCAL_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']: sys.exit(1)


if __name__=='__main__': main()
