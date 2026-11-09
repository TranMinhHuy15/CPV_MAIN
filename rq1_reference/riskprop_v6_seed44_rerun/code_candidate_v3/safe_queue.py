"""V6 three-seed queue with strict completion gates and the safe child launcher."""
import sys
from pathlib import Path
from integrity import LEGACY,ROOT,training,is_evaluated,restore_snapshot
sys.path.insert(0,str(LEGACY))
import run_three_seeds as queue

original_saved_state=queue.saved_state
original_command_for=queue.command_for
original_run_checked=queue.run_checked

def saved_state(root,seed):
    run,status,contract=original_saved_state(root,seed)
    if status:training(run,complete=False)
    return run,status,contract

def command_for(*args,**kwargs):
    command=original_command_for(*args,**kwargs)
    if Path(command[1]).name!='run_vast.py':raise ValueError('Unexpected launcher')
    command[1]=str(ROOT/'safe_entry.py')
    return command

def install():
    queue.saved_state=saved_state
    queue.evaluated=lambda run,status:bool(status and status['complete'] and is_evaluated(run))
    queue.command_for=command_for
    queue.restore_snapshot=restore_snapshot

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-root',default='/workspace/riskprop_v6_vast_3seeds')
    a,_=p.parse_known_args(argv)
    def checked(command,**kwargs):
        if len(command)>1 and Path(command[1]).name=='test_safety.py':
            command=[command[0],str(ROOT/'linux_preflight.py'),'--workspace',str(Path(a.output_root).resolve())]
        return original_run_checked(command,**kwargs)
    queue.run_checked=checked
    install()
    queue.main(argv)

if __name__=='__main__':main()
