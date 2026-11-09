"""Mandatory real Linux process-tree/lock/snapshot checks before GPU work."""
import io,os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'legacy_v6'))
from training_state import atomic_json
def check(workspace):
    if os.name!='posix' or sys.platform!='linux':raise RuntimeError('Run this preflight on rented Linux, not Windows')
    import test_safety
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(test_safety.SafetyChecks)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    atomic_json(Path(workspace)/'linux_safety_preflight.json',dict(tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),output=stream.getvalue()))
    if not result.wasSuccessful() or result.skipped:
        why=[str(t)+': '+str(r) for t,r in result.skipped]+[str(t) for t,_ in result.errors+result.failures]
        raise RuntimeError('Linux process cleanup/lock gate failed; GPU job not started. Cause: '+'; '.join(why)+
                           ' (skips usually mean a restricted container: rent an instance with full process/flock access)')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args()
    check(a.workspace)
