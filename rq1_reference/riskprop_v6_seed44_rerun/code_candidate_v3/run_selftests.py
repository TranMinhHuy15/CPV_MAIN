"""Self-contained CPU regression suite. GPU/Linux checks are separate mandatory gates."""
import argparse,io,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(report=None):
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    print(output.getvalue())
    data={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
          'passed':result.wasSuccessful(),'scope':'CPU fixtures/artifact/backup/wrapper regressions; not CUDA or Linux process-tree certification'}
    if report:Path(report).write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');a=p.parse_args()
    sys.exit(0 if run(a.report)['passed'] else 1)
