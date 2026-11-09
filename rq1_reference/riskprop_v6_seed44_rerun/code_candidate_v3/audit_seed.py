"""Verify or reseal a completed V6 run without retraining or modifying its weights."""
import argparse,json
from pathlib import Path
from integrity import training,evaluation,seal,digest,atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--seal',action='store_true',help='Write artifact hashes only after every strict check passes.')
    args=parser.parse_args();run=Path(args.run).resolve()
    status,contract,history=training(run,complete=True)
    metrics=evaluation(run)
    report=dict(seed=contract['seed'],status=status,history_epochs=[r['epoch'] for r in history],
                best_checkpoint_sha256=digest(run/status['best_checkpoint']),metrics=metrics,
                GPU_inference_rerun=False,weights_changed=False,
                warning='Parent RNG restored does not guarantee persistent DataLoader worker RNG equivalence.')
    if args.seal:report['artifact_seal']=seal(run,inference_rerun=False)
    atomic_json(Path(args.report),report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_seal'},indent=2))

if __name__=='__main__':main()
