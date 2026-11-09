"""Re-score an already completed Vast V6 prediction file with the same pooled test evaluator.

Writes a NEW report only; never mutates the V6 run/checkpoint or trains a model.
"""
import argparse
import json
from pathlib import Path

from evaluate import grouped_test_metrics
from repro import read_csv, write_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v6-run', required=True, help='Completed Vast V6 run directory containing test_predictions.json.')
    parser.add_argument('--output', required=True, help='New report file outside the V6 run directory.')
    args = parser.parse_args()
    run, output = Path(args.v6_run).resolve(), Path(args.output).resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError('Use a new report path OUTSIDE the original V6 run; no overwrite.')
    status = json.loads((run/'training_status.json').read_text())
    contract = json.loads((run/'run_manifest.json').read_text())['contract']
    if not status.get('complete') or status.get('completed_epochs') != 50:
        raise ValueError('Only a completed 50-epoch run may be scored.')
    if contract.get('protocol') != 'riskprop-v6-vast-batch2-eff16-amp-single-gpu-3seed-terminal-v3':
        raise ValueError('This comparison requires the specified Vast V6 control, not Kaggle/V7/another batch profile.')
    predictions = run/'test_predictions.json'
    metrics = grouped_test_metrics(read_csv(Path(__file__).parent/'metadata_snapshot/solution.csv'),
                                   json.loads(predictions.read_text()))
    write_json(output, dict(source_run=str(run), source_predictions_sha256=sha256(predictions),
        source_contract=contract, source_best_checkpoint=status['best_checkpoint'], test_metrics=metrics,
        note='Pooled 1344-video grouped scalar scoring. No retraining or checkpoint reselection; test mTTA is undefined.'))
    print('V6 derived report:', output)


if __name__ == '__main__':
    main()
