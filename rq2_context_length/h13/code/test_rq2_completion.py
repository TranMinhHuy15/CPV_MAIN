"""Synthetic full lifecycle/1344-file checks; these numbers are NOT model results."""
import csv
from pathlib import Path
import tempfile
import unittest

from repro import run_contract
from rq2_profile import REQUIRED_TEST_METRICS, scientific_profile, protocol, validate_metrics, VARIANTS
from run_rq2 import HERE, evaluated, package_digest, seal_evaluation, summarize, save


def fixture(root,variant=VARIANTS[0],seed=42,finalize=True):
    run=root/variant/('seed_%d'%seed)/'run'
    curves=run/'test_capture/curves'
    curves.mkdir(parents=True)
    checkpoint=run/'best_fixture.pth'
    checkpoint.write_bytes(b'SYNTHETIC_NOT_WEIGHTS')
    contract=run_contract(dict(seed=seed,epochs=50,batch_per_gpu=2,effective_batch_size=16,world_size=1,amp=True))
    contract.update(protocol=protocol(variant),rq2=scientific_profile(variant))
    status=dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint=checkpoint.name)
    save(run/'run_manifest.json',dict(contract=contract))
    save(run/'training_status.json',status)
    metrics={name:.5 for name in REQUIRED_TEST_METRICS}
    fingerprint=dict(training_contract=contract,checkpoint_sha256=package_digest(checkpoint))
    report=dict(complete=True,test_count=1344,selected_checkpoint=checkpoint.name,
        checkpoint_sha256=fingerprint['checkpoint_sha256'],capture_fingerprint=fingerprint,summary_metrics=metrics)
    if not finalize:
        save(run/'rq2_test_report.json',report)
        return run,status,report
    with (HERE/'metadata_snapshot/solution.csv').open(encoding='utf-8-sig',newline='') as stream:
        ids=[r['id'] for r in csv.DictReader(stream)]
    for video_id in ids:
        save(curves/(video_id+'.json'),dict(id=video_id,scores=[.5],fixture=True))
    report['curve_file_sha256']={video_id:package_digest(curves/(video_id+'.json')) for video_id in ids}
    save(run/'rq2_test_report.json',report)
    save(run/'test_predictions.json',[dict(id=video_id,score=.5) for video_id in ids])
    save(run/'test_capture/capture_contract.json',fingerprint)
    save(run/'test_capture/capture_manifest.json',dict(complete=True,test_count=1344,checkpoint_sha256=fingerprint['checkpoint_sha256']))
    with (run/'rq2_test_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['metric','value','paper_equivalent'])
        writer.writerows([name,value,False] for name,value in metrics.items())
    seal_evaluation(run,fingerprint['checkpoint_sha256'])
    return run,status,report


class CompletionTests(unittest.TestCase):
    def test_partial_finalization_cannot_skip_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            run,status,_=fixture(Path(folder),finalize=False)
            self.assertFalse(evaluated(run,status))

    def test_complete_artifacts_and_all1344_curves_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            run,status,_=fixture(Path(folder))
            self.assertTrue(evaluated(run,status))

    def test_missing_or_modified_csv_requires_report_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            run,status,_=fixture(Path(folder))
            (run/'rq2_test_metrics.csv').write_text('INTERRUPTED_OR_CHANGED',encoding='utf-8')
            self.assertFalse(evaluated(run,status))

    def test_changed_curve_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            run,status,_=fixture(Path(folder))
            next((run/'test_capture/curves').glob('*.json')).write_text('CHANGED',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'risk curve changed'):
                evaluated(run,status)

    def test_missing_curve_cannot_count_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            run,status,_=fixture(Path(folder))
            next((run/'test_capture/curves').glob('*.json')).unlink()
            self.assertFalse(evaluated(run,status))

    def test_incomplete_or_nonfinite_metric_schema_is_rejected(self):
        for invalid in ({'mAP':.5},{name:float('nan') for name in REQUIRED_TEST_METRICS}):
            with self.assertRaises(ValueError):
                validate_metrics(invalid)

    def test_complete_six_run_aggregate_has28_metrics_per_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for variant in VARIANTS:
                for seed in (42,43,44):
                    fixture(root,variant,seed)
            result=summarize(root)
            self.assertTrue(result['complete_study'])
            for arm in result['mean_sample_sd'].values():
                self.assertEqual(set(arm),set(REQUIRED_TEST_METRICS))
                self.assertTrue(all(v['n']==3 and v['sample_sd']==0 for v in arm.values()))


if __name__=='__main__':
    unittest.main(verbosity=2)
