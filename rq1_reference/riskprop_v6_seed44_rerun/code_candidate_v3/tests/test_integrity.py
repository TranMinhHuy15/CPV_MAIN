import json,random,shutil,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import integrity as guard
from training_state import atomic_json,archive_uncommitted
import safe_entry,safe_queue

class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name)/'seed_44'
        shutil.copytree(guard.ROOT/'evidence/seed_44',self.workspace)
        self.run=self.workspace/'run'
        self.status=guard.load(self.run/'training_status.json')
        for name in ('resume_latest.pth',self.status['best_checkpoint'],'rng_epoch50_rank0.pth'):
            (self.run/name).write_bytes(('fixture '+name).encode())
        self.snapshot()

    def snapshot(self):
        names=('training_status.json','run_manifest.json','validation_history.jsonl',
               'resume_latest.pth',self.status['best_checkpoint'],'rng_epoch50_rank0.pth')
        with zipfile.ZipFile(self.run/'safe_resume_snapshot.zip','w') as archive:
            for name in names:archive.write(self.run/name,name)

    def test_complete_fixture(self):guard.training(self.run,complete=True)
    def test_predictions_recompute(self):
        self.assertAlmostEqual(guard.evaluation(self.run)['Combined']['mAP'],.7962148361915308)
    def test_missing_checkpoint_rejected(self):
        (self.run/self.status['best_checkpoint']).unlink()
        with self.assertRaises(ValueError):guard.training(self.run)
    def test_checkpoint_corruption_rejected(self):
        (self.run/self.status['best_checkpoint']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):guard.training(self.run)
    def test_missing_prediction_rejected(self):
        (self.run/'test_predictions.json').unlink()
        with self.assertRaises(ValueError):guard.evaluation(self.run)
    def test_duplicate_prediction_rejected(self):
        data=guard.load(self.run/'test_predictions.json');data[0]=data[1]
        atomic_json(self.run/'test_predictions.json',data)
        with self.assertRaises(ValueError):guard.evaluation(self.run)
    def test_nan_score_rejected(self):
        data=guard.load(self.run/'test_predictions.json');data[0]['score']=float('nan')
        (self.run/'test_predictions.json').write_text(json.dumps(data))
        with self.assertRaises(ValueError):guard.evaluation(self.run)
    def test_wrong_metric_rejected(self):
        data=guard.load(self.run/'official_test_metrics.json');data['Public']['mAP']+=.01
        atomic_json(self.run/'official_test_metrics.json',data)
        with self.assertRaises(ValueError):guard.evaluation(self.run)
    def test_submission_mismatch_rejected(self):
        p=self.run/'submission.csv';p.write_text(p.read_text().replace('0.49340346455574036','0.1'))
        with self.assertRaises(ValueError):guard.evaluation(self.run)
    def test_history_duplicate_rejected(self):
        p=self.run/'validation_history.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaises(ValueError):guard.training(self.run)
    def test_seal_detects_changes(self):
        guard.seal(self.run);self.assertTrue(guard.is_evaluated(self.run))
        p=self.run/'test_predictions.json';p.write_text(p.read_text()+' ')
        self.assertFalse(guard.is_evaluated(self.run))
    def test_unsealed_is_not_complete(self):self.assertFalse(guard.is_evaluated(self.run))
    def test_restore_replaces_corruption(self):
        (self.run/self.status['best_checkpoint']).write_bytes(b'bad')
        self.assertTrue(guard.restore_snapshot(self.run,guard.PROTOCOL,44))
        guard.training(self.run)
    def test_restore_wrong_seed_no_mutation(self):
        before=(self.run/'training_status.json').read_bytes()
        with self.assertRaises(ValueError):guard.restore_snapshot(self.run,guard.PROTOCOL,43)
        self.assertEqual(before,(self.run/'training_status.json').read_bytes())
    def test_restore_staging_failure_no_mutation(self):
        before=(self.run/'training_status.json').read_bytes()
        with patch('integrity.tempfile.TemporaryDirectory',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):guard.restore_snapshot(self.run,guard.PROTOCOL,44)
        self.assertEqual(before,(self.run/'training_status.json').read_bytes())
    def test_unsafe_zip_rejected(self):
        with zipfile.ZipFile(self.run/'safe_resume_snapshot.zip','a') as archive:archive.writestr('../escape','bad')
        with self.assertRaises(ValueError):guard.restore_snapshot(self.run,guard.PROTOCOL,44)
    def test_settings_mismatch_preserves_file(self):
        p=self.workspace/'settings.json';p.write_text('{"original":true}')
        with patch.object(safe_entry.repro,'run_contract',return_value={'wrong':True}):
            with self.assertRaises(ValueError):safe_entry.validate_settings_before_write(p,{'any':1})
        self.assertEqual(p.read_text(),'{"original":true}')
    def test_previous_evaluation_preserved(self):
        folder=guard.preserve_evaluation(self.run)
        self.assertEqual(guard.digest(folder/'test_predictions.json'),guard.digest(self.run/'test_predictions.json'))
    def test_atomic_json_failure_preserves_original(self):
        p=self.workspace/'settings.json';atomic_json(p,{'ok':1})
        with self.assertRaises(ValueError):atomic_json(p,{'bad':float('nan')})
        self.assertEqual(guard.load(p),{'ok':1})
    def test_committed_run_cannot_be_archived(self):
        with self.assertRaises(ValueError):archive_uncommitted(self.run)
    def test_uncommitted_run_is_preserved(self):
        fresh=self.workspace/'other/run';fresh.mkdir(parents=True)
        atomic_json(fresh/'run_manifest.json',{'contract':{'seed':44}})
        old=archive_uncommitted(fresh)
        self.assertTrue((old/'run_manifest.json').is_file());self.assertFalse(fresh.exists())
    def test_queue_rejects_missing_checkpoint(self):
        (self.run/self.status['best_checkpoint']).unlink()
        with self.assertRaises(ValueError):safe_queue.saved_state(self.workspace.parent,44)
    def test_metric_ties(self):
        self.assertEqual(guard.ap_auc([0,1],[.5,.5]),(.5,.5))
    def test_metric_perfect(self):self.assertEqual(guard.ap_auc([0,1],[.1,.9]),(1.,1.))
    def test_metric_reverse(self):self.assertEqual(guard.ap_auc([0,1],[.9,.1]),(.5,0.))

if __name__=='__main__':unittest.main()
