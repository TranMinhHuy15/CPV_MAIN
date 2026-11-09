import io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import backup_seed44 as bk
import integrity,run_seed44,safe_entry
from training_state import atomic_json
from process_control import exclusive_job

class RerunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ws=self.root/'workspace';self.ws.mkdir()
        atomic_json(self.ws/'settings.json',{'seed':44})
        (self.ws/'log.txt').write_text('original log')
    def test_backup_contains_rng_snapshot_and_checkpoint(self):
        run=self.ws/'run';run.mkdir()
        for name in ('best.pth','resume_latest.pth','rng_epoch31_rank0.pth','safe_resume_snapshot.zip'):(run/name).write_bytes(b'fixture')
        path=bk.backup(self.ws,self.root/'backups',diagnostic=True)
        m=bk.verify_backup(path)
        for name in ('best.pth','resume_latest.pth','rng_epoch31_rank0.pth','safe_resume_snapshot.zip'):self.assertIn('seed_44/run/'+name,m['files'])
        self.assertFalse(m['committed_artifacts_verified'])
    def test_backup_requires_committed_or_explicit_diagnostic(self):
        with self.assertRaises(ValueError):bk.backup(self.ws,self.root/'backups')
    def test_backup_inside_workspace_rejected(self):
        with self.assertRaises(ValueError):bk.backup(self.ws,self.ws/'backups',True)
    def test_backup_refuses_live_workspace(self):
        with exclusive_job(self.ws/'.safe_job.lock'):
            with self.assertRaises(RuntimeError):bk.backup(self.ws,self.root/'backups',True)
    def test_backup_releases_lock(self):
        bk.backup(self.ws,self.root/'backups',True)
        with exclusive_job(self.ws/'.safe_job.lock'):pass
    def test_insufficient_disk_fails_without_archive(self):
        class Disk:free=0
        with patch.object(bk.shutil,'disk_usage',return_value=Disk()):
            with self.assertRaises(OSError):bk.backup(self.ws,self.root/'backups',True)
        self.assertFalse(list((self.root/'backups').glob('*.zip')))
    def test_failed_copy_does_not_publish_backup(self):
        with patch.object(bk,'verify_backup',side_effect=ValueError('CRC')):
            with self.assertRaises(ValueError):bk.backup(self.ws,self.root/'backups',True)
        self.assertFalse(list((self.root/'backups').glob('*.zip')))
        self.assertFalse(list((self.root/'backups').glob('*.tmp')))
    def test_backup_corruption_detected(self):
        path=bk.backup(self.ws,self.root/'backups',True)
        with zipfile.ZipFile(path,'a') as z:z.writestr('unexpected','bad')
        with self.assertRaises(ValueError):bk.verify_backup(path)
    def test_raw_mp4_is_external(self):
        d=self.ws/'prepared_nexar/train';d.mkdir(parents=True);(d/'0001.mp4').write_bytes(b'not real video')
        path=bk.backup(self.ws,self.root/'backups',True);m=bk.verify_backup(path)
        self.assertEqual(len(m['external_videos']),1)
        self.assertNotIn('seed_44/prepared_nexar/train/0001.mp4',m['files'])
    def test_backup_rejects_wrong_seed(self):
        atomic_json(self.ws/'settings.json',{'seed':43})
        with self.assertRaises(ValueError):bk.backup(self.ws,self.root/'backups',True)
    def test_committed_backup_roundtrip_remains_evaluable(self):
        import test_integrity
        helper=test_integrity.IntegrityTests('test_complete_fixture');helper.setUp()
        self.addCleanup(helper.doCleanups)
        atomic_json(helper.workspace/'settings.json',{'seed':44})
        archive=bk.backup(helper.workspace,self.root/'backups')
        manifest=bk.verify_backup(archive);self.assertTrue(manifest['committed_artifacts_verified'])
        with zipfile.ZipFile(archive) as z:z.extractall(self.root/'restored')
        run=self.root/'restored/seed_44/run'
        integrity.training(run,complete=True)
        self.assertAlmostEqual(integrity.evaluation(run)['Combined']['mAP'],.7962148361915308)
    def test_atomic_failure_leaves_no_temp(self):
        with self.assertRaises(ValueError):atomic_json(self.ws/'settings.json',{'seed':float('nan')})
        self.assertEqual(json.loads((self.ws/'settings.json').read_text()),{'seed':44})
        self.assertFalse(list(self.ws.glob('*.tmp')))
    def test_resume_is_explicit(self):
        with patch.object(run_seed44.safe_entry,'main') as call:
            run_seed44.main(['resume','--workspace',str(self.ws)])
        command=call.call_args[0][0]
        self.assertIn('--resume-root',command);self.assertIn('--train',command)
        self.assertEqual(command[command.index('--seed')+1],'44')
    def test_fresh_train_does_not_sneak_resume(self):
        with patch.object(run_seed44.safe_entry,'main') as call:run_seed44.main(['train'])
        self.assertNotIn('--resume-root',call.call_args[0][0])
    def test_unsafe_workspace_in_dataset_rejected(self):
        with self.assertRaises(ValueError):safe_entry.main(['--data-root',str(self.root),'--workspace',str(self.ws)])

if __name__=='__main__':unittest.main()
