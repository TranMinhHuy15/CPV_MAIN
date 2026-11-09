import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import diagnostics
import protocol
import recovery
import study_guard


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_preflight_retry_only_allows_source_staging(self):
        stage = self.root / 'checked_source'
        stage.mkdir()
        study_guard.ensure_output(self.root)
        (self.root / '.study_contract.json.partial.tmp').touch()
        study_guard.ensure_output(self.root)
        (self.root / 'trials').mkdir()
        with self.assertRaises(ValueError):
            study_guard.ensure_output(self.root)

    def test_incomplete_first_epoch_archived_and_recovery_can_continue(self):
        run = self.root / 'run'
        run.mkdir()
        expected = {'seed': 42}
        protocol.save(run / 'run_manifest.json', {'contract': expected})
        protocol.save(run / 'training_status.json', dict(
            completed_epochs=1, target_epochs=50, complete=False,
            best_checkpoint='best_mAUC@_epoch_1.pth'))
        (run / 'safe_resume_snapshot.zip.tmp').write_bytes(b'incomplete')
        recovered = []
        with contextlib.redirect_stdout(io.StringIO()):
            recovery.recover_training(run, expected, lambda path, _: recovered.append(path.exists()))
        self.assertEqual(recovered, [False])
        archived, = self.root.glob('run_uncommitted_epoch1_*')
        self.assertEqual((archived / 'safe_resume_snapshot.zip.tmp').read_bytes(), b'incomplete')

    def test_missing_later_snapshot_is_not_silently_restarted(self):
        run = self.root / 'run'
        run.mkdir()
        expected = {'seed': 42}
        protocol.save(run / 'run_manifest.json', {'contract': expected})
        protocol.save(run / 'training_status.json', dict(completed_epochs=3))
        with self.assertRaises(ValueError):
            recovery.recover_training(run, expected, lambda *_: self.fail('must stop'))
        self.assertTrue(run.exists())

    def test_pilot_checks_recovered_epoch_instead_of_uncommitted_status(self):
        run = self.root / 'run'
        run.mkdir()
        status = run / 'training_status.json'
        (run / 'safe_resume_snapshot.zip').touch()
        protocol.save(status, dict(completed_epochs=3))
        upstream = lambda path, expected: protocol.save(status, dict(completed_epochs=2))
        module = types.SimpleNamespace(recover=upstream)
        with recovery.training_recovery(module, 3):
            module.recover(run, {'seed': 42})
        self.assertIs(module.recover, upstream)
        self.assertEqual(protocol.load(status)['completed_epochs'], 2)
        with recovery.training_recovery(module, 2), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                module.recover(run, {'seed': 42})
        self.assertEqual(stopped.exception.code, 75)

    def test_interrupted_diagnostics_replace_preserves_previous_records(self):
        for epoch in (1, 2):
            diagnostics.append(self.root / diagnostics.FILES[0], dict(epoch=epoch))
            for batch in (1, 2):
                diagnostics.append(self.root / diagnostics.FILES[1], dict(epoch=epoch, batch=batch))
        path = self.root / diagnostics.FILES[0]
        before = path.read_bytes()
        with patch.object(recovery.os, 'replace', side_effect=OSError('interruption')):
            with self.assertRaises(OSError):
                diagnostics.trim_uncommitted(self.root, 1)
        self.assertEqual(path.read_bytes(), before)
        diagnostics.trim_uncommitted(self.root, 1)
        self.assertEqual([json.loads(line)['epoch'] for line in path.read_text().splitlines()], [1])
        with self.assertRaises(ValueError):
            diagnostics.trim_uncommitted(self.root, 2)

    def test_invalid_borrowed_cache_falls_back_and_pins_fresh_cache(self):
        old = self.root / 'old'
        old.mkdir()
        (old / 'cache_manifest.json').write_text('broken')
        cfg = dict(output_root=str(self.root / 'new'), cache_roots={'42': str(old)})
        extracted = []

        def extract(path):
            extracted.append(path)
            protocol.save(path / 'cache_manifest.json', dict(complete=True))

        def verify(path):
            file = path / 'cache_manifest.json'
            return protocol.load(file), protocol.sha(file)

        with patch.object(recovery, 'require_free_space'), contextlib.redirect_stdout(io.StringIO()):
            recovery.prepare_cache(cfg, 42, verify, extract)
            recovery.prepare_cache(cfg, 42, verify, extract)
        self.assertEqual(extracted, [recovery.cache_for(cfg, 42)])
        self.assertEqual((old / 'cache_manifest.json').read_text(), 'broken')
        protocol.save(extracted[0] / 'cache_manifest.json', dict(complete=False))
        with self.assertRaises(ValueError):
            recovery.cache_for(cfg, 42)

    def test_low_disk_stops_before_writing_checkpoint(self):
        with patch.object(recovery.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=3 * recovery.GIB)):
            with self.assertRaisesRegex(RuntimeError, 'Expand the volume'):
                recovery.require_free_space(self.root, 4 * recovery.GIB, 'train/checkpoint')
