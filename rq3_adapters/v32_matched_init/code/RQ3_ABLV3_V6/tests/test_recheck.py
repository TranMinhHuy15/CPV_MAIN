import contextlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import capture_helpers
import console
import protocol
import recovery


class RecheckTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / 'scratch'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_staged_validation_history_survives_failed_replace_and_retries(self):
        source = self.root / 'original.jsonl'
        source.write_bytes(b'complete validation history\n')
        target = self.root / 'test/history.jsonl'
        with patch.object(recovery.os, 'replace', side_effect=OSError('interrupted publication')):
            with self.assertRaises(OSError):
                recovery.stage_file(source, target)
        self.assertFalse(target.exists())
        target.write_bytes(b'complete valid')
        recovery.stage_file(source, target)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        saved, = target.parent.glob('history.jsonl.interrupted_*')
        self.assertEqual(saved.read_bytes(), b'complete valid')
        target.write_bytes(b'unrelated history')
        with self.assertRaises(ValueError):
            recovery.stage_file(source, target)
        self.assertEqual(target.read_bytes(), b'unrelated history')

    def test_cache_support_is_published_only_after_staging(self):
        shared = self.root / 'shared'
        cache = self.root / 'cache'
        cache.mkdir()
        relative = 'author_source/example.txt'
        target = cache / relative
        calls = []

        def export(path, stage):
            self.assertEqual(path, shared)
            calls.append(stage)
            staged_file = stage / relative
            staged_file.parent.mkdir(parents=True)
            staged_file.write_bytes(b'correct source bytes')
            if len(calls) == 1:
                raise OSError('interrupted export')
            return {relative: protocol.sha(staged_file)}

        driver = types.SimpleNamespace(support_export=export)
        with recovery.atomic_cache_support(driver):
            with self.assertRaises(OSError):
                driver.support_export(shared, cache)
            self.assertFalse(target.exists())
            result = driver.support_export(shared, cache)
        self.assertEqual(result, {relative: protocol.sha(target)})
        self.assertIs(driver.support_export, export)
        self.assertTrue(all(path != cache for path in calls))

    def test_partial_shared_source_does_not_unlock_test(self):
        cfg = self.root / 'author_source/configs/predict_anomaly_snippet.py'
        cfg.parent.mkdir(parents=True)
        cfg.touch()
        self.assertFalse(capture_helpers.shared_inputs_ready(self.root))
        data = self.root / 'prepared_nexar'
        data.mkdir()
        annotation = data / 'annotations.csv'
        annotation.write_text('valid annotation')
        protocol.save(data / 'data_manifest.json', dict(
            video_count=2844, verified_content_hashes=True, annotations_sha256=protocol.sha(annotation)))
        self.assertTrue(capture_helpers.shared_inputs_ready(self.root))
        annotation.write_text('partial')
        self.assertFalse(capture_helpers.shared_inputs_ready(self.root))

    def test_linux_nested_worker_inherits_launcher_process_group(self):
        process = types.SimpleNamespace(stdout=io.StringIO(''), poll=lambda: 0, returncode=0)
        os_fixture = types.SimpleNamespace(name='posix', environ={}, killpg=Mock())
        log = console.Log(self.root)
        with patch.object(console, 'os', os_fixture), \
             patch.object(console.signal, 'signal', return_value=console.signal.SIG_DFL), \
             patch.object(console.subprocess, 'Popen', return_value=process) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.command(['fixture'], 'top', log), 0)
            top = launch.call_args.kwargs
            process.stdout = io.StringIO('')
            self.assertEqual(console.command(['fixture'], 'nested', log, env=top['env']), 0)
            nested = launch.call_args.kwargs
        self.assertTrue(top['start_new_session'])
        self.assertFalse(nested['start_new_session'])
        self.assertEqual(nested['env']['RISKPROP_SUPERVISED_GROUP'], '1')

    def test_linux_interrupt_signals_group_and_waits(self):
        process = Mock(pid=61234, returncode=-15, stdout=io.StringIO(''))
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        os_fixture = types.SimpleNamespace(name='posix', environ={}, killpg=Mock())
        log = console.Log(self.root)
        kill_signal = getattr(console.signal, 'SIGKILL', 9)
        with patch.object(console, 'os', os_fixture), \
             patch.object(console.signal, 'SIGKILL', kill_signal, create=True), \
             patch.object(console.signal, 'signal', return_value=console.signal.SIG_DFL), \
             patch.object(console.subprocess, 'Popen', return_value=process), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                console.command(['fixture'], 'interrupt', log)
        os_fixture.killpg.assert_any_call(process.pid, console.signal.SIGTERM)
        os_fixture.killpg.assert_any_call(process.pid, kill_signal)
