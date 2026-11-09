"""Stdlib process-tree checks; mandatory on production Linux before preflight."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import process_control as control


def clean_env():
    env = dict(os.environ)
    env.pop(control.GROUP_ENV, None)
    env.pop(control.DEADLINE_ENV, None)
    return env


def tree_command(heartbeat, exit_parent=False):
    child = "import pathlib,time; p=pathlib.Path(%r); exec('for i in range(600):\\n p.write_text(str(i))\\n time.sleep(.1)')" % str(heartbeat)
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(%s)" % (child, '.6' if exit_parent else '60')
    return [sys.executable, '-c', parent]


class SafetyChecks(unittest.TestCase):
    def test_deadline_stops_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat = Path(tmp)/'heartbeat'
            with self.assertRaises(control.DeadlineExceeded):
                control.run_checked(tree_command(heartbeat), env=clean_env(), deadline=time.time()+2)
            self.assertTrue(heartbeat.exists())
            before = heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(before, heartbeat.read_text())

    def test_interrupt_and_sigterm_stop_grandchild(self):
        for use_signal in (False, True):
            with self.subTest(sigterm=use_signal), tempfile.TemporaryDirectory() as tmp:
                heartbeat = Path(tmp)/'heartbeat'
                old_handler = signal.getsignal(signal.SIGTERM)
                real_popen = subprocess.Popen
                def intercepted(*args, **kwargs):
                    process = real_popen(*args, **kwargs)
                    if args[0][0] == 'taskkill':
                        return process
                    original_wait, first = process.wait, [True]
                    def wait(*a, **k):
                        if first[0]:
                            first[0] = False
                            time.sleep(1.)
                            if use_signal:
                                signal.raise_signal(signal.SIGTERM)
                            else:
                                raise KeyboardInterrupt('fixture Ctrl+C')
                        return original_wait(*a, **k)
                    process.wait = wait
                    return process
                with patch.object(control.subprocess, 'Popen', side_effect=intercepted), self.assertRaises(KeyboardInterrupt):
                    control.run_checked(tree_command(heartbeat), env=clean_env())
                self.assertEqual(old_handler, signal.getsignal(signal.SIGTERM))
                self.assertTrue(heartbeat.exists())
                before = heartbeat.read_text()
                time.sleep(.4)
                self.assertEqual(before, heartbeat.read_text())

    @unittest.skipUnless(os.name == 'posix', 'POSIX orphan cleanup must pass on Vast Linux')
    def test_normal_exit_stops_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat = Path(tmp)/'heartbeat'
            control.run_checked(tree_command(heartbeat, True), env=clean_env(), deadline=time.time()+4)
            before = heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(before, heartbeat.read_text())

    def test_nonfinite_and_expired_deadlines_do_not_launch(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with patch.object(control.subprocess, 'Popen') as child:
                with self.assertRaises(ValueError):
                    control.run_checked(['never-launch'], env=clean_env(), deadline=value)
                with self.assertRaises(ValueError):
                    control.run_checked(['never-launch'], env=dict(clean_env(), **{control.DEADLINE_ENV: str(value)}))
                child.assert_not_called()
        with patch.object(control.subprocess, 'Popen') as child, self.assertRaises(control.DeadlineExceeded):
            control.run_checked(['never-launch'], env=clean_env(), deadline=time.time()-1)
        child.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
