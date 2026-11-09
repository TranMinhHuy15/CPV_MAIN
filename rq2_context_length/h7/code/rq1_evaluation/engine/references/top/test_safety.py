"""Run on Linux BEFORE paid training; local Windows checks have explicit skips."""
import contextlib
import json
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
from checkpoint_safety import commit_snapshot, restore_snapshot

HERE=Path(__file__).resolve().parent


def tree_command(heartbeat, exit_parent=False):
    grandchild=("import pathlib,time; p=pathlib.Path(%r); "
        "exec('for i in range(600):\\n p.write_text(str(i))\\n time.sleep(.1)')"%str(heartbeat))
    parent="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(%s)"%(grandchild,'.6' if exit_parent else '60')
    return [sys.executable,'-c',parent]


def clean_env():
    env=dict(os.environ)
    env.pop(control.GROUP_ENV,None)
    env.pop(control.DEADLINE_ENV,None)
    return env


class SafetyChecks(unittest.TestCase):
    def test_deadline_stops_real_child_and_grandchild(self):
        with tempfile.TemporaryDirectory() as temp:
            heartbeat=Path(temp)/'heartbeat'
            with self.assertRaises(control.DeadlineExceeded):
                control.run_checked(tree_command(heartbeat),env=clean_env(),deadline=time.time()+2)
            self.assertTrue(heartbeat.exists())
            before=heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(heartbeat.read_text(),before,'Grandchild survived the total deadline')

    def test_interrupt_stops_real_child_and_grandchild(self):
        with tempfile.TemporaryDirectory() as temp:
            heartbeat=Path(temp)/'heartbeat'
            real=control.subprocess.Popen
            def interrupted_popen(*args,**kwargs):
                process=real(*args,**kwargs)
                if args[0][0]=='taskkill':
                    return process
                original=process.wait
                first=[True]
                def wait(*a,**k):
                    if first[0]:
                        first[0]=False
                        time.sleep(1.)
                        raise KeyboardInterrupt('fixture Ctrl+C')
                    return original(*a,**k)
                process.wait=wait
                return process
            with patch.object(control.subprocess,'Popen',side_effect=interrupted_popen):
                with self.assertRaises(KeyboardInterrupt):
                    control.run_checked(tree_command(heartbeat),env=clean_env())
            self.assertTrue(heartbeat.exists())
            before=heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(heartbeat.read_text(),before,'Grandchild survived Ctrl+C')

    def test_sigterm_handler_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            heartbeat=Path(temp)/'heartbeat'
            old_handler=signal.getsignal(signal.SIGTERM)
            real=control.subprocess.Popen
            def signalled_popen(*args,**kwargs):
                process=real(*args,**kwargs)
                if args[0][0]=='taskkill':
                    return process
                original=process.wait
                first=[True]
                def wait(*a,**k):
                    if first[0]:
                        first[0]=False
                        time.sleep(1.)
                        signal.raise_signal(signal.SIGTERM)
                    return original(*a,**k)
                process.wait=wait
                return process
            with patch.object(control.subprocess,'Popen',side_effect=signalled_popen):
                with self.assertRaises(KeyboardInterrupt):
                    control.run_checked(tree_command(heartbeat),env=clean_env())
            self.assertEqual(signal.getsignal(signal.SIGTERM),old_handler)
            before=heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(heartbeat.read_text(),before)

    @unittest.skipUnless(os.name=='posix','Real POSIX orphan cleanup must run on rented Linux before training')
    def test_normal_parent_exit_cleans_orphan(self):
        with tempfile.TemporaryDirectory() as temp:
            heartbeat=Path(temp)/'heartbeat'
            control.run_checked(tree_command(heartbeat,exit_parent=True),env=clean_env(),deadline=time.time()+4)
            before=heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(heartbeat.read_text(),before)

    def test_expired_budget_does_not_launch(self):
        with patch.object(control.subprocess,'Popen') as child:
            with self.assertRaises(control.DeadlineExceeded):
                control.run_checked([sys.executable,'-c','pass'],deadline=time.time()-1)
            child.assert_not_called()

    def test_output_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            lock=Path(temp)/'job.lock'
            with control.exclusive_job(lock):
                with self.assertRaisesRegex(RuntimeError,'Another job'):
                    with control.exclusive_job(lock):
                        pass
            with control.exclusive_job(lock):
                pass

    def test_nonfinite_deadlines_do_not_launch(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with patch.object(control.subprocess,'Popen') as child:
                with self.assertRaises(ValueError):
                    control.run_checked(['never-launch'],env=clean_env(),deadline=value)
                env=dict(clean_env(),**{control.DEADLINE_ENV:str(value)})
                with self.assertRaises(ValueError):
                    control.run_checked(['never-launch'],env=env)
                child.assert_not_called()

    def test_logged_child_pipe_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            processes=[]
            real=control.subprocess.Popen
            def capture(*args,**kwargs):
                child=real(*args,**kwargs)
                processes.append(child)
                return child
            with patch.object(control.subprocess,'Popen',side_effect=capture):
                control.run_checked([sys.executable,'-c',"print('pipe-fixture')"],
                    env=clean_env(),log_path=Path(tmp)/'stdout.log',deadline=time.time()+10)
            self.assertTrue(processes[0].stdout.closed)
            self.assertIn('pipe-fixture',(Path(tmp)/'stdout.log').read_text())

    def test_snapshot_rolls_back_interrupted_save(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            status=dict(completed_epochs=17,target_epochs=50,complete=False,best_checkpoint='best_epoch10.pth')
            originals={'resume_latest.pth':b'completed-epoch17-optimizer-scaler',
                'best_epoch10.pth':b'complete-selected-model','rng_epoch17_rank0.pth':b'epoch17-rng',
                'training_status.json':json.dumps(status).encode(),
                'run_manifest.json':json.dumps(dict(contract=dict(protocol='fixture',seed=42))).encode(),
                'validation_history.jsonl':b'{"epoch":17}\n'}
            for name,value in originals.items():
                (run/name).write_bytes(value)
            commit_snapshot(run)
            (run/'resume_latest.pth').write_bytes(b'interrupted-next-save')
            (run/'best_epoch10.pth').unlink() # Simulates original CheckpointHook replacing old best.
            (run/'training_status.json').write_text('{"completed_epochs":18}')
            (run/'safe_resume_snapshot.zip.tmp').write_bytes(b'incomplete new archive')
            self.assertTrue(restore_snapshot(run,'fixture',42))
            for name,value in originals.items():
                self.assertEqual((run/name).read_bytes(),value)
            with self.assertRaisesRegex(RuntimeError,'different protocol/seed'):
                restore_snapshot(run,'fixture',43)

    def test_global_budget_flags_and_strict_eval_contract(self):
        import run_three_seeds as queue
        source=(HERE/'run_vast.py').read_text()
        self.assertIn("env[DEADLINE_ENV] = str(job_deadline)",source)
        self.assertIn('min(time.time()+args.hours*3600, job_deadline)',source)
        self.assertIn('manifest["contract"] != run_contract(settings)',(HERE/'repro.py').read_text())
        class Args:
            data_root='data'
            runtime_root='runtime'
            hours_per_seed=24
            job_deadline=time.time()+3600
            train=True
        command=queue.command_for(Args(),42,Path('results/seed_42/run'),None)
        self.assertEqual(float(command[command.index('--job-deadline')+1]),Args.job_deadline)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SafetyChecks))
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),skipped=len(result.skipped),platform=os.name,
        real_posix_process_tree_verified=os.name=='posix' and result.wasSuccessful(),
        scope='Real local subprocess deadlines/Ctrl+C, lock and synthetic snapshot recovery; POSIX normal-exit cleanup skipped on Windows')
    (HERE/'safety_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
