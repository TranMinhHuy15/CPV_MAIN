import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_rq2 import validate_arguments, summarize, package_digest, command_for, main, save
from repro import run_contract
from rq2_profile import scientific_profile
from rq2_profile import protocol, VARIANTS


class QueueTests(unittest.TestCase):
    def test_preflight_cli_visits_both_arms_all_seeds_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            (base/'data').mkdir()
            calls=[]
            def child(command,**kwargs):
                if Path(command[1]).name=='run_vast.py':
                    self.assertNotIn('--train',command)
                    self.assertNotIn('--evaluate-only',command)
                    calls.append((kwargs['env']['RQ2_VARIANT'],int(command[command.index('--seed')+1])))
            with patch('run_rq2.verify_package'),patch('run_rq2.run_checked',side_effect=child):
                result=main(['--data-root',str(base/'data'),'--output-root',str(base/'out'),
                    '--runtime-root',str(base/'runtime'),'--total-hours','1'])
            self.assertEqual(result,0)
            self.assertEqual(calls,[(v,s) for s in (42,43,44) for v in VARIANTS])

    def test_partial_training_cli_blocks_every_later_job(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            (base/'data').mkdir()
            calls=[]
            def child(command,**kwargs):
                if Path(command[1]).name!='run_vast.py':
                    return
                calls.append(command)
                self.assertIn('--train',command)
                run=Path(command[command.index('--workspace')+1])/'run'
                variant=kwargs['env']['RQ2_VARIANT']
                contract=run_contract(dict(seed=42,epochs=50,batch_per_gpu=2,effective_batch_size=16,world_size=1,amp=True))
                contract.update(protocol=protocol(variant),rq2=scientific_profile(variant))
                save(run/'run_manifest.json',dict(contract=contract))
                save(run/'training_status.json',dict(complete=False,completed_epochs=1,target_epochs=50,best_checkpoint='fixture.pth'))
            with patch('run_rq2.verify_package'),patch('run_rq2.run_checked',side_effect=child):
                result=main(['--data-root',str(base/'data'),'--output-root',str(base/'out'),
                    '--runtime-root',str(base/'runtime'),'--total-hours','1','--train'])
            self.assertEqual(result,75)
            self.assertEqual(len(calls),1)

    def test_pilot_never_opens_test_and_completed_pilot_does_not_retrain(self):
        args = SimpleNamespace(data_root='data',runtime_root='runtime',total_hours=2.,train=True,validation_only=True)
        command = command_for(args,Path('output/long/seed_42/run'),None,42,12345.)
        self.assertIn('--train',command)
        self.assertIn('--validation-only',command)
        self.assertNotIn('--evaluate-only',command)
        self.assertEqual(command[command.index('--job-deadline')+1],'12345.0')
        self.assertIsNone(command_for(args,Path('output/long/seed_42/run'),dict(complete=True),42,12345.))
        args.validation_only = False
        command = command_for(args,Path('output/long/seed_42/run'),dict(complete=True),42,12345.)
        self.assertIn('--evaluate-only',command)
        self.assertNotIn('--train',command)

    def test_arm_protocols_differ_and_reject_unknown(self):
        self.assertNotEqual(protocol('short'), protocol('long'))
        with self.assertRaises(ValueError):
            protocol('unknown')

    def test_paid_training_requires_explicit_budget(self):
        for hours in (None, 0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                validate_arguments(SimpleNamespace(total_hours=hours, train=True, seeds=[42], variants=['short']))

    def test_duplicate_jobs_rejected(self):
        for seeds, variants in [([42,42], ['short']), ([42], ['short','short'])]:
            with self.assertRaises(ValueError):
                validate_arguments(SimpleNamespace(total_hours=1., train=False, seeds=seeds, variants=variants))

    def test_no_final_aggregate_from_partial_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = summarize(Path(folder))
            self.assertFalse(result['complete_study'])
            self.assertEqual(result['mean_sample_sd'], {})

    def test_old_v6_contract_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / VARIANTS[0] / 'seed_42/run'
            run.mkdir(parents=True)
            (run/'run_manifest.json').write_text(json.dumps({'contract': {'protocol': 'riskprop-v6', 'seed':42}}))
            (run/'training_status.json').write_text(json.dumps({'completed_epochs':50, 'target_epochs':50, 'complete':True}))
            with self.assertRaises(ValueError):
                summarize(Path(folder))


if __name__ == '__main__':
    unittest.main()
