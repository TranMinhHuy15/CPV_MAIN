"""Owned small checkpoints/report fixtures verify retention and same-seed deltas."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import protocol as p
import artifacts
import reporting


class ArtifactsReports(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_compaction_preserves_exact_adapter_and_rejects_changed_frozen_weights(self):
        out=self.root/'out';out.mkdir();p.save(out/'study_contract.json',dict(fixture=True))
        base_path=self.root/'original_h7.pth';base_state={'frozen.weight':torch.arange(8,dtype=torch.float32)}
        torch.save(dict(state_dict=base_state),base_path)
        original_bytes=base_path.read_bytes();base=dict(checkpoint=str(base_path),checkpoint_sha256=p.sha(base_path))
        workspace=out/'trials/c2_lr_double/m1/seed_44';run=workspace/'run';run.mkdir(parents=True)
        p.save(workspace/'owned_trial.json',dict(study_contract_sha256=p.sha(out/'study_contract.json')))
        contract=dict(seed=44,arm='m1',hyperparameters=p.candidate('c2_lr_double'))
        p.save(run/'run_manifest.json',dict(contract=contract))
        status=dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint='best_mAUC@_epoch_2.pth');p.save(run/'training_status.json',status)
        history=[]
        for epoch in range(1,51):
            metrics={k:.9 if epoch==2 else .7 for k in p.METRICS};metrics['threshold@0.10']=.6;history.append(dict(epoch=epoch,metrics=metrics))
        (run/'validation_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
        selected=run/status['best_checkpoint'];state=dict(base_state,**{'cls_head.adapter.output.weight':torch.ones(1,4)})
        torch.save(dict(state_dict=state),selected)
        (run/'resume_latest.pth').write_bytes(b'fixture');(run/'safe_resume_snapshot.zip').write_bytes(b'fixture snapshot already verified by stub')
        calls=[];training=types.SimpleNamespace(assert_committed=lambda *args,**kw:calls.append(1))
        with patch.dict(sys.modules,{'training_state':training}),contextlib.redirect_stdout(io.StringIO()):
            row=artifacts.compact(workspace,base,dict(output_root=str(out),compact_completed_trials=True))
        self.assertEqual(calls,[1]);self.assertEqual(base_path.read_bytes(),original_bytes)
        self.assertFalse(selected.exists());self.assertFalse((run/'resume_latest.pth').exists())
        self.assertEqual(artifacts.verify_trial(workspace),row)
        with np.load(workspace/'selected_adapter.npz',allow_pickle=False) as z:self.assertTrue(np.array_equal(z['output.weight'],np.ones((1,4))))
        (workspace/'selected_adapter.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):artifacts.verify_trial(workspace)
        state['frozen.weight']=torch.ones(8)*999;torch.save(dict(state_dict=state),selected)
        (run/'safe_resume_snapshot.zip').write_bytes(b'fixture')
        with patch.dict(sys.modules,{'training_state':training}):
            with self.assertRaisesRegex(ValueError,'Frozen baseline changed'):
                artifacts.compact(workspace,base,dict(output_root=str(out),compact_completed_trials=True))
        self.assertTrue(selected.exists(),'failed validation must not delete even owned generated weights')

    def test_pairwise_tables_keep_each_seed_and_invalidate_stale_completion(self):
        out=self.root/'reports';out.mkdir();p.save(out/'study_contract.json',dict(fixture=True))
        p.save(out/'runtime_identity.json',dict(fixture=True))
        plan=dict(study_contract_sha256=p.sha(out/'study_contract.json'),baselines={'v6':{}},chosen={arm:dict(candidate='c2_lr_double',hyperparameters=p.candidate('c2_lr_double'),selected=[]) for arm in p.ARMS})
        plan['runtime_identity_sha256']=p.sha(out/'runtime_identity.json')
        for seed in p.SEEDS:
            for role in ('b0',*p.ARMS):
                run=out/'test/runs'/role/('seed_%d'%seed)/'run';run.mkdir(parents=True)
                cp=run/'weights.bin';cp.write_bytes((role+str(seed)).encode());checkpoint_sha=p.sha(cp)
                contract=dict(arm=role,seed=seed)
                if role in p.ARMS:
                    plan['chosen'][role]['selected'].append(dict(seed=seed,adapter_sha256=checkpoint_sha,contract_sha256=p.digest(contract)))
                else:
                    contract['original_contract_sha256']='original'+role+str(seed)
                    original=out/'v6_original'/str(seed);original.mkdir(parents=True,exist_ok=True)
                    p.save(original/'test_predictions.json',[dict(id='%05d'%i,score=.5) for i in range(8)])
                    plan['baselines']['v6'][str(seed)]=dict(checkpoint_sha256=checkpoint_sha,contract_sha256=contract['original_contract_sha256'],run=str(original))
                p.save(run/'run_manifest.json',dict(contract=contract));p.save(run/'training_status.json',dict(best_checkpoint='weights.bin'))
        p.save(out/'selection_frozen_before_test.json',plan);lock=p.digest(plan)
        for seed in p.SEEDS:
            p.save(out/('test_complete_seed%d.json'%seed),dict(complete=True,seed=seed,selection_lock_sha256=lock))
            for role in ('b0',*p.ARMS):
                run=out/'test/runs'/role/('seed_%d'%seed)/'run';checkpoint_sha=p.sha(run/'weights.bin')
                fingerprint=dict(selection_lock_sha256=lock,checkpoint_sha256=checkpoint_sha,training_contract=p.load(run/'run_manifest.json')['contract'],blind_index_sha256='same-clock')
                hashes={}
                for i in range(8):
                    path=run/'test_capture/curves'/('%05d.json'%i);p.save(path,dict(fixture=True));hashes['%05d'%i]=p.sha(path)
                score=.7+(seed-42)*.01+(0 if role=='b0' else .04)
                metrics={key:score for key in ('mAP','mAUC_grouped','AP@0.5s','AP@1.0s','AP@1.5s')}
                metrics.update({'fixture_metric%d'%i:score for i in range(23)})
                p.save(run/'rq2_test_report.json',dict(complete=True,test_count=8,seed=seed,capture_fingerprint=fingerprint,checkpoint_sha256=checkpoint_sha,
                       selected_checkpoint='weights.bin',curve_file_sha256=hashes,summary_metrics=metrics))
                (run/'rq2_test_metrics.csv').write_text('fixture');p.save(run/'test_predictions.json',[dict(id='%05d'%i,score=.5+(1e-4 if role=='b0' else 0)) for i in range(8)])
                p.save(run/'test_capture/capture_manifest.json',dict(complete=True));p.save(run/'test_capture/capture_contract.json',fingerprint)
                p.save(run/'evaluation_complete.json',dict(complete=True,test_count=8,metric_schema='rq2-28-v1',checkpoint_sha256=checkpoint_sha,
                       files={name:p.sha(run/name) for name in reporting.REPORT_FILES}))
        with self.assertRaises(ValueError):reporting.report(out)  # Production1344 gate.
        with patch.object(reporting,'TEST_COUNT',8),contextlib.redirect_stdout(io.StringIO()):
            reporting.report(out)
            import csv
            with (out/'comparison/per_seed_all28_metrics_vs_V6.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3*3*28)
            self.assertTrue(all(r['seed']==r['baseline_seed'] and r['baseline']=='V6' for r in rows))
            for row in [r for r in rows if r['metric']=='mAP']:self.assertAlmostEqual(float(row['signed_delta']),4)
            with (out/'comparison/v6_rescore_parity.csv').open(encoding='utf-8-sig',newline='') as f:parity=list(csv.DictReader(f))
            self.assertEqual(len(parity),3);self.assertTrue(all(abs(float(r['max_abs_score_diff'])-1e-4)<1e-9 for r in parity))
            (out/'test/runs/m1_dx/seed_44/run/test_capture/curves/00000.json').write_text('{}')
            with self.assertRaises(ValueError):reporting.report(out)
            self.assertFalse((out/'comparison/TEST_COMPARISON_COMPLETE.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
