"""Bounded CPU fixtures; no installers, network, video, or paid/GPU experiments."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'legacy'))
import protocol as p


class Policy(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def matrix(self):
        rows=[]
        for arm in p.ARMS:
            for c in p.CANDIDATES:
                for seed in p.SEEDS:
                    rows.append(dict(arm=arm,candidate=c['id'],seed=seed,validation_metrics={'mAUC@':.80+seed/1e4,'AP@1.5s':.7}))
        return rows

    def test_single_predeclared_candidate_and_arms(self):
        self.assertEqual([c['id'] for c in p.CANDIDATES],['c2_lr_double'])
        self.assertEqual(p.ARMS,('m1','m1_m2','m1_dx'))
        rows=self.matrix();chosen,ranks=p.rank_common(rows)
        self.assertEqual(len(rows),9);self.assertEqual(len(ranks),3)
        for arm in p.ARMS:
            self.assertEqual(chosen[arm]['candidate'],'c2_lr_double')
            self.assertEqual({r['seed'] for r in chosen[arm]['selected']},{42,43,44})
        with self.assertRaises(ValueError):p.rank_common([r for r in rows if r['seed']!=44])
        with self.assertRaises(ValueError):p.rank_common(rows+[rows[0]])
        rows[0]['validation_metrics']['mAUC@']=float('nan')
        with self.assertRaises(ValueError):p.rank_common(rows)
        self.assertTrue(p.validate_candidates())
        with patch.object(p,'CANDIDATES',(p.CANDIDATES[0],dict(p.CANDIDATES[0],id='other_name'))):
            with self.assertRaises(ValueError):p.validate_candidates()

    def test_config_requires_v6_runs_for_all_seeds_and_no_overlap(self):
        cfg=p.load(ROOT/'inputs.example.json');self.assertIs(p.validate_config(cfg),cfg)
        self.assertNotIn('h7_runs',cfg);self.assertNotIn('h13_runs',cfg)
        cfg['v6_runs'].pop('44')
        with self.assertRaises(ValueError):p.validate_config(cfg)
        cfg=p.load(ROOT/'inputs.example.json');cfg['output_root']=cfg['data_root']
        with self.assertRaises(ValueError):p.validate_config(cfg)
        cfg=p.load(ROOT/'inputs.example.json');cfg['h13_runs']={}
        with self.assertRaises(ValueError):p.validate_config(cfg)

    def v6_fixture(self,run,seed=44,best=2):
        run.mkdir(parents=True)
        contract=dict(seed=seed,source_commit=p.SOURCE_COMMIT,data_revision=p.DATA_REVISION,preprocessing=p.V6_PREPROCESSING,
                      epochs=50,protocol=p.V6_PROTOCOL,amp=True,batch_per_gpu=2,effective_batch_size=16,world_size=1,
                      loss='published_code_no_algorithmic_changes',launcher_sha256={'kaggle_adapter.py':'adapter'})
        p.save(run/'run_manifest.json',dict(contract=contract))
        p.save(run/'training_status.json',dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint='best_mAUC@_epoch_%d.pth'%best))
        rows=[]
        for epoch in range(1,51):
            metrics={k:.9 if epoch==best else .7 for k in p.METRICS};metrics['threshold@0.10']=.6
            rows.append(dict(epoch=epoch,metrics=metrics))
        (run/'validation_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        (run/('best_mAUC@_epoch_%d.pth'%best)).write_bytes(b'fake v6 weights %d'%seed)
        p.save(run/'validation_predictions.json',[dict(id='%05d'%i,target=0,pred=[.1]*30) for i in range(300)])
        p.save(run/'test_predictions.json',[dict(id='%05d'%i,score=.5) for i in range(1344)])
        return contract

    def test_v6_baseline_route(self):
        run=self.root/'v6';self.v6_fixture(run)
        row=p.validate_v6_baseline(run,44,'adapter')
        self.assertEqual(row['selected_epoch'],2);self.assertEqual(row['history'],0)
        self.assertEqual(row['baseline_input_kind'],'frozen_V6_selected_checkpoint')
        with self.assertRaises(ValueError):p.validate_v6_baseline(run,42,'adapter')       # wrong seed
        with self.assertRaises(ValueError):p.validate_v6_baseline(run,44,'other')         # other preprocessing source
        (run/'resume_latest.pth').write_bytes(b'x')
        with self.assertRaises(ValueError):p.validate_v6_baseline(run,44,'adapter')       # extra weights
        (run/'resume_latest.pth').unlink()
        m=p.load(run/'run_manifest.json');m['contract']['rq2']=dict(history_tokens=7);p.save(run/'run_manifest.json',m)
        with self.assertRaises(ValueError):p.validate_v6_baseline(run,44,'adapter')       # an LSTM run is not V6
        run2=self.root/'v6b';self.v6_fixture(run2);(run2/'validation_predictions.json').unlink()
        with self.assertRaises(FileNotFoundError):p.validate_v6_baseline(run2,44,'adapter')
        run3=self.root/'v6c';self.v6_fixture(run3)
        st=p.load(run3/'training_status.json');st['best_checkpoint']='best_mAUC@_epoch_3.pth';p.save(run3/'training_status.json',st)
        with self.assertRaises(ValueError):p.validate_v6_baseline(run3,44,'adapter')       # not earliest max

    def test_eta_units_and_json_stable_binding(self):
        result=p.measured_eta(dict.fromkeys(p.ARMS,20),2,3)
        self.assertEqual(result['training_runs'],9);self.assertEqual(result['epochs'],450)
        self.assertEqual(result['training_hours'],2.5)
        self.assertEqual(result['extraction_hours'],2.5);self.assertEqual(result['test_hours'],3.36)
        obj=dict(candidates=list(p.CANDIDATES));path=self.root/'identity.json'
        self.assertEqual(p.bind(path,obj),p.bind(path,obj))
        with self.assertRaises(ValueError):p.bind(path,dict(changed=True))

    def test_source_syntax_python38(self):
        for path in ROOT.rglob('*.py'):
            if 'scratch' not in path.parts:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))

    def test_failed_log_write_stops_owned_subprocess(self):
        import console
        import types
        import contextlib
        import io
        actual_open=Path.open;actual_popen=console.subprocess.Popen;children=[]
        def opening(path,*args,**kwargs):
            if path.suffix=='.log':raise OSError('fixture log disk failure')
            return actual_open(path,*args,**kwargs)
        def launching(*args,**kwargs):
            process=actual_popen(*args,**kwargs);children.append(process);return process
        log=types.SimpleNamespace(root=self.root,event=lambda *args,**kwargs:None)
        with patch.object(Path,'open',opening),patch.object(console.subprocess,'Popen',side_effect=launching),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError,'fixture log disk failure'):
                console.command([sys.executable,'-u','-c','import time; print("TRAIN fixture",flush=True); time.sleep(2)'],'owned_log_fixture',log)
        self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll())


if __name__=='__main__':unittest.main(verbosity=2)
