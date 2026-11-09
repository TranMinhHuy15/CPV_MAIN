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
        self.assertEqual(p.ARMS,('c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned'))
        rows=self.matrix();chosen,ranks=p.rank_common(rows)
        self.assertEqual(len(rows),18);self.assertEqual(len(ranks),6)
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

    def test_config_requires_h7_runs_prior_study_and_no_overlap(self):
        cfg=p.load(ROOT/'inputs.example.json');self.assertIs(p.validate_config(cfg),cfg)
        self.assertNotIn('v6_runs',cfg);self.assertNotIn('h13_runs',cfg)
        cfg['h7_runs'].pop('44')
        with self.assertRaises(ValueError):p.validate_config(cfg)
        cfg=p.load(ROOT/'inputs.example.json');cfg['output_root']=cfg['data_root']
        with self.assertRaises(ValueError):p.validate_config(cfg)
        cfg=p.load(ROOT/'inputs.example.json');cfg['prior_study_root']='/workspace/previous_v2';cfg['output_root']=cfg['prior_study_root']+'/new'
        with self.assertRaises(ValueError):p.validate_config(cfg)   # never write into the v1.1 study
        cfg=p.load(ROOT/'inputs.example.json');cfg.pop('prior_study_root')
        with self.assertRaises(ValueError):p.validate_config(cfg)

    def run_fixture(self,run,seed=44,history=7):
        run.mkdir(parents=True)
        contract=dict(seed=seed,source_commit=p.SOURCE_COMMIT,data_revision=p.DATA_REVISION,
                      rq2=dict(history_tokens=history,warmup_tokens=25,target_tokens=30,input_tokens=55,snippet_frames=5,native_frame_stride=3))
        p.save(run/'run_manifest.json',dict(contract=contract))
        p.save(run/'training_status.json',dict(complete=True,completed_epochs=50,target_epochs=50,best_checkpoint='best_mAUC@_epoch_2.pth'))
        rows=[]
        for epoch in range(1,51):
            metrics={k:.9 if epoch in (2,3) else .7 for k in p.METRICS};metrics['threshold@0.10']=.6
            rows.append(dict(epoch=epoch,metrics=metrics))
        (run/'validation_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        (run/'best_mAUC@_epoch_2.pth').write_bytes(b'fake no deserialization')
        return contract

    def test_h7_baseline_seed_history_and_earliest_epoch(self):
        run=self.root/'h7';self.run_fixture(run)
        result=p.validate_baseline(run,44,7)
        self.assertEqual(result['selected_epoch'],2)
        with self.assertRaises(ValueError):p.validate_baseline(run,42,7)
        with self.assertRaises(ValueError):p.validate_baseline(run,44,13)
        status=p.load(run/'training_status.json');status['best_checkpoint']='best_mAUC@_epoch_3.pth';p.save(run/'training_status.json',status)
        with self.assertRaises(ValueError):p.history_selection(run)

    def prior_fixture(self,root,candidate='c2_lr_double'):
        (root/'comparison').mkdir(parents=True);p.save(root/'comparison/TEST_COMPARISON_COMPLETE.json',dict(complete=True))
        shas={str(s):'%064d'%s for s in p.SEEDS}
        p.save(root/'selection_frozen_before_test.json',dict(baselines=dict(h7={s:dict(checkpoint_sha256=v) for s,v in shas.items()}),
               chosen={a:dict(candidate=candidate) for a in ('m1','m1_m2','m1_m3')}))
        for role in p.PRIOR_ROLES:
            for s in p.SEEDS:
                f=root/'test/runs'/role/('seed_%d'%s)/'run';f.mkdir(parents=True)
                p.save(f/'test_predictions.json',[dict(id='%05d'%i,score=.5) for i in range(1344)])
        return {s:dict(checkpoint_sha256=v) for s,v in shas.items()}

    def test_prior_study_must_use_same_h7_and_config(self):
        root=self.root/'prior';rows=self.prior_fixture(root)
        result=p.validate_prior_study(root,rows)
        self.assertEqual(set(result['prediction_sha256']),{'%s_%d'%(r,s) for r in p.PRIOR_ROLES for s in p.SEEDS})
        bad=dict(rows);bad['43']=dict(checkpoint_sha256='f'*64)
        with self.assertRaises(ValueError):p.validate_prior_study(root,bad)       # another H7 checkpoint
        other=self.root/'prior2';rows2=self.prior_fixture(other,'c1_base')
        with self.assertRaises(ValueError):p.validate_prior_study(other,rows2)    # not matched to c2
        f=root/'test/runs/m1/seed_44/run/test_predictions.json';p.save(f,p.load(f)[:1000])
        with self.assertRaises(ValueError):p.validate_prior_study(root,rows)      # incomplete predictions
        (root/'comparison/TEST_COMPARISON_COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):p.validate_prior_study(root,rows)

    def test_eta_units_and_json_stable_binding(self):
        result=p.measured_eta(dict.fromkeys(p.ARMS,20),2,3)
        self.assertEqual(result['training_runs'],18);self.assertEqual(result['epochs'],900)
        self.assertAlmostEqual(result['training_hours'],900*20/3600)
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
