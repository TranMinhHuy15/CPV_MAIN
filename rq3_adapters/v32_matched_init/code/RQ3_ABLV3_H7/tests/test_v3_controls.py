"""Regression tests for matched head initialization, guards, ETA, validation logging and pilot recovery."""
import contextlib
import copy
import importlib.util
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
import diagnostics as D
import study_guard as G
import models


class V3Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name='test_v6_routes' if ROOT.name.endswith('V6') else 'test_ablation_routes'
        spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/(name+'.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        cls.route=fixture.V6Routes if ROOT.name.endswith('V6') else fixture.AblationRoutes
        cls.route.setUpClass()

    def setUp(self):
        scratch=ROOT/'tests/scratch';scratch.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch);self.work=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_integrated_six_heads_share_init_and_leave_same_training_rng(self):
        for seed in (42,43,44):
            states=[];rngs=[]
            for arm in p.ARMS:
                torch.manual_seed(seed)
                with patch.dict(sys.modules,self.route.module_map):
                    head=self.route.Head(arm=arm,width=4,in_channels=8,adapter_init_seed=seed)
                states.append({k:v.clone() for k,v in head.adapter.state_dict().items()});rngs.append(torch.get_rng_state().clone())
                self.assertEqual(head.adapter.initialization_seed,seed)
            common=set.intersection(*(set(state) for state in states))
            self.assertIn('residual_gate.weight',common)
            for state,rng in zip(states,rngs):
                for name in common:self.assertTrue(torch.equal(states[0][name],state[name]),(seed,name))
                self.assertTrue(torch.equal(rngs[0],rng),'model creation changed training RNG by arm')

    def test_builder_is_rng_isolated_and_independent_of_ambient_random_consumption(self):
        torch.manual_seed(1);before=torch.get_rng_state().clone()
        first=models.build_adapter('r7_last_state',4,.1,channels=8,init_seed=44)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.manual_seed(999);torch.rand(123);before=torch.get_rng_state().clone()
        second=models.build_adapter('r7_last_state',4,.1,channels=8,init_seed=44)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for k,v in first.state_dict().items():self.assertTrue(torch.equal(v,second.state_dict()[k]))
        with self.assertRaises(ValueError):models.build_adapter('a7_m1',4,.1,channels=8,init_seed=True)

    def test_training_dropout_rng_consumption_same_across_six_heads(self):
        feature=torch.randn(2,55,8,4);states=[]
        for arm in p.ARMS:
            torch.manual_seed(42)
            with patch.dict(sys.modules,self.route.module_map):head=self.route.Head(arm=arm,width=4,in_channels=8)
            head.train();torch.manual_seed(77);head.score_features(feature)
            states.append(torch.get_rng_state().clone())
            self.assertIn('gate',head.last_diagnostics)
            self.assertFalse(head.last_diagnostics['gate'].requires_grad)
        self.assertTrue(all(torch.equal(states[0],s) for s in states))

    def test_eta_caller_handles_coverage_zero_through_six(self):
        import reporting
        for n in range(7):
            with contextlib.redirect_stdout(io.StringIO()):r=reporting.estimate(self.work)
            self.assertEqual(r['training_runs'],18);self.assertEqual(r['epochs'],900)
            self.assertEqual(isinstance(r['current_GPU_projection'],dict),n==6)
            if n<6:
                p.save(self.work/'trials/c2_lr_double'/p.ARMS[n]/'seed_42/run/named_epochs.json',[dict(epoch=1,seconds=20)])

    def test_plan_code_mutation_and_old_output_rejected(self):
        package=self.work/'package';package.mkdir();(package/'legacy').mkdir()
        (package/'x.py').write_text('VALUE=1\n');(package/G.PLAN).write_text('fixed plan\n')
        out=self.work/'new';out.mkdir()
        p.save(out/'study_contract.json',dict(study_version=G.VERSION,code=G.code_hashes(package),analysis_plan=G.plan_identity(package)))
        G.archive_plan(out,package);G.require_study(out,package)
        (package/G.PLAN).write_text('changed plan\n')
        with self.assertRaises(ValueError):G.require_study(out,package)
        (package/G.PLAN).write_text('fixed plan\n');(package/'x.py').write_text('VALUE=2\n')
        with self.assertRaises(ValueError):G.require_study(out,package)
        old=self.work/'v2';old.mkdir();p.save(old/'study_contract.json',dict(study_version='V2'))
        with self.assertRaises(ValueError):G.ensure_output(old)

    def test_fast_bootstrap_ties_duplicates_equal_reference_metrics(self):
        import ablation_stats as S
        y=np.tile([0,1],60);g=np.repeat([0,1,2],40)
        score=np.random.default_rng(4).integers(0,8,size=120)/8.
        index=S.shared_indices(y,g,40)
        point,draws=S.role_draws(y,g,score,index)
        for i,ix in enumerate(index):
            ref=S.metrics(y,g,score,ix)
            for metric in S.METRICS:self.assertAlmostEqual(draws[metric][i],ref[metric],places=13)
        self.assertEqual(point,S.metrics(y,g,score))

    def test_worker_config_retention_and_prepared_source_are_bound(self):
        package=self.work/'package';package.mkdir();(package/G.PLAN).write_text('fixed')
        out=self.work/'out';out.mkdir();cfg=dict(compact_completed_trials=False)
        prepared=dict(source_root=str((self.work/'source').resolve()),baselines={'v6':{}})
        p.save(out/'study_contract.json',dict(study_version=G.VERSION,code=G.code_hashes(package),analysis_plan=G.plan_identity(package),
               config=cfg,baselines=prepared['baselines'],source_root=prepared['source_root']))
        G.archive_plan(out,package);G.require_worker_inputs(out,package,cfg,prepared)
        with self.assertRaises(ValueError):G.require_worker_inputs(out,package,dict(compact_completed_trials=True),prepared)
        with self.assertRaises(ValueError):G.require_worker_inputs(out,package,cfg,dict(prepared,source_root=str(self.work/'changed')))

    def test_cross_base_rejects_analysis_code_changed_after_bootstrap(self):
        comparison=self.work/'comparison';comparison.mkdir()
        meta=dict(study_version=G.VERSION,base='h7',analysis_plan_sha256='a'*64,
                  analysis_code_sha256=G.shared_analysis_hashes(ROOT),resamples=100)
        np.savez(comparison/'bootstrap_draws.npz',metadata_json=np.array(json.dumps(meta)),**{'draw|a7_m1|42|mAP':np.zeros(100)})
        (comparison/'paired_bootstrap.csv').write_text('fixture')
        p.save(comparison/'paired_bootstrap_decisions.json',{})
        G.finish_bootstrap(self.work,meta)
        G.require_bootstrap(self.work,'h7','a'*64)
        with patch.object(G,'shared_analysis_hashes',return_value={'changed':'hash'}):
            with self.assertRaises(ValueError):G.require_bootstrap(self.work,'h7','a'*64)

    def test_diagnostic_hook_precedes_commit_and_survives_postcommit_interruption(self):
        import console
        class Registry:
            def __init__(self):self.values={}
            def get(self,name):return self.values.get(name)
            def register_module(self):
                def register(cls):self.values[cls.__name__]=cls;return cls
                return register
        registry=Registry()
        modules={'mmengine':types.ModuleType('mmengine'),'mmengine.hooks':types.SimpleNamespace(Hook=object),
                 'mmengine.registry':types.SimpleNamespace(HOOKS=registry)}
        with patch.dict(sys.modules,modules):console.register_hook()
        cls=registry.values['MatchedValidationDiagnosticsHook']
        self.assertLess(cls.priority,90);self.assertGreater(registry.values['VastNamedLogHook'].priority,90)
        self.assertFalse(hasattr(registry.values['VastNamedLogHook'],'after_val_iter'))
        diag=dict(gate=torch.ones(1,2)*.5,raw=torch.zeros(1,2),correction=torch.zeros(1,2),attention=[],scale_weights=None)
        runner=types.SimpleNamespace(epoch=1,work_dir=str(self.work),model=types.SimpleNamespace(cls_head=types.SimpleNamespace(last_diagnostics=diag)))
        hook=cls(seed=42,arm='a7_m1');hook.before_val_epoch(runner)
        hook.after_val_iter(runner,0);hook.after_val_iter(runner,1)
        with contextlib.redirect_stdout(io.StringIO()):hook.after_val_epoch(runner)
        # At this point original priority90 commit can occur; interrupt before named priority100 hook.
        self.assertEqual(json.loads((self.work/D.FILES[0]).read_text())['epoch'],1)
        hook2=cls(seed=42,arm='a7_m1');hook2.before_train(runner)
        self.assertEqual(len((self.work/D.FILES[0]).read_text().splitlines()),1)
        for name in D.FILES:D.append(self.work/name,dict(epoch=2,batch=1))
        hook2.before_train(runner)
        self.assertTrue(all(json.loads(s)['epoch']==1 for s in (self.work/D.FILES[0]).read_text().splitlines()))

    def test_validation_epoch_one_samples_and_resume_trim_are_aligned(self):
        diag=dict(gate=torch.tensor([[.99,.99]]),raw=torch.tensor([[-5.,-5.]]),correction=torch.tensor([[-.987,-.987]]),
                  attention=[torch.tensor([[[.5,.5],[.5,.5]]])],scale_weights=None)
        hook=types.SimpleNamespace(arm='a26_m1',seed=44,diag_moments=D.Moments())
        runner=types.SimpleNamespace(epoch=1,work_dir=str(self.work),model=types.SimpleNamespace(cls_head=types.SimpleNamespace(last_diagnostics=diag)))
        D.capture_validation(hook,runner,0);D.capture_validation(hook,runner,1)
        with contextlib.redirect_stdout(io.StringIO()):D.finish_validation(hook,runner)
        sample=[json.loads(s) for s in (self.work/D.FILES[1]).read_text().splitlines()]
        self.assertEqual([(r['epoch'],r['batch']) for r in sample],[(1,1),(1,2)])
        summary=json.loads((self.work/D.FILES[0]).read_text())
        self.assertEqual(summary['values']['correction']['fraction_abs_above_0_95'],1.)
        for name in D.FILES:D.append(self.work/name,dict(epoch=2,batch=1))
        D.trim_uncommitted(self.work,1)
        for name in D.FILES:self.assertTrue(all(json.loads(s)['epoch']==1 for s in (self.work/name).read_text().splitlines()))
        self.assertEqual(len(list(self.work.glob('*uncommitted*'))),2)
        bad=dict(diag,raw=torch.tensor([[float('nan')]]))
        with self.assertRaises(FloatingPointError):D.Moments().update(bad)

    def test_diagnostics_complete_epoch_inventory_and_duplicates(self):
        values={k:dict(mean=.1,std=0.,count=30) for k in ('gate','raw','correction')}
        for epoch in range(1,51):
            D.append(self.work/D.FILES[0],dict(epoch=epoch,values=values,affects_selection=False))
            for batch in (1,2):D.append(self.work/D.FILES[1],dict(epoch=epoch,batch=batch))
        self.assertEqual(len(D.verify_validation(self.work)),50)
        D.append(self.work/D.FILES[1],dict(epoch=50,batch=2))
        with self.assertRaises(ValueError):D.verify_validation(self.work)

    def test_optional_prior_and_config_maker_preserve_v2(self):
        import make_inputs_from_v2 as maker
        cfg=p.load(ROOT/'inputs.example.json');cfg['output_root']=str(self.work/'v2')
        cfg['data_root']=str(self.work/'data');cfg['runtime_root']=str(self.work/'runtime');cfg['source_zip']=str(self.work/'source.zip')
        cfg[maker.BASE+'_runs']={str(s):str(self.work/'bases'/str(s)) for s in p.SEEDS}
        cache=Path(cfg['output_root'])/'caches/seed_42';cache.mkdir(parents=True);p.save(cache/'cache_manifest.json',{})
        path=self.work/'old.json';p.save(path,cfg);before=path.read_bytes()
        new=maker.make(path,self.work/'v3');self.assertEqual(path.read_bytes(),before)
        self.assertIsNone(new['prior_study_root']);self.assertFalse(new['compact_completed_trials'])
        self.assertEqual(new['cache_roots']['42'],str(cache.resolve()))
        with self.assertRaises(ValueError):maker.make(path,Path(cfg['output_root'])/'nested')

    def test_pilot_supervisor_stops_at_two_then_resumes_to_three(self):
        import run
        cfg=p.load(ROOT/'inputs.example.json');cfg['output_root']=str(self.work/'out');cfg['runtime_root']=str(self.work/'runtime')
        out=Path(cfg['output_root']);out.mkdir();p.save(out/'study_contract.json',dict(fixture=True))
        py=Path(cfg['runtime_root'])/'venv/bin/python';py.parent.mkdir(parents=True);py.write_bytes(b'fixture never executed')
        trial=out/'trials/c2_lr_double'/p.ARMS[0]/'seed_42/run'
        stops=[];commits=[]
        def command(args,label,log,env=None,accepted=(0,)):
            if '--stop-after-epoch' in args:
                epoch=int(args[args.index('--stop-after-epoch')+1]);stops.append(epoch);self.assertEqual(accepted,(75,))
                p.save(trial/'training_status.json',dict(completed_epochs=epoch,target_epochs=50,complete=False))
                (trial/'validation_history.jsonl').write_text(''.join(json.dumps(dict(epoch=e))+'\n' for e in range(1,epoch+1)))
        state=types.SimpleNamespace(assert_committed=lambda path,status:commits.append(status['completed_epochs']))
        with patch.object(sys,'argv',['run.py','--config','ignored.json','pilot']),patch.object(run,'load',side_effect=lambda path:cfg if str(path)=='ignored.json' else p.load(path)),\
             patch.object(run,'prepare',return_value=(out,self.work/'source',{})),patch.object(run,'command',side_effect=command),\
             patch.dict(sys.modules,{'training_state':state}),contextlib.redirect_stdout(io.StringIO()):run.main()
        self.assertEqual(stops,[2,3]);self.assertEqual(commits,[2,3])
        self.assertTrue(p.load(out/'PILOT_RESUME_RECEIPT.json')['complete'])


if __name__=='__main__':unittest.main(verbosity=2)
