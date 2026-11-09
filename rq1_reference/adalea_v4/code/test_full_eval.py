"""CPU fixtures: not a paid-GPU certification and not model performance."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data import ROOT, digest, read_csv, sample_frames, save_json
from features import FeatureStore
from full_eval import (aggregate_full, endpoints, evaluate_seed, load_selected_checkpoint,
    probe_clock, rolling_scores, validate_curve, windows, TEMPORAL_KEYS)
from model import AdaLEA
from runner import atomic_checkpoint, cpu_state, make_model
from temporal_metrics import make_anchors, temporal_metrics
from test_adalea import FixtureStore, FakeExtractor, record, tiny_config

torch.set_num_threads(1)


def curve(video_id, scores, times=None):
    return dict(id=video_id, scores=scores, times_s=times or list(np.arange(len(scores))*.1))


class FullChecks(unittest.TestCase):
    def fixture(self):
        curves = [curve('p', [.2,.9], [0.,.3]), curve('n', [.5]*20)]
        labels = dict(p=1,n=0)
        anchors = dict(p=dict(onset_index=0,event_index=10,event_s=1.,onset_s=0.))
        return curves, labels, anchors

    def test_physical_and_source_clocks_are_separate(self):
        result = temporal_metrics(*self.fixture())
        self.assertAlmostEqual(result[TEMPORAL_KEYS[0]], .9)
        self.assertAlmostEqual(result[TEMPORAL_KEYS[1]], .7)
        self.assertAlmostEqual(result[TEMPORAL_KEYS[2]], .7)
        self.assertEqual(result['maximum_actual_negative_frame_FAR_source'],1.)
        self.assertLessEqual(result['maximum_actual_negative_frame_FAR_physical_strict'],.1)
        self.assertFalse(result['paper_test_equivalence_confirmed'])

    def test_missing_onset_and_misses_remain_in_denominator(self):
        curves, labels, anchors = self.fixture()
        curves.append(curve('future', [.99]*4))
        labels['future'] = 1
        anchors['future'] = dict(onset_index=20,event_index=30,event_s=3.,onset_s=2.)
        result = temporal_metrics(curves,labels,anchors)
        self.assertAlmostEqual(result[TEMPORAL_KEYS[0]], .45)
        self.assertAlmostEqual(result[TEMPORAL_KEYS[1]], .35)
        self.assertEqual(result['positives_without_observed_physical_onset_window'],1)
        curves[0]['scores'] = [.1,.1]
        self.assertEqual(temporal_metrics(curves,labels,anchors)[TEMPORAL_KEYS[1]],0.)

    def test_temporal_no_post_event_alarm(self):
        curves, labels, anchors = self.fixture()
        curves[0] = curve('p',[.1,.99],[0.,1.5])
        self.assertEqual(temporal_metrics(curves,labels,anchors)[TEMPORAL_KEYS[1]],0.)

    def test_source_arithmetic_parity(self):
        from collections import OrderedDict
        text = (ROOT/'shared_reference.py').read_text()
        function = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='v6_calculate')
        cut = next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='preds_0' for t in n.targets))
        function.body = function.body[:cut]+[ast.Return(ast.Name('eval_results',ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        namespace = dict(np=np,OrderedDict=OrderedDict,Dict=dict)
        exec(compile(module,'<source-prefix>','exec'),namespace)
        rng = np.random.RandomState(8)
        arrays = [rng.rand(30) for _ in range(7)]
        labels = [1,1,1,0,0,0,0]
        starts,events = [3,8,12,0,0,0,0],[35,38,39,0,0,0,0]
        expected = namespace['v6_calculate'](SimpleNamespace(fpr_max=.1),arrays,labels,starts,events,'@')['mtta@0.1']
        curves = [curve(str(i),p.tolist()) for i,p in enumerate(arrays)]
        anchors = {str(i):dict(onset_index=starts[i],event_index=events[i],onset_s=0.,event_s=4.) for i in range(3)}
        result = temporal_metrics(curves,{str(i):int(y) for i,y in enumerate(labels)},anchors)
        self.assertAlmostEqual(result[TEMPORAL_KEYS[0]],expected,places=12)

    def test_bad_temporal_inputs_fail_closed(self):
        for change in ('nan','order','duplicate','missing'):
            curves,labels,anchors=self.fixture()
            if change=='nan': curves[0]['scores'][0]=float('nan')
            if change=='order': curves[0]['times_s']=[.3,0.]
            if change=='duplicate': curves.append(curves[0])
            if change=='missing': anchors={}
            with self.assertRaises(ValueError): temporal_metrics(curves,labels,anchors)

    def test_window_grid_causal_and_tail_matches_v3(self):
        for n in (1,2,3,50,160,301):
            ends=endpoints(n)
            w=windows(ends)
            self.assertEqual(ends[-1],n-1)
            self.assertTrue(np.all(w<=ends[:,None]))
            self.assertTrue(np.all(w>=0))
        r=record('T',test=True)
        np.testing.assert_array_equal(windows([159])[0],sample_frames(r,False))

    def test_rolling_exact_manual_and_label_guard(self):
        model=AdaLEA(8,4,8,0.).eval()
        r=record('T',test=True)
        ends,scores=rolling_scores(model,r,FixtureStore(),torch.device('cpu'),time.time()+30,8)
        expected=[]
        for end in ends:
            features=FixtureStore().get(r,windows([end])[0])
            tensors=[torch.from_numpy(a[None]) for a in features]
            with torch.no_grad(): expected.append(float(model(tensors[0].float(),tensors[1].float(),tensors[2].bool())[0][0,-1].sigmoid()))
        np.testing.assert_allclose(scores,expected,atol=1e-6,rtol=1e-6)
        with self.assertRaises(ValueError): rolling_scores(model,record('A'),FixtureStore(),torch.device('cpu'),time.time()+30)

    def test_legacy_cache_is_never_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=Path(tmp)/'old',Path(tmp)/'new'
            old.mkdir()
            np.savez_compressed(old/'A.npz',indices=np.array([0,3],np.int32),
                global_features=np.zeros((2,8),np.float16),local_features=np.zeros((2,2,8),np.float16),mask=np.zeros((2,2),bool))
            before=digest(old/'A.npz')
            store=FeatureStore(new,FakeExtractor(),tiny_config(),read_only_fallback=old)
            with patch('features.native_jpeg_frames',return_value={6:np.zeros((8,8,3),np.uint8)}):
                store.get(record('A'),[0,3,6])
            self.assertEqual(digest(old/'A.npz'),before)
            self.assertTrue((new/'A.npz').exists())

    def test_ffprobe_clock_count_boundary_validation(self):
        report=dict(frames=[dict(best_effort_timestamp_time=str(i*.1)) for i in range(3)],
            streams=[dict(duration='.3',start_time='0')])
        with patch('full_eval.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(report))):
            pts,cut=probe_clock(dict(id='X',path='fixture',total_frames=3),time.time()+30)
        self.assertEqual(len(pts),3)
        self.assertEqual(cut,.3)
        report['streams'][0]['duration']='.2'
        with patch('full_eval.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(report))), self.assertRaises(ValueError):
            probe_clock(dict(id='X',path='fixture',total_frames=3),time.time()+30)

    def test_full_1344_three_seed_capture_report_and_resume(self):
        """Real ID/metadata structure, tiny synthetic predictions/model; NOT research results."""
        cfg=tiny_config()
        solution=read_csv(ROOT/'metadata_snapshot/solution.csv')
        rows=[dict(record(r['id'],test=True),total_frames=6,video_sha256='fixture-video') for r in solution]
        contract={'fixture':'V4 end-to-end, not real checkpoint','config':cfg}
        model=make_model(cfg,torch.device('cpu'))
        def fake_clock(r,deadline): return [i/30. for i in range(6)],.2
        def fake_rolling(model,r,store,device,deadline,batch_windows): return endpoints(6),[.5,.5]
        with tempfile.TemporaryDirectory() as tmp, patch('full_eval.probe_clock',side_effect=fake_clock), patch('full_eval.rolling_scores',side_effect=fake_rolling) as predict:
            for seed in (42,43,44):
                directory=Path(tmp)/('seed_%d'%seed)
                directory.mkdir()
                predictions=[dict(id=r['id'],score=.5) for r in rows]
                save_json(directory/'test_predictions.json',predictions)
                state=dict(contract=contract,seed=seed,completed_epochs=50,best_epoch=1,adaptive_epoch=50,
                    best_model=cpu_state(model),history=[dict(epoch=i,metrics={cfg['selection']:.6}) for i in range(1,51)])
                atomic_checkpoint(directory/'resume_latest.pth',state)
                save_json(directory/'completion.json',dict(seed=seed,complete=True,completed_epochs=50,all_test_ids=1344,
                    best_epoch=1,contract=contract,checkpoint_sha256=digest(directory/'resume_latest.pth'),
                    predictions_sha256=digest(directory/'test_predictions.json')))
                result=evaluate_seed(cfg,seed,rows,FixtureStore(),directory,torch.device('cpu'),contract,time.time()+180)
                self.assertTrue(result['complete'])
                self.assertTrue(all(np.isfinite(result['temporal']['Combined'][k]) for k in TEMPORAL_KEYS))
            summary=aggregate_full(tmp,cfg,contract)
            self.assertEqual(summary['n'],3)
            self.assertTrue((Path(tmp)/'full_metrics_summary.csv').is_file())
            count=predict.call_count
            evaluate_seed(cfg,42,rows,FixtureStore(),Path(tmp)/'seed_42',torch.device('cpu'),contract,time.time()+180)
            self.assertEqual(predict.call_count,count,'Completed curves should resume without inference')
            sample=Path(tmp)/'seed_42/full_test_curves'/(rows[0]['id']+'.json')
            data=json.loads(sample.read_text()); data['scores'][0]=.51; save_json(sample,data)
            with self.assertRaises(ValueError): aggregate_full(tmp,cfg,contract)
            with self.assertRaises(ValueError): evaluate_seed(cfg,42,rows,FixtureStore(),Path(tmp)/'seed_42',torch.device('cpu'),contract,time.time()+180)

    def test_legacy_v3_source_model_unchanged(self):
        expected=json.loads((ROOT/'legacy_v3_contract_files.json').read_text())
        self.assertEqual(digest(ROOT/'model.py'),expected['files']['model.py'])
        self.assertEqual(digest(ROOT/'runner.py'),expected['files']['runner.py'])
        self.assertEqual(digest(ROOT/'shared_reference.py'),expected['files']['shared_reference.py'])
        self.assertEqual(digest(ROOT/'shared_test_metrics.py'),expected['files']['shared_test_metrics.py'])
        cfg=json.loads((ROOT/'config.json').read_text())
        for key,value in expected['config'].items():
            if key!='protocol': self.assertEqual(cfg[key],value)


if __name__=='__main__': unittest.main(verbosity=2)
