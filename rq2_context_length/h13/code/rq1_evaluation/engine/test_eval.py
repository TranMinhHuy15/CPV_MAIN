"""Small deterministic guards; no training and no synthetic benchmark claims."""
import copy
import io
import json
from pathlib import Path
import tempfile
import tarfile
import hashlib
from unittest.mock import patch

import numpy as np
from metrics import temporal_metrics
from run_all import import_runs, probe_video


def test_reports(frozen_root):
    """Exercise all report stages with synthetic curves, NOT benchmark results."""
    import shutil
    from metrics import report
    from run_all import ROOT, rows, save, summary
    solution = rows(ROOT / 'references/top/metadata_snapshot/solution.csv')
    with tempfile.TemporaryDirectory(prefix='synthetic_eval_report_test_') as tmp:
        output = Path(tmp)
        alignment = {r['id']: dict(event_s=3., onset_s=0., event_last_pts_s=2.9, onset_last_pts_s=0.)
                     for r in solution if int(r['target']) == 1}
        save(output / 'alignment_inferred.json', {'videos': alignment})
        for model in ('riskprop', 'top'):
            for seed in (42, 43, 44):
                relative = Path('runs') / model / ('seed_%d' % seed)
                (output / relative).mkdir(parents=True)
                shutil.copyfile(Path(frozen_root) / relative / 'test_predictions.json',
                                output / relative / 'test_predictions.json')
                predictions = json.loads((output / relative / 'test_predictions.json').read_text())
                directory = output / model / ('seed_%d' % seed)
                for p in predictions:
                    score = p['score']
                    save(directory / 'curves' / (p['id'] + '.json'), dict(id=p['id'], scores=[score / 4, score / 2, score],
                         times_s=[0., 1., 2.], coverage='full_observed_clip', uses_future=False, checkpoint_sha256='synthetic_test'))
                save(directory / 'capture_manifest.json', dict(complete=True, test_count=1344, checkpoint_sha256='synthetic_test'))
                result = report(output, model, seed, ROOT / 'references/top/metadata_snapshot', save)
                assert all(v == 0 for v in result['scalar_metric_replay_delta'].values())
                assert result['paper_metric_equivalence_confirmed'] is False
        combined = summary(output)
        assert abs(combined['models']['riskprop']['mean_sample_sd']['mAP']['mean'] - .8254507298) < 1e-9
        assert abs(combined['models']['top']['mean_sample_sd']['mAP']['mean'] - .7870802634) < 1e-9
        assert combined['models']['riskprop']['mean_sample_sd']['mTTA_observed_tp_only_s']['n'] == 3
        assert combined['paper_metric_equivalence_confirmed'] is False
    print('PASS: all six report stages, scalar replay exactly unchanged, mean/sample SD and equivalence flags (synthetic curves, not scientific results).')


def main():
    fake_probe = {'frames': [{'best_effort_timestamp_time': str(t)} for t in [0., .1, .2]],
                  'streams': [{'start_time': '0', 'duration': '.3'}]}
    with patch('run_all.subprocess.check_output', return_value=json.dumps(fake_probe)):
        clock = probe_video('fixture.mp4', 3)
        assert abs(clock['cut_s'] - .3) < 1e-12
        assert abs(clock['cut_minus_last_pts_s'] - .1) < 1e-12
    broken_probe = copy.deepcopy(fake_probe)
    broken_probe['streams'][0]['duration'] = '.2'
    with patch('run_all.subprocess.check_output', return_value=json.dumps(broken_probe)):
        try:
            probe_video('fixture.mp4', 3)
        except ValueError:
            pass
        else:
            raise AssertionError('Last frame was accepted as stream end.')
    solution = [{'id': 'P', 'target': '1'}, {'id': 'N', 'target': '0'}]
    curves = [dict(id='P', scores=[.1, .8, .9], times_s=[0., 1., 2.],
                   coverage='full_observed_clip', uses_future=False),
              dict(id='N', scores=[.1] * 9 + [.8], times_s=list(range(10)),
                   coverage='full_observed_clip', uses_future=False)]
    alignment = {'P': dict(event_s=3., onset_s=0., event_last_pts_s=2.9, onset_last_pts_s=0.)}
    result = temporal_metrics(curves, solution, alignment)
    assert abs(result['mTTA_observed_tp_only_s'] - 2.) < 1e-12
    assert result['max_measured_frame_FAR'] == .1
    assert abs(temporal_metrics(curves, solution, alignment, True)['mTTA_observed_tp_only_s'] - 1.9) < 1e-12
    outside = copy.deepcopy(alignment)
    outside['P']['onset_s'] = 2.5
    result = temporal_metrics(curves, solution, outside)
    assert result['mTTA_observed_tp_only_s'] is None and result['mTTA_observed_misses_zero_s'] == 0.
    assert result['positive_videos_with_no_observed_post_onset_endpoint'] == 1
    tied = copy.deepcopy(curves)
    tied[1]['scores'] = [.8] * 10
    try:
        temporal_metrics(tied, solution, alignment)
    except ValueError:
        pass
    else:
        raise AssertionError('Tied threshold exceeding FAR was accepted.')
    for mutate in ('duplicate', 'future', 'clock'):
        broken = copy.deepcopy(curves)
        if mutate == 'duplicate':
            broken.append(broken[0])
        elif mutate == 'future':
            broken[0]['uses_future'] = True
        else:
            broken[0]['times_s'] = [1., 1., 2.]
        try:
            temporal_metrics(broken, solution, alignment)
        except ValueError:
            pass
        else:
            raise AssertionError(mutate)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        content = b'{"own":true}'
        sha = hashlib.sha256(content).hexdigest()
        spec = [{'model': 'top', 'seed': 42, 'archive_prefix': 'jobs/seed_42/run', 'files': {'own.json': sha}}]
        archive = root / 'a.tar'
        with tarfile.open(archive, 'w') as t:
            # Even a malicious extra member is ignored, never unpacked.
            for name in ('jobs/seed_42/run/own.json', '../../escape.txt'):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                t.addfile(info, io.BytesIO(content))
        out = root / 'output'
        import_runs(archive, 'top', out, spec)
        import_runs(archive, 'top', out, spec)
        assert not (root / 'escape.txt').exists()
        saved = out / 'runs/top/seed_42/own.json'
        saved.write_bytes(b'altered')
        try:
            import_runs(archive, 'top', out, spec)
        except ValueError:
            pass
        else:
            raise AssertionError('Altered frozen artifact was accepted.')
    print('PASS: TTC formula, boundary sensitivity, unobservable onset, low-FAR ties, duplicate IDs, causality, clock, extraction scope and resume integrity.')


if __name__ == '__main__':
    main()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-root')
    args = parser.parse_args()
    if args.frozen_root:
        test_reports(args.frozen_root)
