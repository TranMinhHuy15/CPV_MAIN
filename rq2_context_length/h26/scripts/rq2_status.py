"""Xem nhanh tien do RQ2 (chi doc file, khong sua gi). Dung: python3 rq2_status.py [output_root]"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/rq2_lstm_checked')
KEYS = ['mAUC@', 'mAP@', 'AP@0.5s', 'AP@1.0s', 'AP@1.5s', 'mAUC@0.1', 'mtta@0.1']
TEST = ['mAP', 'mAUC_grouped', 'mPAUC_raw_divided_by_0.1', 'mTTA_test_author_index10_inferred_s',
        'pAUC_raw_div_0.1@1.0s', 'pAUC_raw_div_0.1@1.5s', 'AP@1.5s']


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return None


def history(run):
    rows = {}
    p = run / 'validation_history.jsonl'
    if p.is_file():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                d = json.loads(line)
                rows[int(d['epoch'])] = d['metrics']
    return rows


best_by = {}
for variant in ('short', 'long'):
    for seed in (42, 43, 44):
        ws = ROOT / variant / ('seed_%d' % seed)
        if not ws.exists():
            continue
        run = ws / 'run'
        print('\n===== %s seed %d =====' % (variant.upper(), seed))
        smoke = load(ws / 'smoke_report.json')
        if smoke:
            print('SMOKE: passed=%s | GPU %s | peak VRAM %.2f GB | %.3f s/microstep | %d microstep/epoch | uoc tinh train 50 epoch ~%.1f gio (chua gom val/test)'
                  % (smoke.get('passed'), smoke.get('gpu'), smoke.get('peak_memory_gb_max_across_gpus', float('nan')),
                     smoke.get('seconds_per_microstep', float('nan')), smoke.get('train_microsteps_per_epoch_per_rank', -1),
                     smoke.get('rough_train_only_50_epoch_hours', float('nan'))))
        status = load(run / 'training_status.json')
        if status:
            print('TRAIN: %d/%d epoch | complete=%s | best=%s' % (status['completed_epochs'], status['target_epochs'],
                                                                 status['complete'], status['best_checkpoint']))
        h = history(run)
        if h:
            be = max(h, key=lambda e: h[e].get('mAUC@', -1))
            best_by[(variant, seed)] = h[be]
            last = max(h)
            print('%-12s ' % 'val' + ' '.join('%9s' % k for k in KEYS))
            for tag, e in (('best ep%d' % be, be), ('last ep%d' % last, last)):
                print('%-12s ' % tag + ' '.join('%9.4f' % h[e].get(k, float('nan')) for k in KEYS))
        done = load(run / 'evaluation_complete.json')
        rep = load(run / 'rq2_test_report.json')
        if done and rep:
            m = rep.get('summary_metrics', {})
            print('TEST (da niem phong): ' + ' | '.join('%s=%.4f' % (k, m[k]) for k in TEST if k in m))
        elif status and status.get('complete'):
            print('TEST: chua chay (dang khoa hoac --validation-only)')

print('\n===== SO SANH LONG - SHORT tren VALIDATION (epoch tot nhat) =====')
for seed in (42, 43, 44):
    s, l = best_by.get(('short', seed)), best_by.get(('long', seed))
    if s and l:
        print('seed %d: ' % seed + ' | '.join('%s %+.4f' % (k, l.get(k, 0) - s.get(k, 0)) for k in KEYS))
summary = load(ROOT / 'summary.json')
if summary and summary.get('mean_sample_sd'):
    print('\n===== TEST mean +- SD (chi khi du 3 seed/nhanh) =====')
    for variant, ms in summary['mean_sample_sd'].items():
        print(variant, ' | '.join('%s %.4f+-%.4f' % (k, ms[k]['mean'], ms[k]['sample_sd']) for k in TEST if k in ms))
