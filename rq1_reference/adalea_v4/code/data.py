import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shared_reference import V6SampleFramesBeforeAccident, v6_calculate

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_records(data_root, verify_hashes=True):
    if not Path(data_root).is_dir():
        raise FileNotFoundError('Data root missing: %s' % data_root)
    expected = json.loads((ROOT / 'provenance.json').read_text())['annotations_sha256']
    annotation_path = ROOT / 'author_preprocessing/annotations.csv'
    if digest(annotation_path) != expected:
        raise RuntimeError('Bundled author annotations changed.')
    inventory = {r['id']: r for r in read_csv(ROOT / 'inference_inventory.csv')}
    validation = {r['video_id'].zfill(5): int(r['have_accident'])
                  for r in read_csv(ROOT / 'author_preprocessing/nexar_val_annotations.csv')}
    if len(validation) != 300 or sum(validation.values()) != 150:
        raise RuntimeError('Unexpected V6 validation split.')
    paths = {}
    for p in Path(data_root).rglob('*.mp4'):
        paths.setdefault(p.stem, []).append(p.resolve())
    records = {'train': [], 'validation': [], 'test': []}
    annotations = read_csv(annotation_path)
    if len(annotations) != 2844 or len(inventory) != 2844 or {r['id'] for r in annotations} != set(inventory):
        raise RuntimeError('Incomplete/duplicate video inventory.')
    for i, ann in enumerate(annotations):
        video_id, item = ann['id'], inventory[ann['id']]
        found = paths.get(video_id, [])
        if len(found) != 1:
            raise RuntimeError('Video %s found %d times; require exactly once.' % (video_id, len(found)))
        path = found[0]
        if path.stat().st_size != int(item['bytes']) or (verify_hashes and digest(path) != item['sha256']):
            raise RuntimeError('Video content differs from verified HF inventory: %s' % video_id)
        test = int(ann['is_test']) == 1
        if test != (item['split'] != 'train') or int(ann['target']) != (-1 if test else int(item['target'])):
            raise RuntimeError('Video split/label mismatch: %s' % video_id)
        target = None if test else bool(int(ann['target']))
        split = 'test' if test else ('validation' if video_id in validation else 'train')
        if split == 'validation' and int(target) != validation[video_id]:
            raise RuntimeError('Validation label mismatch.')
        records[split].append(dict(id=video_id, path=str(path), video_sha256=item['sha256'], total_frames=int(ann['total_frames']),
            native_fps=float(ann['fps']), target=target,
            event_frame=None if test or not target else int(ann['accident_frame']),
            onset_frame=None if test or not target else int(ann['abnormal_start_frame'])))
        if (i+1) % 250 == 0:
            print('Verified %d/2844 videos' % (i+1), flush=True)
    if {k: len(v) for k, v in records.items()} != {'train': 1200, 'validation': 300, 'test': 1344}:
        raise RuntimeError('Dataset split counts differ from V6.')
    return records


def sample_frames(record, train=False):
    sampler = V6SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=not train)
    result = sampler.transform(dict(total_frames=record['total_frames'], fps=30,
        accident_frame=record['event_frame'], target=record['target'], start_index=0))
    # V6 emits a risk at the last frame of each causal 5-frame snippet.
    frames = result['frame_inds'].reshape(30, 5)[:, -1]
    if np.any(frames >= record['total_frames']):
        raise RuntimeError('Sampled index outside native video.')
    return frames


def prediction_record(record, frames, scores):
    if record['target'] is None:
        return dict(id=record['id'], score=float(scores[-1]), frame_inds=frames.tolist(),
                    pred=scores.tolist(), coverage='last_30_v6_endpoints', uses_future=False,
                    native_fps=record['native_fps'])
    def anchor(frame):
        found = np.where(frames >= frame)[0]
        return int(found[0]) if len(found) else 0
    return dict(id=record['id'], pred=scores.tolist(), frame_inds=frames.tolist(),
        target=int(record['target']), native_fps=record['native_fps'],
        event_frame=record['event_frame'], onset_frame=record['onset_frame'],
        abnormal_start_ind=anchor(record['onset_frame']) if record['target'] else 0,
        accident_ind=anchor(record['event_frame']) if record['target'] else 0)


def validation_metrics(predictions):
    result = v6_calculate(SimpleNamespace(fpr_max=.1), [np.asarray(r['pred']) for r in predictions],
        [bool(r['target']) for r in predictions], [r['abnormal_start_ind'] for r in predictions],
        [r['accident_ind'] for r in predictions], '@')
    return {k: float(v) for k, v in result.items()}
