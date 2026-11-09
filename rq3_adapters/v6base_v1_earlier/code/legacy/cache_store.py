"""FP32 temporal cache with per-record atomic commits; no pickle deserialization."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

PROTOCOL = 'rq3-h7-fixed-view-temporal-cache-fp32-v1'
SEEDS = (42, 43)  # User holds seed44 until its new H7 run is independently checked.
ARMS = ('m1', 'm1_m2', 'm1_m3', 'full')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bind(root, header):
    root = Path(root)
    path = root / 'cache_header.json'
    if path.exists():
        if load(path) != header:
            raise ValueError('Cache identity changed; use another directory.')
    else:
        save(path, header)
    return object_sha(header)


def safe_relative(root, relative):
    path = Path(relative)
    root = Path(root).resolve()
    result = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or root not in result.parents:
        raise ValueError('Unsafe cache path: ' + str(relative))
    return result


def validate_features(features):
    import numpy as np
    if (features.dtype != np.float32 or features.ndim != 5 or
            features.shape[:2] != (55, 2048) or features.shape[2] < 3 or
            features.shape[3:] != (1, 1) or not np.isfinite(features).all()):
        raise ValueError('Require finite FP32 [55,2048,T+1,1,1], T>=2 for M3, plus static column.')


def validate_metadata(record):
    import numpy as np
    m = record['metadata']
    split, video_id = record['split'], record['video_id']
    if (split not in ('train','validation') or not isinstance(video_id,str) or not video_id.isalnum()
            or record['file'] != '%s/%s.npz' % (split,video_id)
            or m.get('video_id') != video_id or type(m.get('target')) is not bool
            or m.get('is_test') is not False or type(m.get('is_val')) is not bool
            or m['is_val'] != (split == 'validation') or m.get('clip_len') != 5
            or m.get('num_clips') != 55 or m.get('frame_interval') != 3
            or type(m.get('total_frames')) is not int or m['total_frames'] < 1):
        raise ValueError('Cache metadata alignment/split changed.')
    indices = np.asarray(m.get('frame_inds', []))
    if (indices.shape != (275,) or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices < 0) or np.any(indices >= m['total_frames'])):
        raise ValueError('Cache native-frame indices invalid.')
    grid = indices.reshape(55,5)
    targets = grid[25:,-1]
    prefix = np.maximum(targets[0] - 3*np.arange(25,0,-1),0)
    ends = np.concatenate([prefix,targets])
    expected = np.maximum(ends[:,None]-3*np.arange(4,-1,-1),0)
    if np.any(np.diff(targets) < 0) or not np.array_equal(grid,expected):
        raise ValueError('Cache warmup/native snippet grid differs from source H7/H13/H26 grid.')


def commit(root, split, video_id, features, metadata, header_sha):
    import numpy as np
    if split not in ('train', 'validation') or not video_id.isalnum():
        raise ValueError('Invalid split/ID.')
    validate_features(features)
    root = Path(root)
    relative = '%s/%s.npz' % (split, video_id)
    path = safe_relative(root, relative)
    marker = path.with_suffix('.json')
    if marker.exists():
        record = load(marker)
        read_record(root, record, header_sha)
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash may leave an uncommitted array. Preserve it; never silently adopt it.
    if path.exists():
        path.rename(path.with_name(path.name + '.orphan.%d' % time.time_ns()))
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, features=features)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    record = dict(split=split, video_id=video_id, file=relative,
                  sha256=sha(path), bytes=path.stat().st_size,
                  shape=list(features.shape), metadata=metadata, header_sha256=header_sha)
    validate_metadata(record)
    save(marker, record)  # Commit point, only after array bytes are durable.
    return record


def read_record(root, record, header_sha, verify_hash=True):
    import numpy as np
    if record['header_sha256'] != header_sha:
        raise ValueError('Record belongs to another seed/cache.')
    path = safe_relative(root, record['file'])
    if path.stat().st_size != record['bytes'] or (verify_hash and sha(path) != record['sha256']):
        raise ValueError('Corrupt cache record: ' + record['file'])
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'features'}:
            raise ValueError('Unexpected cache fields.')
        features = data['features']
    validate_features(features)
    if list(features.shape) != record['shape']:
        raise ValueError('Cache shape changed.')
    validate_metadata(record)
    return features


def seal(root, header, records, support):
    counts = {s: sum(r['split'] == s for r in records) for s in ('train', 'validation')}
    if counts != {'train': 1200, 'validation': 300}:
        raise ValueError('Cannot seal incomplete train/validation cache: ' + str(counts))
    ids = [r['video_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicated IDs or train/validation leakage.')
    for record in records:
        validate_metadata(record)
    parity = {split: load(Path(root)/('parity_'+split+'.json')) for split in ('train','validation')}
    if any(row.get('passed') is not True for row in parity.values()):
        raise ValueError('Both source online/cache parity checks must pass before sealing.')
    value = dict(header=header, header_sha256=object_sha(header), records=records,
                 support_files=support, counts=counts, parity_checks=parity, complete=True)
    save(Path(root) / 'cache_manifest.json', value)
    return value


def verify(root, seed, expected_checkpoint, expected_tools, full=True):
    root = Path(root)
    value = load(root / 'cache_manifest.json')
    h = value['header']
    if (value.get('complete') is not True or h != load(root / 'cache_header.json')
            or h.get('protocol') != PROTOCOL or h.get('seed') != seed
            or h.get('checkpoint_sha256') != expected_checkpoint or h.get('tools') != expected_tools
            or value['header_sha256'] != object_sha(h)
            or value['counts'] != {'train': 1200, 'validation': 300}):
        raise ValueError('Incomplete/wrong seed/checkpoint/protocol cache.')
    records = value['records']
    counts = {s: sum(r['split'] == s for r in records) for s in ('train', 'validation')}
    if counts != value['counts'] or len({r['video_id'] for r in records}) != 1500:
        raise ValueError('Cache membership/count mismatch.')
    if any(value.get('parity_checks',{}).get(split,{}).get('passed') is not True for split in ('train','validation')):
        raise ValueError('Cache lacks source online/cache parity receipts.')
    for relative, digest in value['support_files'].items():
        if sha(safe_relative(root, relative)) != digest:
            raise ValueError('Cache support file changed: ' + relative)
    for i, record in enumerate(records, 1):
        if full:
            read_record(root, record, value['header_sha256'])
        if i % 250 == 0:
            print('CACHE VERIFY %d/1500' % i, flush=True)
    return value, sha(root / 'cache_manifest.json')


@contextlib.contextmanager
def fixed_rng(seed, split, video_id):
    import numpy as np
    import torch
    n = int(object_sha(dict(seed=seed, split=split, video_id=video_id, view=0))[:8], 16)
    py, nr = random.getstate(), np.random.get_state()
    # Dataset CPU transforms only. Preserve unrelated model RNG.
    tr = torch.get_rng_state()
    try:
        random.seed(n)
        np.random.seed(n)
        torch.random.default_generator.manual_seed(n)
        yield n
    finally:
        random.setstate(py)
        np.random.set_state(nr)
        torch.set_rng_state(tr)
