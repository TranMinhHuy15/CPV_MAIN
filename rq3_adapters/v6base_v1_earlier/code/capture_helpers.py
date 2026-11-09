"""Reviewed cache/parity helpers; no main entry point or training selection."""
import os
import math
import shutil
import time
from pathlib import Path
from protocol import load,save,sha

def budget(deadline, reserve=0):
    if time.time() + reserve >= deadline:
        raise TimeoutError('BUDGET STOP: Save Version; attach this output as PREVIOUS_OUTPUT_ROOT to resume.')

def feature_array(output, borrowed, identity, encode, export):
    """A label-free per-video cache, bound to H7/clock/runtime/preprocessing."""
    import numpy as np
    output = Path(output)
    choices = [output] + ([Path(borrowed)] if borrowed is not None else [])
    for root in choices:
        marker = root.with_suffix('.json')
        if not marker.exists():
            continue
        record = load(marker)
        if record.get('identity') != identity or sha(root) != record.get('sha256'):
            raise ValueError('Test feature identity/hash changed: ' + str(root))
        with np.load(root, allow_pickle=False) as saved:
            if set(saved.files) != {'features'}:
                raise ValueError('Unexpected feature archive fields.')
            array = saved['features']
        if list(array.shape) != record.get('shape'):
            raise ValueError('Test feature shape changed.')
        break
    else:
        array = encode()
    if (array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != len(identity['encoded_ends'])
            or array.shape[1] != 2048 or array.shape[2] < 3 or not np.isfinite(array).all()):
        raise ValueError('Invalid temporal/static FP32 test features.')
    if export and not output.with_suffix('.json').exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < array.nbytes + (2 << 30):
            raise RuntimeError('Need space for test features plus2GiB reserve; preserve output, no silent cache disable.')
        tmp = output.with_name(output.name + '.tmp')
        with tmp.open('wb') as stream:
            np.savez(stream, features=array)
            stream.flush(); os.fsync(stream.fileno())
        tmp.replace(output)
        save(output.with_suffix('.json'), dict(identity=identity, shape=list(array.shape), sha256=sha(output)))
    return array

def operating_point(curves, labels, validation_threshold):
    # This threshold is frozen on selected validation, never fit on test negatives.
    if len({c['id'] for c in curves}) != len(curves) or {c['id'] for c in curves} != set(labels):
        raise ValueError('Operating-point curve IDs must match labels exactly.')
    negative, alarms, positive, detected = 0, 0, 0, 0
    for c in curves:
        p = c['scores']
        if not p or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p):
            raise ValueError('Invalid probability curve.')
        if labels[c['id']] == 0:
            negative += len(p)
            alarms += sum(v >= validation_threshold for v in p)
        elif labels[c['id']] == 1:
            positive += 1
            detected += any(v >= validation_threshold for v in p)
        else:
            raise ValueError('Binary labels required.')
    if not negative or not positive:
        raise ValueError('Both classes required.')
    return dict(threshold=validation_threshold, threshold_source='selected validation threshold@0.10',
                test_negative_frame_FAR=alarms / negative, test_positive_clip_any_alarm_rate=detected / positive,
                note='clip alarm rate; not pre-event TTA or calibrated test FAR target')
