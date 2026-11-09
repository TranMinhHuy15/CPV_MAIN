"""Recovery overlay for the pinned upstream runner, under the worker lock."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

from protocol import ARMS, CANDIDATES, load, save, sha

GIB = 1 << 30


def atomic_bytes(path, data):
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recover_training(run, expected, upstream_recover, stop_after=None):
    run = Path(run)
    if run.is_symlink():
        raise ValueError('Refusing recovery through a linked run directory.')
    manifest = run / 'run_manifest.json'
    status_path = run / 'training_status.json'
    snapshot = run / 'safe_resume_snapshot.zip'
    if not snapshot.exists() and status_path.exists():
        # The upstream writes status before publishing the first snapshot.
        if not manifest.is_file() or load(manifest).get('contract') != expected:
            raise ValueError('First-epoch recovery requires the original run contract.')
        status = load(status_path)
        if (type(status.get('completed_epochs')) is not int or status['completed_epochs'] != 1
                or status.get('target_epochs') != 50 or status.get('complete') is not False
                or status.get('best_checkpoint') != 'best_mAUC@_epoch_1.pth'
                or any(p.name != 'rng_epoch1_rank0.pth' for p in run.glob('rng_epoch*_rank*.pth'))):
            raise ValueError('Missing snapshot outside the first-epoch commit window. Preserve run and restore a backup.')
        destination = run.with_name('run_uncommitted_epoch1_' + uuid.uuid4().hex)
        run.rename(destination)
        print('RECOVERY archived incomplete first epoch at ' + str(destination), flush=True)
    upstream_recover(run, expected)
    if stop_after is not None and status_path.exists():
        committed = load(status_path)['completed_epochs']
        if committed > 3:
            raise ValueError('Pilot cannot use a trial already past epoch 3.')
        if committed >= stop_after:
            print('PLANNED STOP already verified at committed epoch %d.' % committed, flush=True)
            raise SystemExit(75)


@contextmanager
def training_recovery(run_module, stop_after):
    original = run_module.recover

    def recover(run, expected):
        recover_training(run, expected, original, stop_after)

    run_module.recover = recover
    try:
        yield
    finally:
        run_module.recover = original


def require_free_space(path, needed, phase):
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(
            '%s needs %.2f GiB free at %s (available %.2f GiB). Expand the volume and rerun.'
            % (phase, needed / GIB, path, free / GIB)
        )


def check_training_space(out, baselines, compact):
    remaining = []
    for base in next(iter(baselines.values())).values():
        for arm in ARMS:
            for candidate in CANDIDATES:
                trial = out / 'trials' / candidate['id'] / arm / ('seed_%d' % base['seed'])
                if not (trial / 'trial_complete.json').is_file():
                    remaining.append(trial)
    if not remaining:
        return
    # Latest, best, periodic checkpoint and snapshot copies, plus optimizer/adapter and logs.
    # Use the measured pilot footprint as a second estimate on retries.
    model_bytes = max(Path(row['checkpoint']).stat().st_size for row in next(iter(baselines.values())).values())
    per_trial = 5 * model_bytes + (GIB // 2)
    occupied = [
        sum(p.stat().st_size for p in (trial / 'run').glob('*') if p.is_file())
        for trial in remaining
    ]
    per_trial = max([per_trial] + occupied)
    extra = per_trial - min(occupied) if compact else sum(per_trial - used for used in occupied)
    require_free_space(out, 4 * GIB + extra, 'Estimated remaining training storage')


def cache_route(cfg, seed):
    return Path(cfg['output_root']) / 'caches' / ('route_seed%d.json' % seed)


def cache_for(cfg, seed):
    marker = cache_route(cfg, seed)
    if not marker.is_file():
        raise FileNotFoundError('Run extract first to verify or build the cache for seed %d.' % seed)
    row = load(marker)
    cache = Path(row['path']).resolve()
    allowed = {(marker.parent / ('seed_%d' % seed)).resolve()}
    borrowed = cfg['cache_roots'][str(seed)]
    if borrowed is not None:
        allowed.add(Path(borrowed).resolve())
    if (cache not in allowed or row.get('seed') != seed
            or sha(cache / 'cache_manifest.json') != row['manifest_sha256']):
        raise ValueError('Cache route/manifest changed after verification for seed %d.' % seed)
    return cache


def prepare_cache(cfg, seed, verify, extract):
    marker = cache_route(cfg, seed)
    if marker.exists():
        cache = cache_for(cfg, seed)
        verify(cache)
        print('CACHE COMPLETE verified bound cache for seed%d.' % seed, flush=True)
        return
    borrowed = cfg['cache_roots'][str(seed)]
    if borrowed is not None:
        try:
            cache = Path(borrowed).resolve()
            _, digest = verify(cache)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile) as error:
            print('CACHE REBUILD seed%d rejected borrowed cache: %s' % (seed, error), flush=True)
        else:
            save(marker, dict(seed=seed, path=str(cache), manifest_sha256=digest))
            print('CACHE COMPLETE reused seed%d original cache.' % seed, flush=True)
            return
    cache = (marker.parent / ('seed_%d' % seed)).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / 'cache_manifest.json').is_file():
        require_free_space(cache, (4 if (cache / 'cache_header.json').is_file() else 6) * GIB,
                           'Fresh feature extraction')
        extract(cache)
    _, digest = verify(cache)
    save(marker, dict(seed=seed, path=str(cache), manifest_sha256=digest))



def stage_file(source, target):
    """Publish an exact copy and repair a recognizable truncated legacy copy."""
    data = Path(source).read_bytes()
    if target.is_symlink():
        raise ValueError('Refusing to stage over a link: ' + str(target))
    if target.exists():
        existing = target.read_bytes()
        if existing == data:
            return
        if len(existing) >= len(data) or not data.startswith(existing):
            raise ValueError('Existing staged artifact changed: ' + str(target))
        target.rename(target.with_name(target.name + '.interrupted_' + uuid.uuid4().hex))
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(target, data)


@contextmanager
def atomic_cache_support(driver):
    original = driver.support_export

    def export(shared, cache):
        with tempfile.TemporaryDirectory(dir=cache, prefix='.support_') as directory:
            stage = Path(directory)
            inventory = original(shared, stage)
            for relative in inventory:
                stage_file(stage / relative, Path(cache) / relative)
        return inventory

    driver.support_export = export
    try:
        yield
    finally:
        driver.support_export = original
