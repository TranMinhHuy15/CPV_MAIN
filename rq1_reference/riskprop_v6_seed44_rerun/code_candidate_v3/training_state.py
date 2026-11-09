"""Validate committed training artifacts without loading arbitrary Torch pickles."""
import json
import math
import os
from pathlib import Path
import time
import tempfile
import uuid
import zipfile
import zlib

_verified={}


class UncommittedRun(ValueError):
    """A contract exists but no epoch was committed; preserve it before restarting."""


def basename(value):
    if (not isinstance(value,str) or not value or value in ('.','..') or ':' in value
            or '/' in value or '\\' in value or Path(value).name!=value):
        raise ValueError('Unsafe checkpoint basename')
    return value


def atomic_replace(source,target):
    """Keep replacement atomic; briefly retry Windows sharing/AV locks only."""
    for attempt in range(5):
        try:
            os.replace(source,target)
            return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==4:
                raise
            time.sleep(.05 * (2**attempt))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=str(path.parent),
                                         prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        atomic_replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def assert_committed(run,status,require_complete=False):
    run=Path(run)
    if not isinstance(status,dict):raise ValueError('Training status missing')
    epoch=status.get('completed_epochs')
    if (type(epoch) is not int or not 1<=epoch<=50 or status.get('target_epochs')!=50
            or type(status.get('complete')) is not bool or status['complete']!=(epoch==50)):
        raise ValueError('Invalid committed epoch/status')
    if require_complete and epoch!=50:raise ValueError('All selected seeds must have 50 committed epochs before evaluation')
    best=basename(status.get('best_checkpoint'))
    required=('run_manifest.json','training_status.json','validation_history.jsonl',
              'resume_latest.pth',best,'rng_epoch%d_rank0.pth'%epoch,'safe_resume_snapshot.zip')
    missing=[n for n in required if not (run/n).is_file() or (run/n).stat().st_size==0]
    if missing:raise ValueError('Committed training artifacts missing: %s'%missing)
    history=[json.loads(line) for line in (run/'validation_history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row.get('epoch') for row in history]!=list(range(1,epoch+1)):
        raise ValueError('Validation history must contain every committed epoch exactly once')
    for row in history:
        value=row.get('metrics',{}).get('mAUC@')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Missing/non-finite validation selection metric')
    signature=tuple((name,(run/name).stat().st_size,(run/name).stat().st_mtime_ns,
                     (run/name).stat().st_ctime_ns) for name in required)
    if _verified.get(str(run.resolve()))==signature:
        return True
    with zipfile.ZipFile(run/'safe_resume_snapshot.zip') as archive:
        expected=set(required)-{'safe_resume_snapshot.zip'}
        if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):
            raise ValueError('Unexpected committed snapshot inventory')
        if archive.testzip() is not None:raise ValueError('Committed snapshot CRC mismatch')
        snapshot=json.loads(archive.read('training_status.json'))
        if snapshot!=status:raise ValueError('Snapshot/status mismatch; restore the committed snapshot before checking completion')
        contract=json.loads(archive.read('run_manifest.json'))
        if contract!=json.loads((run/'run_manifest.json').read_text(encoding='utf-8')):
            raise ValueError('Snapshot/manifest mismatch')
        for name in expected:
            crc=0
            with (run/name).open('rb') as stream:
                for block in iter(lambda:stream.read(4<<20),b''):crc=zlib.crc32(block,crc)
            info=archive.getinfo(name)
            if info.file_size!=(run/name).stat().st_size or info.CRC!=(crc&0xffffffff):
                raise ValueError('Current artifact differs from committed snapshot: '+name)
    after=tuple((name,(run/name).stat().st_size,(run/name).stat().st_mtime_ns,
                 (run/name).stat().st_ctime_ns) for name in required)
    if after!=signature:raise ValueError('Training artifacts changed during integrity validation')
    _verified[str(run.resolve())]=signature
    return True


def archive_uncommitted(run):
    run=Path(run)
    if run.name!='run' or run.is_symlink() or not run.is_dir():
        raise ValueError('Expected a regular run directory')
    if (run/'training_status.json').exists() or (run/'safe_resume_snapshot.zip').exists():
        raise ValueError('A possible committed epoch exists; automatic fresh restart refused')
    rng=list(run.glob('rng_epoch*_rank*.pth'))
    if any(p.name!='rng_epoch1_rank0.pth' for p in rng):
        raise ValueError('Later-epoch RNG exists; automatic restart refused')
    history=run/'validation_history.jsonl'
    if history.exists():
        rows=[json.loads(line) for line in history.read_text(encoding='utf-8').splitlines() if line.strip()]
        if any(row.get('epoch')!=1 for row in rows):
            raise ValueError('Later-epoch history exists; automatic restart refused')
    if (run/'resume_latest.pth').exists() and not rng:
        raise ValueError('Unidentified recovery checkpoint; automatic restart refused')
    if not (run/'run_manifest.json').is_file():raise ValueError('Uncommitted contract missing')
    destination=run.with_name('run_uncommitted_%s_%s'%(time.strftime('%Y%m%d_%H%M%S'),uuid.uuid4().hex[:8]))
    if destination.parent.resolve()!=run.parent.resolve():raise ValueError('Recovery target outside workspace')
    os.replace(run,destination)
    atomic_json(run.parent/'uncommitted_recovery.json',dict(preserved_directory=str(destination),
                 reason='No committed epoch; restart from the same pretrained initialization and seed'))
    return destination
