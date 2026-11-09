"""Atomic complete-epoch snapshot: rollback interrupted multi-file saves safely."""
import json
from pathlib import Path
import zipfile

SNAPSHOT = 'safe_resume_snapshot.zip'


def commit_snapshot(run):
    run = Path(run)
    status = json.loads((run/'training_status.json').read_text())
    epoch, best = status['completed_epochs'], status['best_checkpoint']
    if type(epoch) is not int or not 0 < epoch <= 50 or not isinstance(best,str) or Path(best).name != best or '\\' in best:
        raise RuntimeError('Unsafe snapshot metadata.')
    names = ['resume_latest.pth',best,'rng_epoch%d_rank0.pth'%epoch,
             'training_status.json','run_manifest.json','validation_history.jsonl']
    if any(not (run/name).is_file() or not (run/name).stat().st_size for name in names):
        raise RuntimeError('Cannot snapshot an incomplete completed epoch.')
    temporary = run/(SNAPSHOT+'.tmp')
    # Stored, not compressed: avoid expensive compression of float checkpoints.
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as archive:
        for name in names:
            archive.write(run/name,name)
    temporary.replace(run/SNAPSHOT)


def restore_snapshot(run, expected_protocol, expected_seed):
    run = Path(run)
    snapshot = run/SNAPSHOT
    if not snapshot.exists():
        return False
    with zipfile.ZipFile(snapshot) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Committed recovery snapshot is corrupt; refusing rollback.')
        status = json.loads(archive.read('training_status.json'))
        contract = json.loads(archive.read('run_manifest.json'))['contract']
        if contract.get('protocol') != expected_protocol or contract.get('seed') != expected_seed:
            raise RuntimeError('Recovery snapshot belongs to a different protocol/seed.')
        epoch, best = status['completed_epochs'], status['best_checkpoint']
        expected = {'resume_latest.pth',best,'rng_epoch%d_rank0.pth'%epoch,
                    'training_status.json','run_manifest.json','validation_history.jsonl'}
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != expected or any(Path(n).name != n or '\\' in n for n in names):
            raise RuntimeError('Unsafe/unexpected snapshot members.')
        for name in names:
            target = run/name
            temporary = run/(name+'.restore.tmp')
            # Streaming avoids copying a whole checkpoint into Python RAM.
            with archive.open(name) as source, temporary.open('wb') as output:
                while True:
                    chunk = source.read(4<<20)
                    if not chunk:
                        break
                    output.write(chunk)
            temporary.replace(target)
    return True
