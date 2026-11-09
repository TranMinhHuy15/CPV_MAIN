"""Complete, verified workspace backup; raw videos and installed CUDA runtime stay external."""
import argparse,hashlib,json,os,shutil,sys,tempfile,time,uuid,zipfile
from pathlib import Path
from integrity import ROOT,LEGACY,training,digest,load
from training_state import atomic_json
sys.path.insert(0,str(LEGACY))
from process_control import exclusive_job

def signature(path):
    s=path.stat();return s.st_size,s.st_mtime_ns

def verify_backup(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names) or z.testzip() is not None:raise ValueError('Duplicate member or bad ZIP CRC')
        manifest=json.loads(z.read('BACKUP_MANIFEST.json'))
        if manifest.get('schema')!='seed44-full-workspace-backup-v1':raise ValueError('Wrong backup schema')
        if set(names)!={'BACKUP_MANIFEST.json'}|set(manifest['files']):raise ValueError('Backup inventory mismatch')
        for name,expected in manifest['files'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe member')
            h=hashlib.sha256()
            with z.open(name) as stream:
                for b in iter(lambda:stream.read(4<<20),b''):h.update(b)
            if h.hexdigest()!=expected:raise ValueError('Backup hash mismatch: '+name)
    return manifest

def backup(workspace,output,diagnostic=False):
    workspace=Path(workspace).resolve();output=Path(output).resolve()
    if not workspace.is_dir() or not (workspace/'settings.json').is_file():raise ValueError('Workspace/settings.json missing')
    if output==workspace or workspace in output.parents:raise ValueError('Backup output must be outside workspace')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive_job(workspace/'.safe_job.lock'):
        settings=load(workspace/'settings.json')
        if settings.get('seed')!=44:raise ValueError('This backup entry point is for seed44 only')
        run=workspace/'run';committed=False
        if (run/'training_status.json').exists():
            try:training(run,complete=False);committed=True
            except (ValueError,OSError,KeyError,zipfile.BadZipFile):
                if not diagnostic:raise
        elif not diagnostic:
            raise ValueError('No committed epoch: use --diagnostic to preserve a failed/preflight attempt')
        paths=[];external=[]
        for p in sorted(workspace.rglob('*')):
            relative=p.relative_to(workspace)
            if relative.parts[:2] in (('prepared_nexar','train'),('prepared_nexar','test')) and p.suffix.lower()=='.mp4':
                external.append(dict(path=relative.as_posix(),target=str(p.resolve())));continue
            if p.is_symlink():raise ValueError('Unexpected symlink; refusing to silently follow it: '+str(p))
            if p.is_file() and not p.name.endswith('.lock'):paths.append(p)
        initial={p:signature(p) for p in paths}
        required=sum(size for size,_ in initial.values())+(64<<20)
        if shutil.disk_usage(output).free<required:raise OSError('Not enough free disk for full backup')
        destination=output/('seed44_full_%s_%s.zip'%(time.strftime('%Y%m%d_%H%M%S'),uuid.uuid4().hex[:8]))
        fd,temporary_name=tempfile.mkstemp(prefix=destination.stem+'.',suffix='.tmp',dir=str(output));os.close(fd)
        temporary=Path(temporary_name)
        try:
            files={}
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for p in paths:
                    name='seed_44/'+p.relative_to(workspace).as_posix();h=hashlib.sha256()
                    with p.open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
                        for b in iter(lambda:src.read(4<<20),b''):dst.write(b);h.update(b)
                    if signature(p)!=initial[p]:raise RuntimeError('Workspace changed during backup: '+str(p))
                    files[name]=h.hexdigest()
                if any(signature(p)!=initial[p] for p in paths):raise RuntimeError('Workspace changed during backup')
                manifest=dict(schema='seed44-full-workspace-backup-v1',seed=44,committed_artifacts_verified=committed,
                              diagnostic=bool(diagnostic),files=files,external_videos=external,
                              dataset='Original 2844 Nexar MP4 files required separately; prepare recreates video links',
                              installed_runtime='Not included; recreate the pinned runtime with the unchanged V6 installer')
                z.writestr('BACKUP_MANIFEST.json',json.dumps(manifest,indent=2))
            verify_backup(temporary)
            os.replace(temporary,destination)
            sha=digest(destination)
            atomic_json(destination.with_suffix('.zip.verified.json'),dict(sha256=sha,bytes=destination.stat().st_size,
                files=len(files),committed_artifacts_verified=committed,raw_video_files_included=False))
            return destination
        finally:
            if temporary.exists():temporary.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',default=os.environ.get('WS','/workspace/riskprop_v6_seed44_rerun'))
    parser.add_argument('--output-dir',default=os.environ.get('OUT','/workspace/seed44_backups'))
    parser.add_argument('--diagnostic',action='store_true',help='Preserve a broken/uncommitted run, without claiming resumability')
    args=parser.parse_args()
    print('Verified backup:',backup(args.workspace,args.output_dir,args.diagnostic))
