"""Package source/metadata only; exclude caches, weights and runtime artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    destination=Path(args.zip).resolve()
    if destination.exists():
        raise FileExistsError('Refuse replacing an existing release ZIP.')
    if destination==root or root in destination.parents:
        raise ValueError('Archive must be outside source directory.')
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix not in ('.pyc','.pth','.npz') and p.name not in
        ('PACKAGE_SHA256.json','runtime_bootstrap.json','runtime_pip_freeze.txt','pip_check.txt'))
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=root/'PACKAGE_SHA256.json'
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files+[path]:
            archive.write(p,root.name+'/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failure.')
        for relative,expected in manifest.items():
            if hashlib.sha256(archive.read(root.name+'/'+relative)).hexdigest()!=expected:
                raise RuntimeError('Packaged byte mismatch: '+relative)
    print(json.dumps(dict(zip=str(destination),bytes=destination.stat().st_size,files=len(manifest),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),crc_and_manifest_verified=True),indent=2))


if __name__=='__main__': main()
