"""Check shipped code/reference hashes; generated runtime files are not part of this manifest."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'FILE_SHA256.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        p=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file():raise ValueError('Missing/unsafe package file: '+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Package file changed: '+name)
    return len(manifest)
if __name__=='__main__':print('Package hashes PASS:',verify())
