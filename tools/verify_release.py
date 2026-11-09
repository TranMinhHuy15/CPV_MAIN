from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parents[1]
manifest=R/'checksums/SHA256SUMS_repo.txt';bad=[];listed=set()
for line in manifest.read_text().splitlines():
    if not line or line.startswith('#'):continue
    expected,_,rel=line.partition('  ');p=R/rel;listed.add(rel)
    if not p.is_file():bad.append({'file':rel,'error':'missing'});continue
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append({'file':rel,'error':'checksum'})
files=[p for p in R.rglob('*') if p.is_file() and '.git' not in p.parts]
for p in files:
    rel=p.relative_to(R).as_posix()
    if p.stat().st_size>50*1024**2:bad.append({'file':rel,'error':'file>50MiB'})
    if p.suffix.lower() in {'.pth','.pt','.ckpt','.mp4','.avi','.npy'}:bad.append({'file':rel,'error':'excluded artifact'})
    if p.suffix=='.npz' and not (p.name=='bootstrap_draws.npz' and p.parent.name=='summary'):bad.append({'file':rel,'error':'unexpected NPZ'})
size=sum(p.stat().st_size for p in files)
if size>=300000000:bad.append({'error':'repository>=300MB','bytes':size})
print(json.dumps({'verified_files':len(listed),'files':len(files),'bytes':size,'errors':bad},indent=2))
sys.exit(1 if bad else 0)
