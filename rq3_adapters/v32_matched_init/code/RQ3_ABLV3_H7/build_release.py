"""Build explicit code/config/docs/test ZIP; never collect datasets, run outputs or checkpoints."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    verification=json.loads((ROOT/'verification.json').read_text(encoding='utf-8'))
    if verification.get('cpu_tests_pass') is not True:raise ValueError('Record actual passing CPU checks before release.')
    paths=list(ROOT.glob('*.py'))+[ROOT/n for n in ('README.md','AI_HANDOFF.md','PREREGISTERED_ANALYSIS.md','inputs.example.json','verification.json')]
    paths+=list((ROOT/'legacy').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths=sorted(set(paths))
    for path in paths:
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
        if not path.is_file() or path.stat().st_size>1024*1024:raise ValueError('Unexpected release member: '+str(path))
    manifest=dict(version='V3.2 recovery',code_only=True,planned_runs=18,epochs_per_run=50,
        planned_arms=['c7_current_repeat','r7_last_state','a7_m1','a26_m1','u3_uniform','g3_learned'],
        historical_predictions_used_for_main=False,cuda_end_to_end_executed=False,
        pinned_original_source_zip_sha256='464a7635fe5e5c3d837078daab8cd42f4d657896f5602f80822c74f19a2a4220',
        files={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    marker=ROOT/'release_manifest.json';marker.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    target=ROOT.parent/(ROOT.name+'_CODE_ONLY.zip');partial=target.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths+[marker]:archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:raise ValueError('CRC failed.')
        for path in paths+[marker]:
            if hashlib.sha256(archive.read(ROOT.name+'/'+path.relative_to(ROOT).as_posix())).hexdigest()!=sha(path):raise ValueError('Member hash failed.')
    partial.replace(target)
    result=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(paths)+1,crc_and_member_hashes_pass=True)
    (ROOT/'PACKAGE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
