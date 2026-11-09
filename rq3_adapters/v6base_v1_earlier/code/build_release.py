"""Build a code/config/docs-only ZIP from an explicit allowlist, never user inputs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths=list(ROOT.glob('*.py'))+[ROOT/name for name in ('README.md','AI_HANDOFF.md','PREREGISTERED_ANALYSIS.md','inputs.example.json','verification.json')]
    paths+=list((ROOT/'legacy').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths=sorted(set(paths))
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
        if path.stat().st_size>1<<20:raise ValueError('Unexpected non-code payload size: '+str(path))
    manifest=dict(code_only=True,files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
                  external_inputs_embedded=False,frozen_base='RiskProp V6 (seeds 42, 43 original; 44 rerun)',
                  modules=['m1','m1_m2','m1_dx'],candidates_per_module=1,candidate='c2_lr_double',module_training_runs=9,
                  selection='single predeclared candidate; epoch per run by earliest max validation mAUC; no test selection',
                  cuda_end_to_end_executed=False)
    marker=ROOT/'release_manifest.json';marker.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    destination=ROOT.parent/(ROOT.name+'_CODE_ONLY.zip')
    partial=destination.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths+[marker]:archive.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:raise ValueError('Release ZIP CRC failed.')
        if len(archive.namelist())!=len(paths)+1:raise ValueError('Release inventory mismatch.')
        for p in paths+[marker]:
            if hashlib.sha256(archive.read(ROOT.name+'/'+p.relative_to(ROOT).as_posix())).hexdigest()!=sha(p):
                raise ValueError('Release member hash differs: '+str(p))
    partial.replace(destination)
    result=dict(archive=destination.name,bytes=destination.stat().st_size,sha256=sha(destination),files=len(paths)+1,
                crc_and_member_hashes_pass=True,inputs_or_checkpoint_files_embedded=False)
    (ROOT/'PACKAGE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
