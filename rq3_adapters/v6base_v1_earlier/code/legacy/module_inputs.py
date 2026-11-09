"""Read exact manually supplied cache, H7 export and source locations."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

SOURCE_ZIP_SHA256='464a7635fe5e5c3d837078daab8cd42f4d657896f5602f80822c74f19a2a4220'
SOURCE_MANIFEST_SHA256='2904278392d47b885e9ffc4d3bbcfd60fa4e3ec6e3a5083bdaa37fd7a4e97f2f'
PACKAGE='RQ3_H7_ThreeModules_T4_v1'
# The outer manifest deliberately omits nested manifests/verification records.
# Pins below come from the complete authenticated SOURCE_ZIP_SHA256 archive.
ANCILLARY_SHA256={
    'base_h7/PACKAGE_SHA256.json':'bfd8dac640f3049feea97598d0bafaa8a58c3bf4ffc8df17f556e86e40c9a4ed',
    'base_h7/rq1_evaluation/engine/PACKAGE_SHA256.json':'04b054758c2bbf3617254911fcaeb44c2bc14f56f53205155e62d27141d25d0d',
    'base_h7/rq1_evaluation/PACKAGE_SHA256.json':'c69a8085dbda55d9789f3fa1d34330530e5c39d36ab85a41ba20bdde1746c0f5',
    'base_h7/VERIFICATION.json':'8aad78117ca4c6dfb3f3d4bd386c1045dc2c7104217483dd0751895c4d22ca7d',
    'base_h7/verification_history/v1/VERIFICATION.json':'cb93440cfb7bef93df4e9fa7277db4b7d1c05f15390885de33b6e3fd746048a9',
    'VERIFICATION.json':'09926293a8fe850ea886864e1cd4f7c2f39d4806885757836f3a27a72006a985',
    'base_h7/rq1_evaluation/engine/references/riskprop/safety_checks.json':'919ae2bc11e4e1193d288e34eb1c397acbe5945864dd32206427cda65e5789b5',
    'base_h7/rq1_evaluation/engine/references/top/safety_checks.json':'919ae2bc11e4e1193d288e34eb1c397acbe5945864dd32206427cda65e5789b5',
}
NESTED_MANIFESTS=('base_h7/PACKAGE_SHA256.json',
                  'base_h7/rq1_evaluation/PACKAGE_SHA256.json',
                  'base_h7/rq1_evaluation/engine/PACKAGE_SHA256.json')


def read_reference_record(read,name):
    """Restore an omitted historical report only if its bytes match the original pin.

    Both manifest entries pin identical bytes, embedded in authenticated
    riskprop/verification.json. The original Windows serialization used
    json.dumps(indent=2)/write_text (CRLF, no tail).
    This restores historical evidence; it does not rerun or claim new checks.
    """
    try:
        return read(name)
    except (KeyError,FileNotFoundError):
        allowed={'base_h7/rq1_evaluation/engine/references/'+role+'/safety_checks.json'
                 for role in ('riskprop','top')}
        if name not in allowed:raise
        origin='base_h7/rq1_evaluation/engine/references/riskprop/verification.json'
        record=json.loads(read(origin))['safety_checks.json']
        data=json.dumps(record,indent=2).replace('\n','\r\n').encode('utf-8')
        if hashlib.sha256(data).hexdigest()!=ANCILLARY_SHA256[name]:
            raise ValueError('Historical evaluator record cannot be restored exactly: '+name)
        return data


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def discover_manual(cache_root,h7_root,source_path):
    fields={'CACHE_INPUT_ROOT':cache_root,'H7_ROOT_OVERRIDE':h7_root,'SOURCE_OVERRIDE':source_path}
    for name,value in fields.items():
        if value is None or not str(value).strip():
            raise ValueError(name+' chưa điền: dán Copy path thủ công, không dùng tên notebook hoặc README.md.')
    base=Path(cache_root).resolve();h7=Path(h7_root).resolve();source=Path(source_path).resolve()
    if not base.is_dir():raise FileNotFoundError('CACHE_INPUT_ROOT không tồn tại: '+str(base))
    caches={s:base/('seed_%d'%s) for s in (42,43)}
    for seed,directory in caches.items():
        marker=directory/'cache_manifest.json'
        if not marker.is_file():raise FileNotFoundError('Thiếu cache_manifest.json seed%d: %s'%(seed,marker))
        m=json.loads(marker.read_text(encoding='utf-8'))
        if (m.get('complete') is not True or m.get('header',{}).get('seed')!=seed
                or m['header'].get('protocol')!='rq3-h7-fixed-view-temporal-cache-fp32-v1'
                or m.get('counts')!={'train':1200,'validation':300}):
            raise ValueError('Cache seed%d chưa complete hoặc sai seed/protocol/count: %s'%(seed,directory))
    runs={str(s):h7/'medium_h7'/('seed_%d'%s)/'run' for s in (42,43)}
    for seed,run in runs.items():
        if not (run/'selected_export.json').is_file():
            raise FileNotFoundError('Thiếu selected H7 seed'+seed+' ở '+str(run)+'. Giữ h7_selected_input_v1 trong Output CPV_1 hoặc điền H7_ROOT_OVERRIDE.')
    if not source.exists():
        raise FileNotFoundError('SOURCE_OVERRIDE không tồn tại: '+str(source))
    if source.is_dir() and not (source/'PACKAGE_SHA256.json').is_file():
        raise FileNotFoundError('SOURCE_OVERRIDE phải trỏ trực tiếp package có PACKAGE_SHA256.json hoặc source ZIP: '+str(source))
    return dict(caches=caches,h7_runs={s:str(p) for s,p in runs.items()},source=source)


def materialize_source(source,target,zip_sha=SOURCE_ZIP_SHA256,manifest_sha=SOURCE_MANIFEST_SHA256):
    source,target=Path(source).resolve(),Path(target).resolve()
    if source.is_file():
        if sha(source)!=zip_sha:raise ValueError('Source ZIP khác bản đã kiểm; không thực thi.')
        with zipfile.ZipFile(source) as z:
            raw=z.read(PACKAGE+'/PACKAGE_SHA256.json')
            if hashlib.sha256(raw).hexdigest()!=manifest_sha:raise ValueError('Source manifest mismatch.')
            manifest=json.loads(raw)
            def read(name):return z.read(PACKAGE+'/'+name)
            _copy_checked(target,manifest,raw,read)
    else:
        if sha(source/'PACKAGE_SHA256.json')!=manifest_sha:raise ValueError('Source folder manifest mismatch.')
        raw=(source/'PACKAGE_SHA256.json').read_bytes();manifest=json.loads(raw)
        def read(name):
            p=(source/name).resolve()
            if source not in p.parents:raise ValueError('Source path escapes package.')
            return p.read_bytes()
        _copy_checked(target,manifest,raw,read)
    return target


def _copy_checked(target,manifest,raw,read):
    target.mkdir(parents=True,exist_ok=True)
    complete=dict(manifest)
    for name,digest in ANCILLARY_SHA256.items():
        if name in complete and complete[name]!=digest:raise ValueError('Ancillary manifest digest conflicts: '+name)
        complete[name]=digest
    for name,digest in complete.items():
        p=(target/name).resolve()
        if target not in p.parents:raise ValueError('Manifest path escapes working source.')
        if p.exists():
            if sha(p)!=digest:raise ValueError('Existing code differs, preserved: '+name)
            continue
        data=read_reference_record(read,name)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Missing/changed package member: '+name)
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.copy.tmp');tmp.write_bytes(data);tmp.replace(p)
    marker=target/'PACKAGE_SHA256.json'
    if marker.exists() and marker.read_bytes()!=raw:raise ValueError('Existing package marker differs.')
    if not marker.exists():marker.write_bytes(raw)
    verify_materialized(target)


def verify_materialized(target):
    target=Path(target).resolve()
    marker=target/'PACKAGE_SHA256.json'
    if sha(marker)!=SOURCE_MANIFEST_SHA256:raise ValueError('Materialized outer manifest changed.')
    expected=dict(json.loads(marker.read_text()))
    expected.update(ANCILLARY_SHA256)
    for name,digest in expected.items():
        p=(target/name).resolve()
        if target not in p.parents or not p.is_file() or sha(p)!=digest:
            raise ValueError('Source incomplete/changed before runtime: '+name)
    for marker_name in NESTED_MANIFESTS:
        marker=target/marker_name
        for relative,digest in json.loads(marker.read_text(encoding='utf-8')).items():
            p=(marker.parent/relative).resolve()
            if target not in p.parents or not p.is_file() or sha(p)!=digest:
                raise ValueError('Nested evaluator/source incomplete or changed before runtime: '+str(p))
    return len(expected)+1
