"""Reuse explicit dataset/base/cache paths; create a NEW V3 config without touching old outputs."""
import argparse
from pathlib import Path
from protocol import load,save,validate_config,SEEDS

BASE='v6'


def make(old_path,output_root):
    old=load(old_path);base_key=BASE+'_runs'
    if base_key not in old:raise ValueError('Wrong base in the supplied V2 config: expected '+base_key)
    previous=Path(old['output_root']).resolve();output=Path(output_root).resolve()
    if output==previous or output in previous.parents or previous in output.parents:
        raise ValueError('V3 output must be separate from V2 output.')
    common=Path(__file__).resolve().parents[1]/'inputs/rq3_selected_source_v1.zip'
    source=common if common.is_file() else Path(old['source_zip']).resolve()
    caches={}
    for seed in SEEDS:
        key=str(seed);borrow=old.get('cache_roots',{}).get(key)
        path=Path(borrow).resolve() if borrow else previous/'caches'/('seed_%d'%seed)
        caches[key]=str(path) if (path/'cache_manifest.json').is_file() else None
    cfg=dict(source_zip=str(source),data_root=str(Path(old['data_root']).resolve()),output_root=str(output),
             runtime_root=str(Path(old['runtime_root']).resolve()),prior_study_root=None,
             cache_roots=caches,encode_chunk=old.get('encode_chunk',4),
             log_every_batches=old.get('log_every_batches',20),compact_completed_trials=False)
    cfg[base_key]={str(s):str(Path(old[base_key][str(s)]).resolve()) for s in SEEDS}
    validate_config(cfg)
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-config',required=True);parser.add_argument('--output-root',required=True);parser.add_argument('--write',default='inputs.json')
    args=parser.parse_args();target=Path(args.write)
    if target.exists():raise FileExistsError('Preserve existing config: '+str(target))
    cfg=make(args.v2_config,args.output_root);save(target,cfg)
    print('CONFIG '+str(target.resolve()))
    for s,value in cfg['cache_roots'].items():print('CACHE seed%s: %s'%(s,value or 'extract afresh; no valid old cache path found'))
    print('Full generated checkpoints retained. Preflight will verify base/source/cache hashes.')


if __name__=='__main__':main()
