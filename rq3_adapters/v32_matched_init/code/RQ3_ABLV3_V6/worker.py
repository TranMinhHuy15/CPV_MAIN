"""One GPU phase per fresh process (frozen-V6 study), heavy work only via explicit CLI execution."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

from protocol import SEEDS,ARMS,CANDIDATES,load,save,sha,bind,candidate
from backend import activate,configure

DEADLINE=sys.float_info.max  # No session-time cap; original complete-epoch recovery stays active.


from recovery import atomic_cache_support, cache_for, prepare_cache, training_recovery

def inputs_for(cfg,out):
    # Legacy field name 'h7_runs' is internal plumbing only; it carries the frozen V6 run folders.
    return dict(data_root=cfg['data_root'],out_root=str(out),h7_runs=cfg['v6_runs'],arms=list(ARMS),encode_chunk=cfg['encode_chunk'])


VALIDATION_PARITY_MAX_ABS=1e-2   # V6 validation used PyTorch-default cuDNN TF32 convs; this path is strict FP32
VALIDATION_PARITY_MEAN_ABS=1e-3  # a wrong checkpoint or pipeline gives mean |dp| >> 1e-2


def validation_parity(base,ids,logits,output):
    """Gate before any training: frozen-V6 logits from our cache must reproduce V6's own validation
    snippet probabilities (300 videos x 30 snippets) saved when the V6 run was trained."""
    import numpy as np
    original={r['id']:np.asarray(r['pred'],dtype=np.float64) for r in load(Path(base['run'])/'validation_predictions.json')}
    index={v:i for i,v in enumerate(ids)}
    missing=[v for v in original if v not in index]
    if missing:raise ValueError('Validation IDs of V6 run absent from cache: %d'%len(missing))
    ours=np.stack([1/(1+np.exp(-logits[index[v]].astype(np.float64))) for v in original])
    theirs=np.stack([original[v] for v in original])
    diff=np.abs(ours-theirs)
    result=dict(videos=len(original),snippets=int(diff.size),max_abs_prob_diff=float(diff.max()),
                mean_abs_prob_diff=float(diff.mean()),tolerance=VALIDATION_PARITY_MAX_ABS,
                passed=bool(diff.max()<=VALIDATION_PARITY_MAX_ABS and diff.mean()<=VALIDATION_PARITY_MEAN_ABS),mean_tolerance=VALIDATION_PARITY_MEAN_ABS,
                scope='frozen V6 fc(mean feature) on cached validation features vs V6 validation_predictions.json')
    save(output/'v6_validation_parity.json',result)
    print('V6 PARITY seed=%d validation 300x30: max|dp|=%.2e mean|dp|=%.2e -> %s'%(base['seed'],result['max_abs_prob_diff'],
          result['mean_abs_prob_diff'],'PASS' if result['passed'] else 'FAIL'),flush=True)
    if not result['passed']:raise ValueError('Frozen V6 path does not reproduce V6 validation predictions; stop before training.')
    return result


def teacher_logits(cfg,root,base,seed):
    import numpy as np
    import torch
    from models import checked_teacher
    from cache_store import verify,read_record
    import run_modules as run
    import integration
    from cached_driver import tools_hashes
    cache=cache_for(cfg,seed);manifest,cache_sha=verify(cache,seed,base['checkpoint_sha256'],tools_hashes())
    output=Path(cfg['output_root'])/'teachers'/('seed_%d'%seed);output.mkdir(parents=True,exist_ok=True)
    identity=dict(cache_manifest_sha256=cache_sha,checkpoint_sha256=base['checkpoint_sha256'],seed=seed,
                  runtime=run.runtime(),models_sha256=sha(Path(__file__).parent/'models.py'))
    bind(output/'identity.json',identity)
    if (output/'teacher_manifest.json').exists():
        m,values=checked_teacher(output,cache_sha,base['checkpoint_sha256'])
        validation_parity(base,list(values),np.stack(list(values.values())),output);return
    inp=inputs_for(cfg,Path(cfg['output_root'])/'teacher_build')
    settings=run.settings_for(inp,{str(seed):base},cache,run.runtime(),'m1',seed,DEADLINE)
    settings['cache_mode']='extract'
    runner=run.build(settings,'evaluate');run.load_weights(runner,base['checkpoint'],baseline_only=True)
    model=runner.model.eval();rows=[];ids=[];started=time.monotonic();batch_check=[];batch_errors=[]
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
        for i,record in enumerate(manifest['records']):
            features=read_record(cache,record,manifest['header_sha256'])
            tensor=torch.from_numpy(features).to(device='cuda').squeeze(-1).squeeze(-1)[None]
            value=model.cls_head.baseline(tensor)
            if i<8:
                direct=model.cls_head.score_features(tensor).reshape(1,30)
                if not torch.equal(value,direct):raise ValueError('Frozen baseline/logit-cache alignment failed.')
                batch_check.append((tensor.detach(),value.detach()))
                if len(batch_check)==2:
                    batched=model.cls_head.baseline(torch.cat([r[0] for r in batch_check],0))
                    cached=torch.cat([r[1] for r in batch_check],0)
                    if not torch.allclose(batched,cached,atol=1e-5,rtol=1e-5):
                        raise ValueError('Frozen teacher batch1 vs training microbatch2 FP32 parity failed.')
                    batch_errors.append(float((batched-cached).abs().max()));batch_check=[]
            rows.append(value.cpu().float().numpy()[0]);ids.append(record['video_id'])
            if (i+1)%100==0:print('TEACHER seed=%d videos=%d/1500 elapsed_s=%.2f'%(seed,i+1,time.monotonic()-started),flush=True)
    array=np.asarray(rows,dtype=np.float32)
    path=output/'logits.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,logits=array)
    tmp.replace(path)
    validation_parity(base,ids,array,output)
    save(output/'teacher_manifest.json',dict(identity,complete=True,video_ids=ids,array_sha256=sha(path),
         original_frozen_path_checked=True,training_microbatch2_checked=True,
         batched_reference_max_error=max(batch_errors),batched_reference_atol=1e-5,batched_reference_rtol=1e-5,
         seconds=time.monotonic()-started))
    checked_teacher(output,cache_sha,base['checkpoint_sha256'])
    del runner,model;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--phase',choices=('extract','teacher','train','test'),required=True)
    parser.add_argument('--stop-after-epoch',type=int,choices=range(1,50))
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--candidate',choices=[c['id'] for c in CANDIDATES])
    args=parser.parse_args();cfg=load(args.config);out=Path(cfg['output_root']);prepared=load(out/'prepared.json')
    from study_guard import require_worker_inputs
    require_worker_inputs(out,Path(__file__).resolve().parent,cfg,prepared)
    import os
    os.environ.pop('RQ3_STOP_AFTER_EPOCH',None)
    if args.stop_after_epoch is not None:
        if args.phase!='train':raise ValueError('Planned stop only applies to train.')
        import os
        os.environ['RQ3_STOP_AFTER_EPOCH']=str(args.stop_after_epoch)
    root=Path(prepared['source_root'])
    from legacy.module_inputs import verify_materialized
    verify_materialized(root)
    base=prepared['baselines']['v6'][str(args.seed)]
    if sha(base['checkpoint'])!=base['checkpoint_sha256']:raise ValueError('Original V6 checkpoint changed after preflight.')
    trial=candidate(args.candidate or CANDIDATES[0]['id'])
    teacher=out/'teachers'/('seed_%d'%args.seed) if args.phase=='train' else None
    integration,study,driver=configure(root,trial,cfg['log_every_batches'],teacher)
    import run_modules as run
    runtime=run.runtime()
    bind(out/'runtime_identity.json',runtime)
    from process_control import exclusive_job
    with exclusive_job(out/'.gpu_phase.lock'):
        if args.phase=='extract':
            from cache_store import verify
            tools = driver.tools_hashes()
            seed = args.seed
            def verify_candidate(cache):
                manifest, digest = verify(cache, seed, base['checkpoint_sha256'], tools)
                header = manifest['header']
                if (header['extension_hashes'] != integration.extension_hashes()
                        or header['h7_contract_sha256'] != base['contract_sha256']):
                    raise ValueError('Cache source or baseline contract differs.')
                return manifest, digest

            with atomic_cache_support(driver):
                prepare_cache(
                    cfg, seed, verify_candidate,
                    lambda cache: driver.extract(inputs_for(cfg, out / 'extraction'), base, runtime, seed, cache, DEADLINE),
                )
        elif args.phase=='teacher':teacher_logits(cfg,root,base,args.seed)
        elif args.phase=='train':
            if args.arm is None or args.candidate is None:raise ValueError('Train needs module and candidate.')
            workspace=out/'trials'/trial['id']/args.arm/('seed_%d'%args.seed)
            workspace.mkdir(parents=True,exist_ok=True)
            bind(workspace/'owned_trial.json',dict(study_contract_sha256=sha(out/'study_contract.json'),candidate=trial['id'],arm=args.arm,seed=args.seed))
            from artifacts import verify_trial,compact
            if (workspace/'trial_complete.json').exists():verify_trial(workspace);print('TRIAL complete verified; skip',flush=True);return
            print('TRIAL module=%s seed=%d hyperparameters=%s'%(args.arm,args.seed,trial),flush=True)
            with training_recovery(run, args.stop_after_epoch):
                driver.fit(inputs_for(cfg,out/'trials'/trial['id']),base,runtime,args.seed,args.arm,cache_for(cfg,args.seed),DEADLINE,'train')
            compact(workspace,base,cfg)
        else:
            from test_backend import test_seed
            test_seed(cfg,root,prepared['baselines'],args.seed,runtime)


if __name__=='__main__':main()
