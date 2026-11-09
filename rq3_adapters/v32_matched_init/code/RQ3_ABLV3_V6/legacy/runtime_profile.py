"""Measured GPU timings and transparent MAC estimates; never convert T4 peak TFLOPS to ETA."""
import csv
import gc
from pathlib import Path
import time

from cache_store import ARMS, load, save


def analytic_cost(arm, temporal=5, batch=2):
    if arm not in ARMS or type(temporal) is not int or temporal < 2:
        raise ValueError('Require actual temporal positions>=2 and known arm.')
    c,d,l,n = 2048,512,55,30
    histories = (7,13,26) if arm in ('m1_m2','full') else (7,)
    baseline = batch*n*7*4*(c*c+c*c) + batch*n*c
    projection = batch*l*c*d
    recurrent = batch*n*sum(histories)*8*d*d
    attention = batch*n*len(histories)*d*d + batch*n*sum(histories)*d*d
    attention += 2*batch*n*sum(histories)*d
    fusion = batch*n*(3*d)*3 if len(histories)>1 else 0
    output_gate = 2*batch*n*d
    motion = 0
    if arm in ('m1_m3','full'):
        positions = batch*l*(temporal-1)
        motion = positions*c*d + 2*positions*(2*d*d)  # phi(diff) AND phi(0), two linear layers each.
    adapter = projection+recurrent+attention+fusion+output_gate+motion
    parameters = 3676675 + (4611 if arm in ('m1_m2','full') else 0) + (526337 if arm in ('m1_m3','full') else 0)
    return dict(arm=arm, batch=batch, temporal_positions=temporal,
        trainable_parameters=parameters,
        baseline_forward_GMAC=baseline/1e9, adapter_forward_GMAC=adapter/1e9,
        total_forward_GMAC=(baseline+adapter)/1e9,
        scope='dense linear/LSTM MACs only; excludes CNN, backward, normalization, nonlinear ops, I/O and launch cost')


def benchmark(settings, smoke_receipt):
    import torch
    import run_modules as run
    from integration import contract, strict_fp32
    if smoke_receipt.get('passed') is not True or smoke_receipt.get('contract') != contract(settings):
        raise ValueError('Timing requires matching passed real-data smoke.')
    runner = run.build(settings,'smoke')
    run.load_weights(runner,settings['h7_checkpoint'])
    runner.model.eval()
    iterator=iter(runner.val_dataloader)
    samples=[]
    for i in range(12):
        if time.time()+60 >= settings['deadline']:
            raise TimeoutError('Budget reached during validation timing; output preserved.')
        torch.cuda.synchronize()
        started=time.perf_counter()
        batch=next(iterator)
        strict_fp32()
        with torch.no_grad():
            predictions=runner.model.val_step(batch)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        if i>=2:
            samples.append(elapsed)
        del batch,predictions
    train_s=smoke_receipt['seconds']/24
    val_s=sum(samples)/len(samples)
    root=Path(settings['cache_root'])
    manifest=load(root/'cache_manifest.json')
    temporal=manifest['records'][0]['shape'][2]-1
    row=dict(arm=settings['arm'],seed=settings['seed'],contract=contract(settings),
        train_microbatch2_seconds_from_smoke=train_s,
        validation_batch1_seconds_including_cache_read=val_s,
        validation_timing_samples=samples,
        train_microbatches_per_epoch=600,validation_batches_per_epoch=300,
        estimated_train_seconds_per_epoch=600*train_s,
        estimated_validation_seconds_per_epoch=300*val_s,
        estimated_compute_io_seconds_per_epoch=600*train_s+300*val_s,
        estimated_compute_io_hours_50epochs=(600*train_s+300*val_s)*50/3600,
        train_peak_allocated_GiB=smoke_receipt['peak_memory_bytes']/2**30,
        analytic=analytic_cost(settings['arm'],temporal),
        caveat='pilot projection, not guarantee; smoke includes audit overhead; excludes full metric computation, checkpoint writes, extraction, setup; epoch_timing.json records complete measured epochs')
    save(Path(settings['workspace'])/'timing_profile.json',row)
    print('TIMING PILOT %s seed%d | train batch2 %.4fs | val batch1 %.4fs | epoch compute+IO %.1fs | 50epoch %.2fh + metrics/checkpoints/setup | peak %.2fGiB' %
        (settings['arm'],settings['seed'],train_s,val_s,row['estimated_compute_io_seconds_per_epoch'],
         row['estimated_compute_io_hours_50epochs'],row['train_peak_allocated_GiB']),flush=True)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return row


def summarize_profiles(out, seed):
    rows=[]
    for arm in ARMS:
        path=Path(out)/arm/('seed_%d'%seed)/'timing_profile.json'
        if path.exists():
            p=load(path)
            rows.append({k:p[k] for k in ('arm','seed','train_microbatch2_seconds_from_smoke',
                'validation_batch1_seconds_including_cache_read','estimated_train_seconds_per_epoch',
                'estimated_validation_seconds_per_epoch','estimated_compute_io_seconds_per_epoch',
                'estimated_compute_io_hours_50epochs','train_peak_allocated_GiB')})
    path=Path(out)/('timing_by_arm_seed%d.csv'%seed)
    if rows:
        temporary=path.with_suffix('.csv.tmp')
        with temporary.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(path)
    return path


def register_hook():
    import torch
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    if HOOKS.get('CacheEpochTimingHook') is not None:
        return

    @HOOKS.register_module()
    class CacheEpochTimingHook(Hook):
        # ReproductionStateHook priority90 commits snapshot first; time that cost too.
        priority=100
        def before_train_epoch(self,runner):
            torch.cuda.synchronize()
            self.start=time.perf_counter()
            self.val_start=None

        def before_val(self,runner):
            torch.cuda.synchronize()
            self.val_start=time.perf_counter()

        def after_val_epoch(self,runner,metrics=None):
            if getattr(self,'start',None) is None or self.val_start is None:
                return
            torch.cuda.synchronize()
            end=time.perf_counter()
            path=Path(runner.work_dir)/'epoch_timing.json'
            epoch=runner.epoch
            rows=load(path) if path.exists() else []
            rows=[r for r in rows if r['epoch']<epoch]
            row=dict(epoch=epoch,train_and_regular_save_seconds=self.val_start-self.start,
                     validation_metrics_best_and_snapshot_seconds=end-self.val_start,
                     complete_epoch_seconds=end-self.start)
            rows.append(row);save(path,rows)
            recent=rows[-3:]
            mean=sum(r['complete_epoch_seconds'] for r in recent)/len(recent)
            print('EPOCH TIME %d/50 | train+regular-save %.1fs | val+metric+best+snapshot %.1fs | total %.1fs | remaining %d epochs ~%.2fh (rolling%d; excludes setup/extraction)' %
                (epoch,row['train_and_regular_save_seconds'],row['validation_metrics_best_and_snapshot_seconds'],
                 row['complete_epoch_seconds'],50-epoch,mean*(50-epoch)/3600,len(recent)),flush=True)
