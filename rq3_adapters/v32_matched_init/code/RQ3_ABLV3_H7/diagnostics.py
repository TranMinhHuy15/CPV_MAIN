"""Non-intervening validation diagnostics; they never select epochs or modify loss."""
import json
import math
import os
from pathlib import Path
import shutil
import time

from recovery import atomic_bytes

FILES = ('adapter_diagnostics_validation.jsonl', 'adapter_samples_validation.jsonl')
SAMPLE_BATCHES = 2
SATURATION = .95


def detached(value):
    if isinstance(value, dict):return {k:detached(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):return [detached(v) for v in value]
    return value.detach() if hasattr(value, 'detach') else value


class Moments:
    def __init__(self):self.values = {}

    def add(self, name, tensor):
        if tensor is None:return
        import torch
        v = tensor.detach().double()
        if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite diagnostic ' + name)
        if not v.numel():raise ValueError('Empty diagnostic ' + name)
        old = self.values.setdefault(name, dict(n=0, total=0., squares=0., minimum=float('inf'), maximum=-float('inf'), near_bound=0))
        old['n'] += v.numel();old['total'] += float(v.sum());old['squares'] += float((v*v).sum())
        old['minimum'] = min(old['minimum'], float(v.min()));old['maximum'] = max(old['maximum'], float(v.max()))
        old['near_bound'] += int((v.abs() > SATURATION).sum())

    def update(self, diag):
        import torch
        if not all(k in diag for k in ('gate','raw','correction')):
            raise ValueError('Missing gate/raw/correction diagnostics from the integrated head.')
        for key in ('gate','raw','correction'):self.add(key, diag[key])
        self.add('tanh_raw', torch.tanh(diag['raw']))
        for i, attention in enumerate(diag.get('attention', [])):
            if attention is not None:
                entropy = -(attention.double()*attention.double().clamp_min(1e-30).log()).sum(-1)
                self.add('attention_entropy_window%d' % i, entropy)
        omega = diag.get('scale_weights')
        if omega is not None:
            for i in range(omega.shape[-1]):self.add('omega%d' % i, omega[...,i])

    def summary(self):
        if not self.values:raise ValueError('No validation diagnostic batches were observed.')
        out = {}
        for key, v in self.values.items():
            mean=v['total']/v['n']
            out[key]=dict(count=v['n'],mean=mean,std=math.sqrt(max(0.,v['squares']/v['n']-mean*mean)),
                          minimum=v['minimum'],maximum=v['maximum'],fraction_abs_above_0_95=v['near_bound']/v['n'])
        return out


def append(path, row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,allow_nan=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def trim_uncommitted(run, committed_epoch):
    for name in FILES:
        path=Path(run)/name
        if not path.exists():
            if committed_epoch:
                raise ValueError('Missing diagnostics for a committed epoch: ' + str(path))
            continue
        lines=path.read_text(encoding='utf-8').splitlines();kept=[]
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                continue
            if row['epoch']<=committed_epoch:kept.append(row)
        if name == FILES[0]:
            aligned = [row['epoch'] for row in kept] == list(range(1, committed_epoch + 1))
        else:
            aligned = [(row['epoch'], row['batch']) for row in kept] == [
                (epoch, batch)
                for epoch in range(1, committed_epoch + 1)
                for batch in range(1, SAMPLE_BATCHES + 1)
            ]
        if not aligned:
            raise ValueError('Incomplete committed diagnostics, preserve and inspect: ' + str(path))
        if len(kept)!=len(lines):
            shutil.copyfile(path,path.with_name(path.stem+'_uncommitted_%d.jsonl'%time.time_ns()))
            atomic_bytes(path, ''.join(json.dumps(r, allow_nan=False, sort_keys=True) + '\n' for r in kept).encode('utf-8'))


def capture_validation(hook, runner, batch_idx):
    model=getattr(runner.model,'module',runner.model)
    diag=model.cls_head.last_diagnostics
    hook.diag_moments.update(diag)
    if batch_idx<SAMPLE_BATCHES:
        row=dict(epoch=runner.epoch,batch=batch_idx+1,arm=hook.arm,seed=hook.seed,
                 scope='first two validation batches; token arrays, no test-based selection')
        for k in ('gate','raw','correction','scale_weights'):
            value=diag.get(k);row[k]=None if value is None else value.detach().cpu().tolist()
        append(Path(runner.work_dir)/FILES[1],row)


def finish_validation(hook, runner):
    row=dict(epoch=runner.epoch,seed=hook.seed,arm=hook.arm,values=hook.diag_moments.summary(),
             scope='all validation target tokens; correlated observations, not independent samples',
             saturation_threshold=SATURATION,affects_selection=False)
    append(Path(runner.work_dir)/FILES[0],row)
    correction=row['values']['correction']
    print('DIAGNOSTICS module=%s seed=%d epoch=%d correction_mean=%.6f correction_std=%.6f saturation=%.4f gate_mean=%.6f'%
          (hook.arm,hook.seed,runner.epoch,correction['mean'],correction['std'],correction['fraction_abs_above_0_95'],row['values']['gate']['mean']),flush=True)


def verify_validation(run):
    rows=[json.loads(s) for s in (Path(run)/FILES[0]).read_text(encoding='utf-8').splitlines() if s.strip()]
    if [r['epoch'] for r in rows]!=list(range(1,51)):
        raise ValueError('Need exactly 50 committed validation diagnostic epochs.')
    for row in rows:
        if row.get('affects_selection') is not False or not all(k in row['values'] for k in ('gate','raw','correction')):
            raise ValueError('Wrong validation diagnostic schema.')
    samples=[json.loads(s) for s in (Path(run)/FILES[1]).read_text(encoding='utf-8').splitlines() if s.strip()]
    if [(r['epoch'],r['batch']) for r in samples]!=[(e,b) for e in range(1,51) for b in range(1,SAMPLE_BATCHES+1)]:
        raise ValueError('Incomplete/duplicate diagnostic validation samples.')
    return rows
