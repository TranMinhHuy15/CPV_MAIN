"""Named plain-text progress plus durable structured logs; no progress boxes."""
import datetime
import json
from pathlib import Path
import subprocess
import os
import signal
import threading
import time


class Log:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def event(self,phase,**values):
        record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**values)
        print(' | '.join('%s=%s'%(k,v) for k,v in record.items()),flush=True)
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def command(args,label,log,env=None):
    """Drain pipes continuously, show semantic lines, retain full raw traceback."""
    path=log.root/(label+'_'+str(time.time_ns())+'.log')
    log.event('START',job=label,raw_log=str(path))
    process=subprocess.Popen([str(x) for x in args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace',bufsize=1,env=env,start_new_session=os.name=='posix')
    tail=[];failure=[]
    def consume():
        try:
            with path.open('w',encoding='utf-8') as f:
                for line in process.stdout:
                    f.write(line);f.flush();tail.append(line);del tail[:-100]
                    clean=line.strip()
                    if clean.startswith(('TRAIN ','VALIDATION ','EPOCH ','TRIAL ','TEACHER ','EXTRACT ','CACHE COMPLETE',
                                         'TEST ','RESULT ','FREEZE ','INPUT ','PROFILE ','TIMING ','DONE ','ETA ','COMPACT ')):
                        print('job='+label+' | '+clean,flush=True)
        except BaseException as error:
            failure.append(error)
            # A failed log write must not leave a GPU worker blocked on its stdout pipe.
            if process.poll() is None:
                try:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                    else:process.terminate()
                except ProcessLookupError:pass
    reader=threading.Thread(target=consume,daemon=True);reader.start();start=time.monotonic()
    try:
        while process.poll() is None:
            if failure:raise failure[0]
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:log.event('RUNNING',job=label,elapsed_s=round(time.monotonic()-start,1))
    except BaseException:
        if process.poll() is None:
            try:
                if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
            except ProcessLookupError:pass
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
            process.wait()
        reader.join(timeout=5)
        process.stdout.close()
        raise
    reader.join()
    process.stdout.close()
    if failure:raise failure[0]
    if process.returncode:
        print(''.join(tail),end='',flush=True)
        log.event('FAILED',job=label,exit=process.returncode,raw_log=str(path))
        raise subprocess.CalledProcessError(process.returncode,args)
    log.event('DONE',job=label,elapsed_s=round(time.monotonic()-start,1))


def register_hook():
    import torch
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    from protocol import load,save
    if HOOKS.get('VastNamedLogHook') is not None:return
    @HOOKS.register_module()
    class VastNamedLogHook(Hook):
        priority=100  # Registered after the original state/timing hooks at this same priority.
        def __init__(self,seed,arm,candidate,every):
            self.seed,self.arm,self.candidate,self.every=seed,arm,candidate,every

        def before_train(self,runner):
            path=Path(runner.work_dir)/'named_steps.jsonl'
            if not path.exists():return
            lines=path.read_text(encoding='utf-8').splitlines();rows=[]
            for i,line in enumerate(lines):
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i!=len(lines)-1:raise
            kept=[r for r in rows if r['epoch']<=runner.epoch]
            if len(kept)!=len(lines):
                import shutil
                shutil.copyfile(path,path.with_name('named_steps_uncommitted_%d.jsonl'%time.time_ns()))
                path.write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')

        def before_train_epoch(self,runner):
            self.started=time.perf_counter();self.count=0;self.sums={}

        def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
            if not isinstance(outputs,dict) or not all(k in outputs for k in ('loss','loss_cls','loss_ffr','loss_mono')):
                raise ValueError('Original BCE/FFR/AMC loss outputs missing.')
            import math
            row={k:float(v) for k,v in outputs.items() if 'loss' in k}
            if not all(math.isfinite(v) for v in row.values()):raise FloatingPointError('Nonfinite loss.')
            self.count+=1
            for k,v in row.items():self.sums[k]=self.sums.get(k,0.)+v
            lr=max(g['lr'] for g in runner.optim_wrapper.optimizer.param_groups)
            step=dict(epoch=runner.epoch+1,batch=batch_idx+1,seed=self.seed,arm=self.arm,candidate=self.candidate,
                      lr=lr,elapsed_s=time.perf_counter()-self.started,**row)
            with (Path(runner.work_dir)/'named_steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(step)+'\n')
            if (batch_idx+1)%self.every==0 or batch_idx+1==len(runner.train_dataloader):
                print('TRAIN module=%s seed=%d config=%s epoch=%d/50 batch=%d/%d loss=%.6f BCE=%.6f FFR=%.6f AMC=%.6f lr=%.6g batches_per_s=%.3f gpu_GiB=%.3f'%
                      (self.arm,self.seed,self.candidate,runner.epoch+1,batch_idx+1,len(runner.train_dataloader),
                       row['loss'],row['loss_cls'],row['loss_ffr'],row['loss_mono'],lr,self.count/(time.perf_counter()-self.started),torch.cuda.memory_allocated()/2**30),flush=True)

        def after_val_epoch(self,runner,metrics=None):
            if not getattr(self,'count',0):return
            from protocol import history_selection
            status=load(Path(runner.work_dir)/'training_status.json')
            history=[json.loads(s) for s in (Path(runner.work_dir)/'validation_history.jsonl').read_text().splitlines() if s.strip()]
            if status['completed_epochs']!=runner.epoch or history[-1]['epoch']!=runner.epoch:
                raise ValueError('Named log must follow committed epoch.')
            elapsed=time.perf_counter()-self.started;m=history[-1]['metrics']
            row=dict(epoch=runner.epoch,seconds=elapsed,seed=self.seed,arm=self.arm,candidate=self.candidate,
                     losses={k:v/self.count for k,v in self.sums.items()},validation=m)
            path=Path(runner.work_dir)/'named_epochs.json'
            rows=[r for r in load(path) if r['epoch']<runner.epoch] if path.exists() else []
            rows.append(row);save(path,rows)
            mean=sum(r['seconds'] for r in rows[-3:])/len(rows[-3:])
            print('EPOCH module=%s seed=%d config=%s epoch=%d/50 val_mAP=%.4f%% val_mAUC=%.4f%% AP_0.5s=%.4f%% AP_1.0s=%.4f%% AP_1.5s=%.4f%% selected=%s epoch_s=%.2f remaining_h=%.3f'%
                  (self.arm,self.seed,self.candidate,runner.epoch,m['mAP@']*100,m['mAUC@']*100,m['AP@0.5s']*100,m['AP@1.0s']*100,m['AP@1.5s']*100,status['best_checkpoint'],elapsed,mean*(50-runner.epoch)/3600),flush=True)
            self.count=0
