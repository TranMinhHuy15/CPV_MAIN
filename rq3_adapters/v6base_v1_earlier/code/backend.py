"""Runtime wiring for the frozen-V6 study. Author model/loss/data/evaluator code remain intact."""
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def activate(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root));sys.path.insert(0,str(HERE/'legacy'))
    os.environ['RQ2_VARIANT']='medium_h7'
    import integration,study,cached_driver
    from protocol import SEEDS,PROTOCOL
    cached_driver.SEEDS=SEEDS  # Policies only; original four cache-format script bytes stay identical.
    integration.strict_fp32()
    return integration,study,cached_driver


def configure(root,trial,log_every=20,teacher=None):
    integration,study,driver=activate(root)
    driver.install_patch()
    if getattr(integration,'_vast_tuned',False):raise RuntimeError('Use one candidate per worker process.')
    from protocol import PROTOCOL,sha,candidate
    original_config,original_contract=integration.config,integration.contract
    def config(settings,source,data,phase='train'):
        cfg=original_config(settings,source,data,phase)
        from models import register_v6
        register_v6()
        current=settings.get('hyperparameters',trial)
        cfg.model.cls_head.type='VastTunedV6Head'
        cfg.model.cls_head.width=current['width'];cfg.model.cls_head.adapter_dropout=current['dropout']
        cfg.model.cls_head.reference_history=settings.get('reference_history',7)
        cfg.model.backbone.pretrained=None
        cfg.optim_wrapper.optimizer.lr=current['lr'];cfg.optim_wrapper.optimizer.weight_decay=current['weight_decay']
        if phase=='evaluate':cfg.resume=False;cfg.load_from=None
        elif teacher is not None:
            cfg.model.type='VastTeacherCachedRecognizer'
            cfg.model.data_preprocessor=dict(type='VastTeacherPreprocessor')
            for name in ('train_dataloader','val_dataloader'):
                getattr(cfg,name).dataset.type='VastTeacherDataset'
                getattr(cfg,name).dataset.teacher_root=str(teacher)
        if phase=='train':
            from console import register_hook
            register_hook()
            cfg.custom_hooks.append(dict(type='VastNamedLogHook',seed=settings['seed'],arm=settings['arm'],candidate=current['id'],every=log_every))
        return cfg
    def contract(settings):
        value=original_contract(settings)
        value.update(protocol=PROTOCOL,hyperparameters=settings.get('hyperparameters',trial),
                     base_model='RiskProp V6 frozen (no RNN)',base_history=0,selection_policy='single predeclared candidate c2_lr_double (v1.1 winner)',
                     teacher_logits_manifest_sha256=None if teacher is None else sha(Path(teacher)/'teacher_manifest.json'),
                     vast_backend_sha256=sha(Path(__file__)),vast_models_sha256=sha(HERE/'models.py'))
        value['spec']=dict(value['spec'],**{k:value['hyperparameters'][k] for k in ('lr','weight_decay','width','dropout')})
        return value
    integration.config,integration.contract=config,contract
    integration._vast_tuned=True;study.PROTOCOL=PROTOCOL
    import runtime_profile
    def cost(arm,temporal=5,batch=2):
        # V6 base: frozen fc only (cached as teacher logits during fit). Adapter MACs as in v1.1, plus DX.
        if arm not in ('m1','m1_m2','m1_dx') or type(temporal) is not int or temporal<2:raise ValueError('Unknown arm/temporal.')
        c,d,l,n=2048,512,55,30
        histories=(7,13,26) if arm=='m1_m2' else (7,)
        projection=batch*l*c*d;recurrent=batch*n*sum(histories)*8*d*d
        attention=batch*n*len(histories)*d*d+batch*n*sum(histories)*d*d+2*batch*n*sum(histories)*d
        fusion=batch*n*(3*d)*3 if len(histories)>1 else 0
        dx=2*batch*l*(3*d*d+d*d) if arm=='m1_dx' else 0
        adapter=projection+recurrent+attention+fusion+2*batch*n*d+dx
        params={'m1':3676675,'m1_m2':3681286,'m1_dx':4727300}[arm]  # counted from the module code
        return dict(arm=arm,batch=batch,temporal_positions=temporal,trainable_parameters=params,baseline_forward_GMAC=0.,
                    adapter_forward_GMAC=adapter/1e9,total_forward_GMAC=adapter/1e9,
                    scope='adapter dense MACs only; frozen V6 logits precomputed once per video; excludes CNN/backward/I-O')
    runtime_profile.analytic_cost=cost
    return integration,study,driver
