"""Runtime wiring for ablation V2 on the frozen H7 (C7, R7, A26, U3). Author model/loss/data/evaluator code remain intact."""
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
        from models import register_ablation
        register_ablation()
        current=settings.get('hyperparameters',trial)
        cfg.model.cls_head.type='VastTunedH7AblationHead'
        cfg.model.cls_head.width=current['width'];cfg.model.cls_head.adapter_dropout=current['dropout']
        cfg.model.cls_head.reference_history=settings.get('reference_history',7)
        cfg.model.cls_head.adapter_init_seed=settings['seed']
        cfg.randomness=dict(getattr(cfg,'randomness',{}) or {},seed=settings['seed'])
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
            cfg.custom_hooks.append(dict(type='MatchedValidationDiagnosticsHook',seed=settings['seed'],arm=settings['arm']))
            cfg.custom_hooks.append(dict(type='VastNamedLogHook',seed=settings['seed'],arm=settings['arm'],candidate=current['id'],every=log_every))
        return cfg
    def contract(settings):
        value=original_contract(settings)
        value.update(protocol=PROTOCOL,hyperparameters=settings.get('hyperparameters',trial),
                     base_model='H7 frozen (bounded LSTM2048, 7 snippets)',base_history=7,selection_policy='single predeclared candidate c2_lr_double (matched to v1.1 M1/M1+M2)',
                     teacher_logits_manifest_sha256=None if teacher is None else sha(Path(teacher)/'teacher_manifest.json'),
                     vast_backend_sha256=sha(Path(__file__)),vast_models_sha256=sha(HERE/'models.py'))
        from matched_initialization import POLICY
        value['initialization']=dict(policy=POLICY,adapter_init_seed=settings['seed'],training_seed=settings['seed'],shared_state_across_six_arms=True)
        value['spec']=dict(value['spec'],**{k:value['hyperparameters'][k] for k in ('lr','weight_decay','width','dropout')})
        from ablation_modules import HISTORIES,PAST_SNIPPETS,TRAINABLE,BASE_PARAMETERS
        arm=settings['arm']
        if arm in HISTORIES:
            # Real arm identity (the source contract only knows m1/m1_m2): windows, past snippets, parameters, lookback.
            value['ablation_arm']=dict(id=arm,histories=list(HISTORIES[arm]),repeat_current=arm=='c7_current_repeat',
                readout='last_state' if arm=='r7_last_state' else 'attention_age',
                fusion='uniform_mean' if arm=='u3_uniform' else 'softmax_learned' if arm=='g3_learned' else None,
                past_snippets=PAST_SNIPPETS[arm],trainable_parameters=TRAINABLE[arm],
                deployed_parameters=BASE_PARAMETERS['h7']+TRAINABLE[arm],adapter_code_sha256=sha(HERE/'ablation_modules.py'))
            value['maximum_native_frame_lookback']=12+3*(max(7,PAST_SNIPPETS[arm]+1)-1)
        return value
    integration.config,integration.contract=config,contract
    integration._vast_tuned=True;study.PROTOCOL=PROTOCOL
    import runtime_profile
    from protocol import ARMS
    runtime_profile.ARMS=ARMS  # collect timing profiles for all six modes without changing the pinned cache-format file
    def cost(arm,temporal=5,batch=2):
        # H7 base cached as teacher logits during fit. Dense adapter MACs only (same formula as v1.1 M1/M1+M2).
        from ablation_modules import HISTORIES,TRAINABLE
        if arm not in HISTORIES or type(temporal) is not int or temporal<2:raise ValueError('Unknown arm/temporal.')
        c,d,l,n=2048,512,55,30;histories=HISTORIES[arm]
        projection=batch*l*c*d;recurrent=batch*n*sum(histories)*8*d*d
        attention=0 if arm=='r7_last_state' else batch*n*len(histories)*d*d+batch*n*sum(histories)*d*d+2*batch*n*sum(histories)*d
        fusion=batch*n*(3*d)*3 if arm=='g3_learned' else 0
        adapter=projection+recurrent+attention+fusion+2*batch*n*d
        return dict(arm=arm,batch=batch,temporal_positions=temporal,trainable_parameters=TRAINABLE[arm],baseline_forward_GMAC=0.,
                    adapter_forward_GMAC=adapter/1e9,total_forward_GMAC=adapter/1e9,
                    scope='adapter dense MACs only; frozen H7 logits precomputed once per video; excludes CNN/backward/I-O')
    runtime_profile.analytic_cost=cost
    return integration,study,driver
