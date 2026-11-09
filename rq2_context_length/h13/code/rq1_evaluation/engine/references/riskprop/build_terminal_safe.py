"""Preserve v2 package; generate the audited terminal-first safety release."""
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'RiskProp_V6_VastAI_RTX4080_3Seeds'
PROTOCOL = 'riskprop-v6-vast-batch2-eff16-amp-single-gpu-3seed-terminal-v3'


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def main():
    report=json.loads((BASE/'three_seed_port_verification.json').read_text())
    names=list(report['reference_sha256'])+report['changed_single_seed_files']
    names+=['run_three_seeds.py','test_three_seeds.py']
    for name in names:
        path=HERE/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((BASE/name).read_bytes())
    repro=(HERE/'repro.py').read_text(encoding='utf-8').replace(report['protocol'],PROTOCOL)
    repro=once(repro,"'amp_adapter.py')}",
        "'amp_adapter.py', 'run_vast.py', 'run_three_seeds.py', 'process_control.py', 'checkpoint_safety.py')}")
    repro=once(repro,'    launch(settings, source, args.stage)',
        '    if args.stage == "evaluate":\n'
        '        manifest = json.loads((Path(settings["workspace"])/"run/run_manifest.json").read_text())\n'
        '        if manifest["contract"] != run_contract(settings):\n'
        '            raise RuntimeError("Evaluation code/runtime differs from the completed training contract.")\n'
        '    launch(settings, source, args.stage)')
    (HERE/'repro.py').write_bytes(repro.encode('utf-8'))
    adapter=(HERE/'kaggle_adapter.py').read_text(encoding='utf-8')
    adapter=once(adapter,'        barrier()\n        elapsed = time.monotonic() - self.epoch_start',
        '        if get_rank() == 0:\n'
        '            from checkpoint_safety import commit_snapshot\n'
        '            commit_snapshot(Path(runner.work_dir))\n'
        '        barrier()\n        elapsed = time.monotonic() - self.epoch_start')
    (HERE/'kaggle_adapter.py').write_bytes(adapter.encode('utf-8'))
    single=(HERE/'run_vast.py').read_text(encoding='utf-8').replace(report['protocol'],PROTOCOL)
    single=once(single,'import time\n',
        'import time\nfrom process_control import run_checked, DEADLINE_ENV\nfrom checkpoint_safety import restore_snapshot\n')
    start=single.index('def execute(')
    end=single.index('\n\ndef main(',start)
    single=single[:start]+'''def execute(command, log_path, env=None):
    print('Running:', ' '.join(str(arg) for arg in command), flush=True)
    return run_checked(command, log_path=log_path, env=env)
'''+single[end:]
    single=once(single,"    parser.add_argument('--hours', type=float, default=24.)",
        "    parser.add_argument('--hours', type=float, default=24.)\n    parser.add_argument('--job-deadline', type=float)")
    single=once(single,'    args = parser.parse_args(argv)',
        '    args = parser.parse_args(argv)\n'
        '    job_deadline = args.job_deadline if args.job_deadline is not None else time.time()+args.hours*3600\n'
        '    if not math.isfinite(job_deadline) or job_deadline <= time.time():\n'
        '        raise ValueError("Total job deadline must be finite and in the future.")')
    single=once(single,'    existing = workspace / \'run/run_manifest.json\'',
        '    existing = workspace / \'run/run_manifest.json\'\n'
        '    recovery = Path(args.resume_root).resolve() if args.resume_root else (workspace/"run" if args.evaluate_only else None)\n'
        '    if recovery is not None:\n'
        '        restore_snapshot(recovery, PROTOCOL, args.seed)')
    single=once(single,"    env.update(PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS='2')",
        "    env.update(PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS='2')\n"
        "    env[DEADLINE_ENV] = str(job_deadline)")
    single=once(single,'deadline=time.time()+args.hours*3600, runtime_contract=runtime[\'contract\'])',
        'deadline=min(time.time()+args.hours*3600, job_deadline), runtime_contract=runtime[\'contract\'])')
    (HERE/'run_vast.py').write_bytes(single.encode('utf-8'))
    queue=(HERE/'run_three_seeds.py').read_text(encoding='utf-8')
    queue=once(queue,'import math\n','import math\nimport os\nimport time\n')
    queue=once(queue,'from run_vast import PROTOCOL',
        'from run_vast import PROTOCOL\nfrom process_control import run_checked, exclusive_job, DeadlineExceeded, GROUP_ENV, DEADLINE_ENV\nfrom checkpoint_safety import restore_snapshot')
    queue=once(queue,"               '--hours', str(args.hours_per_seed)]",
        "               '--hours', str(args.hours_per_seed), '--job-deadline', str(args.job_deadline)]")
    queue=once(queue,"    parser.add_argument('--hours-per-seed', type=float, default=24.)",
        "    parser.add_argument('--hours-per-seed', type=float, default=24.)\n"
        "    parser.add_argument('--total-hours', type=float, default=24., help='TOTAL hard wall-clock cap, includes setup/evaluation; does not stop billing.')")
    queue=once(queue,'    args = parser.parse_args(argv)',
        '    args = parser.parse_args(argv)\n'
        '    args.job_deadline = time.time()+args.total_hours*3600\n'
        '    if not math.isfinite(args.total_hours) or args.total_hours < 1:\n'
        '        parser.error("--total-hours must be finite and at least 1.")')
    queue=once(queue,'        subprocess.run(command, check=True)',
        '        run_checked(command, env=job_env, deadline=args.job_deadline)')
    start=queue.index("    print('ONE GPU, SEQUENTIAL")
    end=queue.index('\n\n\nif __name__',start)
    body=queue[start:end]
    body=body.replace("    print('ONE GPU, SEQUENTIAL", "    print('TOTAL job cap %.1fh, including setup/evaluation; NOT an instance billing cap.' % args.total_hours, flush=True)\n    print('ONE GPU, SEQUENTIAL")
    wrapped='''    with exclusive_job(root/'job.lock'):
        job_env = dict(os.environ)
        job_env.pop(GROUP_ENV,None)
        job_env.pop(DEADLINE_ENV,None)
        for seed in SEEDS:
            restore_snapshot(root/('seed_%d'%seed)/'run', PROTOCOL, seed)
        try:
            # Before installing/training: test real POSIX group cleanup on THIS Linux host.
            if any(not evaluated(root/('seed_%d'%seed)/'run', saved_state(root,seed)[1]) for seed in SEEDS):
                run_checked([sys.executable, str(HERE/'test_safety.py')], env=job_env, deadline=args.job_deadline)
'''+''.join('        '+line+'\n' for line in body.splitlines())+'''        except (KeyboardInterrupt, DeadlineExceeded, subprocess.CalledProcessError) as error:
            (root/'job_stop.json').write_text(json.dumps(dict(reason=type(error).__name__,
                detail=str(error), total_hours=args.total_hours, deadline=args.job_deadline,
                instruction='Back up all outputs. Rerun same command to restore last committed epoch; instance billing is still active.'),indent=2),encoding='utf-8')
            raise
'''
    queue=queue[:start]+wrapped+queue[end:]
    (HERE/'run_three_seeds.py').write_bytes(queue.encode('utf-8'))
    test=(HERE/'test_three_seeds.py').read_text(encoding='utf-8')
    test=test.replace("patch.object(queue.subprocess,'run'", "patch.object(queue,'run_checked'")
    test=test.replace('def partial(command,check):','def partial(command,**unused):').replace('def complete(command,check):','def complete(command,**unused):')
    test=test.replace('            def partial(command,**unused):\n',
        '            def partial(command,**unused):\n                if "--seed" not in command:\n                    return\n')
    test=test.replace('            def complete(command,**unused):\n',
        '            def complete(command,**unused):\n                if "--seed" not in command:\n                    return\n')
    test=once(test,"        nb = queue.read_json(HERE/'RiskProp_V6_VastAI_RTX4080_3Seeds.ipynb')\n"
        "        for cell in nb['cells']:\n            if cell['cell_type']=='code':\n                compile(''.join(cell['source']),'notebook','exec')",'        # Terminal-first release; all entry points compiled above.')
    test=test.replace("        for name in ('amp_adapter.py'", "        for name in ('amp_adapter.py'")
    # The adapter adds a post-validation persistence call, not an algorithm change.
    test=once(test,'        self.assertEqual(before,after)',
        '        self.assertEqual(before,after)')
    (HERE/'test_three_seeds.py').write_bytes(test.encode('utf-8'))
    for path in HERE.glob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    evidence=dict(protocol=PROTOCOL,seeds=[42,43,44],default_total_hours=24,
        full_rtx4080_linux_224_verified=False,model_loss_sampler_metric_algorithms_unchanged=True,
        changes=['process-group cleanup','global wall-clock budget including setup/evaluation',
                 'exclusive output lock','atomic completed-epoch recovery snapshot','strict evaluation contract'])
    for name in ('three_seed_checks.json','safety_checks.json','reference_checks.json'):
        if (HERE/name).exists():
            evidence[name]=json.loads((HERE/name).read_text())
    diagnostic=HERE.parent/'RiskProp_VastAI_3Seed_Audit_2026-10-07/pretrained_smoke_checks.json'
    if diagnostic.exists():
        (HERE/'pretrained_smoke_checks.json').write_bytes(diagnostic.read_bytes())
        evidence['pretrained_local_gpu_smoke']=json.loads(diagnostic.read_text())
    snapshot_diagnostic=diagnostic.parent/'completed_epoch_snapshot_checks.json'
    if snapshot_diagnostic.exists():
        (HERE/'completed_epoch_snapshot_checks.json').write_bytes(snapshot_diagnostic.read_bytes())
        evidence['real_local_gpu_checkpoint_resume']=json.loads(snapshot_diagnostic.read_text())
    (HERE/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    destination=HERE.parent/(HERE.name+'.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in HERE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path,HERE.name+'/'+path.relative_to(HERE).as_posix())
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print('Built',destination, destination.stat().st_size)


if __name__=='__main__':
    main()
