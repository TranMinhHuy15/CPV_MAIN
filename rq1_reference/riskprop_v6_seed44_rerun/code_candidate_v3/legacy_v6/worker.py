"""Distributed worker, invoked by torchrun; never execute in a notebook kernel."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from repro import config, write_json, run_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--phase", choices=["smoke", "train", "evaluate"], required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    accumulation = settings["effective_batch_size"] // (settings["batch_per_gpu"] * settings["world_size"])
    source = Path(settings["workspace"]) / "author_source"
    data = Path(settings["workspace"]) / "prepared_nexar"
    sys.path.insert(0, str(source))
    import torch
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select the single RTX 4080 and a CUDA-enabled Linux container.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if int(os.environ.get("RANK", 0)) == 0:
        write_json(Path(settings["workspace"]) / "environment.json", {
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opencv_version": cv2.__version__, "opencv_build": cv2.getBuildInformation(),
            "gpus": [{"name": torch.cuda.get_device_name(i),
                      "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(settings["world_size"])],
            "packages": {name: importlib.metadata.version(name) for name in
                         ["mmengine", "mmcv", "numpy", "pandas", "scikit-learn", "opencv-python"]},
        })
    from mmengine.runner import Runner
    from mmengine.dist import get_rank
    cfg = config(settings, source, data, args.phase)
    # config() adds custom_imports after Config.fromfile; load them explicitly in every fresh worker.
    from mmengine.utils import import_modules_from_strings
    import_modules_from_strings(**cfg.custom_imports)
    os.environ["RISKPROP_RUN_DIR"] = cfg.work_dir
    cfg.dump(str(Path(settings["workspace"]) / f"resolved_{args.phase}_config.py")) if int(os.environ.get("RANK", 0)) == 0 else None
    if args.phase == "evaluate":
        work = Path(settings["workspace"]) / "run"
        status = json.loads((work / "training_status.json").read_text())
        if not status["complete"]:
            if int(os.environ.get("RANK", 0)) == 0:
                print("PARTIAL_RUN: held-out test evaluation skipped until the requested 50 epochs are complete.", flush=True)
            return
        cfg.resume = False
        cfg.load_from = str(work / status["best_checkpoint"])
        cfg.custom_hooks = [dict(type="EpochHook")]
        # Evaluation never saves/reselects training checkpoints.
        cfg.default_hooks["checkpoint"] = None
    runner = Runner.from_cfg(cfg)
    if args.phase == "smoke":
        runner.model.init_weights() if not hasattr(runner.model, "module") else runner.model.module.init_weights()
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(runner.model, cfg.load_from, map_location="cpu")
        loader = runner.train_dataloader
        if len(loader.dataset) != 1200 or len(runner.val_dataloader.dataset) != 300:
            raise RuntimeError("Wrong training/validation split.")
        runner.model.train()
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
        # Smoke-only: flush a final partial accumulation to verify AMP/SGD actually step.
        runner.optim_wrapper.initialize_count_status(runner.model, init_counts=0, max_counts=16)
        timings = []
        losses = []
        iterator = iter(loader)
        backbone = (runner.model.module if hasattr(runner.model, 'module') else runner.model).backbone
        probe_before = backbone.conv1.conv.weight.detach().clone()
        gradient_present = False
        for index in range(16):
            torch.cuda.synchronize()
            start = time.monotonic()
            batch = next(iterator)
            if len(batch["data_samples"]) != 2 or len(batch["inputs"]) != 2:
                raise RuntimeError("Smoke must process two real videos together, not two batch-1 forwards.")
            try:
                values = runner.model.train_step(batch, runner.optim_wrapper)
            except torch.cuda.OutOfMemoryError:
                print("SMOKE_OOM: batch 2 + AMP still exceeds this GPU. Training is blocked; no automatic input/backbone reduction.", flush=True)
                raise
            gradient_present |= backbone.conv1.conv.weight.grad is not None
            torch.cuda.synchronize()
            if not all(torch.isfinite(torch.as_tensor(v)).all() for v in values.values()):
                raise RuntimeError("Non-finite smoke loss.")
            timings.append(time.monotonic() - start)
            losses.append({name: float(torch.as_tensor(value).detach().cpu()) for name, value in values.items()})
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        updated = not torch.equal(probe_before, model.backbone.conv1.conv.weight.detach())
        if not gradient_present or not updated:
            raise RuntimeError("Smoke failed: backbone gradient or SGD update is missing.")
        # Technical mini-validation BEFORE a long train; predictions are not performance results.
        val_dataset = runner.val_dataloader.dataset
        selected = {}
        for target in (True, False):
            selected[target] = [i for i in range(len(val_dataset))
                                if val_dataset.get_data_info(i)["target"] is target][:settings["world_size"]]
            if len(selected[target]) != settings["world_size"]:
                raise RuntimeError("Mini-validation requires one positive and one negative per rank.")
        runner.model.eval()
        with torch.no_grad():
            for target in (True, False):
                sample = val_dataset[selected[target][get_rank()]]
                batch = dict(inputs=[sample["inputs"]], data_samples=[sample["data_samples"]])
                predictions = runner.model.val_step(batch)
                scores = predictions[0].pred_score
                if scores.dtype != torch.float32 or scores.numel() != 30 or not torch.isfinite(scores).all():
                    raise RuntimeError("Mini-validation failed: expected 30 finite FP32 risk scores.")
                runner.val_evaluator.process(data_samples=predictions, data_batch=batch)
        mini_metrics = runner.val_evaluator.evaluate(size=2 * settings["world_size"])
        if not all(torch.isfinite(torch.as_tensor(v)).all() for v in mini_metrics.values()):
            raise RuntimeError("Mini-validation produced non-finite metrics.")
        memory = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device='cuda')
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
        if get_rank() == 0:
            step_seconds = sum(timings[1:]) / len(timings[1:])
            write_json(Path(settings["workspace"]) / "smoke_report.json", {
                "passed": True, "gpu": torch.cuda.get_device_name(),
                "contract": run_contract(settings), "physical_batch_per_gpu": 2,
                "effective_batch_size": 16, "accumulation_steps": accumulation,
                "train_microsteps_per_epoch_per_rank": len(loader),
                "optimizer_steps_per_epoch": (len(loader) + accumulation - 1) // accumulation,
                "rank0_smoke_losses": losses,
                "smoke_microsteps": 16, "nominal_smoke_optimizer_steps": 16 // accumulation,
                "amp_scaler_state": runner.optim_wrapper.loss_scaler.state_dict(),
                "training_precision": "AMP float16 forward; original losses FP32",
                "evaluation_precision": "FP32",
                "mini_validation_passed": True, "mini_validation_videos": 2 * settings["world_size"],
                "mini_validation_scope": "technical pipeline check after discarded smoke updates; NOT a trained-model result",
                "world_size": settings["world_size"], "seconds_per_microstep": step_seconds,
                "rough_train_only_50_epoch_hours": step_seconds * len(loader) * 50 / 3600,
                "estimate_caveat": "15 post-warm-up microsteps including data wait; AMP updates may be skipped if gradients overflow. Excludes validation/preparation/test; not a runtime guarantee.",
                "peak_memory_gb_max_across_gpus": memory.item(),
                "backbone_gradient_present": True, "backbone_sgd_update_verified": True,
            })
        return
    if args.phase == "train":
        runner.train()
    else:
        runner.val()
        runner.test()
        if get_rank() == 0:
            from evaluate import create_reports
            create_reports(Path(cfg.work_dir), Path(__file__).parent / "metadata_snapshot")


if __name__ == "__main__":
    main()
