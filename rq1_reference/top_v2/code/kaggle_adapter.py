"""Execution, I/O and persistence adapters; the published TOP model/head/loss stay in accident_anticipation."""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.dist import barrier, get_rank
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.evaluator import BaseMetric
from mmaction.registry import DATASETS, METRICS, TRANSFORMS
# datasets/__init__.py of the release does not export MultiDataset; importing its module registers it.
from accident_anticipation.datasets.multi import MultiDataset
from accident_anticipation.metrics import AnticipationMetric

from repro import write_json, json_scalar
from v6_reference import V6SampleFramesBeforeAccident, v6_calculate


@TRANSFORMS.register_module()
class NativeJpegDecode(BaseTransform):
    """Author frame indices and OpenCV JPEG pixels, without writing millions of JPGs."""
    def transform(self, results):
        import cv2
        import mmcv

        indices = np.asarray(results["frame_inds"]).reshape(-1)
        if not len(indices) or np.any(indices < 0) or np.any(indices >= results["total_frames"]):
            raise RuntimeError("Native frame indices are outside the original video; refusing to clamp.")
        wanted = set(int(i) for i in indices)
        cap = cv2.VideoCapture(results["filename"])
        cache = {}
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != results["total_frames"]:
                raise RuntimeError(f"Cannot open original video/frame count: {results['filename']}")
            first, last = min(wanted), max(wanted)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, first) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - first) > .01:
                raise RuntimeError("OpenCV native-frame seek failed.")
            for index in range(first, last + 1):
                if not cap.grab():
                    raise RuntimeError(f"Video decode ended early at original frame {index}.")
                if index in wanted:
                    ok, bgr = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f"Cannot decode original frame {index}.")
                    # Same default JPEG parameters as author's cv2.imwrite('.jpg', frame).
                    ok, encoded = cv2.imencode(".jpg", bgr)
                    if not ok:
                        raise RuntimeError("JPEG encoding failed.")
                    cache[index] = mmcv.imfrombytes(encoded.tobytes(), channel_order="rgb", backend="cv2")
        finally:
            cap.release()
        imgs = [cache[int(i)].copy() for i in indices]
        results["imgs"] = imgs
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"] = imgs[0].shape[:2]
        return results


@DATASETS.register_module()
class NexarSubsetDataset(MultiDataset):
    def __init__(self, split="train", seed=42, **kwargs):
        self.split = split
        # Use the author's inherited prepare_data and worker RNG, not a per-video reseeding policy.
        super().__init__(**kwargs)

    def load_data_list(self):
        records = super().load_data_list()
        for record in records:
            # The released sampler/loss read `have_accident`; the released Nexar loader stores the label as `target`.
            record["have_accident"] = record["target"]
        if self.split == "train":
            return [r for r in records if not r["is_test"] and not r["is_val"]]
        if self.split == "validation":
            return [r for r in records if r["is_val"] and not r["is_test"]]
        if self.split == "test":
            return [r for r in records if r["is_test"]]
        raise ValueError(f"Unknown split: {self.split}")


def output_directory():
    return Path(os.environ["TOP_RUN_DIR"])


@METRICS.register_module()
class SavingAnticipationMetric(AnticipationMetric):
    calculate = v6_calculate

    def compute_metrics(self, results):
        if not results or any(not r['is_val'] or r['is_test'] for r in results):
            raise RuntimeError('Matched validation evaluator accepts held-out validation records only.')
        # Avoid native TOP evaluator side effects and keep the V6 calculation exactly.
        values = self.calculate([r['pred'] for r in results], [r['target'] for r in results],
                                [r['abnormal_start_ind'] for r in results],
                                [r['accident_ind'] for r in results], '@')
        directory = output_directory()
        write_json(directory / "validation_metrics.json", dict(values))
        write_json(directory / "validation_predictions.json", [
            {"id": r["video_id"], "target": int(r["target"]),
             "pred": np.asarray(r["pred"]).tolist(),
             "frame_inds": np.asarray(r["frame_inds"]).tolist(),
             "abnormal_start_ind": int(r["abnormal_start_ind"]),
             "accident_ind": int(r["accident_ind"])}
            for r in results if r["is_val"] and not r["is_test"]
        ])
        return values


@METRICS.register_module()
class TestPredictionMetric(BaseMetric):
    def process(self, data_batch, data_samples):
        for sample in data_samples:
            if hasattr(sample, "to_dict"):
                sample = sample.to_dict()
            if sample["target"] is not None or not sample["is_test"]:
                raise RuntimeError("Test inference must not receive test labels.")
            # Released AnticipationMetric submission rule: max over the 20 horizon scores of the last snippet.
            self.results.append({"id": sample["video_id"], "score": float(sample["pred_score"][-1].max().cpu())})

    def compute_metrics(self, results):
        ids = [r["id"] for r in results]
        if len(set(ids)) != len(ids) or len(ids) != 1344:
            raise RuntimeError("Incomplete or duplicated test predictions.")
        write_json(output_directory() / "test_predictions.json", sorted(results, key=lambda r: r["id"]))
        return {"num_test_predictions": len(results)}


@HOOKS.register_module()
class ReproductionStateHook(Hook):
    priority = "LOWEST"

    def __init__(self, deadline, seed):
        self.deadline = deadline
        self.seed = seed
        self.epoch_start = None

    def before_train(self, runner):
        # Native checkpoint paths point at a prior session. Restore to copied artifact basenames.
        for key in ("best_ckpt", "last_ckpt"):
            value = runner.message_hub.get_info(key)
            if value:
                runner.message_hub.update_info(key, str(Path(runner.work_dir) / Path(value).name))
        for hook in runner.hooks:
            for attribute in ("best_ckpt_path", "last_ckpt"):
                value = getattr(hook, attribute, None)
                if value:
                    setattr(hook, attribute, str(Path(runner.work_dir) / Path(value).name))
        sidecar = Path(runner.work_dir) / f"rng_epoch{runner.epoch}_rank{get_rank()}.pth"
        if runner.epoch > 0:
            if not sidecar.exists():
                raise RuntimeError("Resume RNG sidecar missing. Attach the complete prior run output.")
            state = torch.load(sidecar, map_location="cpu", weights_only=False)
            if state["epoch"] != runner.epoch:
                raise RuntimeError("Checkpoint/RNG epoch mismatch; refusing an inexact resume.")
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state["cuda"], device=torch.cuda.current_device())

    def before_train_epoch(self, runner):
        self.epoch_start = time.monotonic()
        # Flush accumulation at each epoch, including its final partial effective batch.
        runner.optim_wrapper.initialize_count_status(
            runner.model, init_counts=0, max_counts=len(runner.train_dataloader),
        )

    def after_val_epoch(self, runner, metrics=None):
        if self.epoch_start is None:  # A standalone evaluation, not a training epoch.
            return
        epoch = runner.epoch  # The epoch loop increments before calling validation.
        sidecar = Path(runner.work_dir) / f"rng_epoch{epoch}_rank{get_rank()}.pth"
        temporary = sidecar.with_suffix(".tmp")
        torch.save({
            "epoch": epoch, "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, temporary)
        temporary.replace(sidecar)
        barrier()
        if get_rank() == 0:
            runner.save_checkpoint(
                runner.work_dir, "resume_staging.pth", save_optimizer=True,
                save_param_scheduler=True, meta={"epoch": epoch, "iter": runner.iter},
                by_epoch=True,
            )
            (Path(runner.work_dir) / "resume_staging.pth").replace(
                Path(runner.work_dir) / "resume_latest.pth")
            write_json(Path(runner.work_dir) / "training_status.json", {
                "completed_epochs": epoch, "target_epochs": runner.max_epochs,
                "complete": epoch >= runner.max_epochs,
                "checkpoint_selection": "validation full mAUC@ (published config)",
                "best_checkpoint": Path(runner.message_hub.get_info("best_ckpt")).name,
            })
            with (Path(runner.work_dir) / "validation_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "metrics": dict(metrics or {})}, default=json_scalar) + "\n")
        if get_rank() == 0:
            from checkpoint_safety import commit_snapshot
            commit_snapshot(Path(runner.work_dir))
        barrier()
        elapsed = time.monotonic() - self.epoch_start
        # All ranks use rank-0 decision so wall-clock differences cannot deadlock DDP.
        stop = torch.tensor([int(time.time() + max(900, elapsed * 1.5) >= self.deadline)],
                            device=next(runner.model.parameters()).device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stop, src=0)
        if bool(stop.item()) and epoch < runner.max_epochs:
            runner.train_loop.stop_training = True
            if get_rank() == 0:
                print("SESSION_BUDGET_STOP: checkpoint complete; resume next session. Test not unlocked.", flush=True)
