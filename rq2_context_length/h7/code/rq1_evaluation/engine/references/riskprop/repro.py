"""Kaggle reproduction controller. Keep published model/loss in a separate source copy."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile

PROTOCOL = 'riskprop-v6-vast-batch2-eff16-amp-single-gpu-3seed-terminal-v3'
SOURCE_COMMIT = "579376fa1d879f28a9d68c2565d1e449de24b666"
SOURCE_ZIP_SHA256 = "7d3f6f28b027f5e53802600afde28980c38e1d77409e2934da742069f99571bf"
DATA_REVISION = "7535d0656dac31d7da2846913bb69c6331d2a70a"
PREPROCESS_ZIP_SHA256 = "5626cf3268f6223ff4f571db17ca433cbcc687a0991ff1934cb9106c314e169b"
ANNOTATIONS_SHA256 = "0cc944e13e11cdeed50036c2a28b190fb675173aed964b66c5c713ca87989d50"
PAPER_VALUES = {"mAP": 0.870, "mAUC": 0.869, "mAUC_0.1": 0.472, "mTTA_0.1_s": 0.958}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_scalar), encoding="utf-8")
    temporary.replace(path)


def json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_patches(root):
    """Only mechanical execution repairs. Do not change FFR, AMC or their reductions."""
    root = Path(root)
    path = root / "taa/models.py"
    original = path.read_text(encoding="utf-8")
    repaired = original.replace(
        "loss_cls=loss_cls * loss_cls_weight\n",
        "loss_cls=loss_cls * loss_cls_weight,\n",
    ).replace(
        "self.pos_weight = torch.tensor(pos_weight)",
        'self.register_buffer("pos_weight", torch.tensor(pos_weight), persistent=False)',
    )
    ast.parse(repaired)
    path.write_text(repaired, encoding="utf-8")
    (root / "taa/__init__.py").write_text(
        "from .datasets import *\nfrom .transforms import *\nfrom .models import *\n"
        "from .metrics import *\nfrom .hooks import *\n", encoding="utf-8",
    )
    return [
        "Fix missing comma in taa/models.py; no loss formula change.",
        "Register scalar pos_weight as non-persistent buffer to follow the model device.",
        "Import taa.models and remove import of absent taa.model_AdaLEA.",
    ]


def prepare_source(settings):
    root = Path(settings["workspace"]) / "author_source"
    if root.exists():
        record = root / "reproduction_source.json"
        if not record.exists():
            raise RuntimeError(f"Refusing to overwrite untracked source directory: {root}")
        info = json.loads(record.read_text())
        for relative, expected in info["patched_hashes"].items():
            if sha256(root / relative) != expected:
                raise RuntimeError(f"Source changed since setup: {relative}")
        return root
    archive = Path(settings["workspace"]) / "author_source_original.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(Path(settings["input_root"]).rglob("RiskProp-main.zip"))
    if len(candidates) > 1:
        raise RuntimeError("More than one RiskProp-main.zip; attach only the audited source ZIP.")
    if candidates:
        if sha256(candidates[0]) != SOURCE_ZIP_SHA256:
            raise RuntimeError("RiskProp-main.zip does not match the audited archive.")
        shutil.copyfile(candidates[0], archive)
        origin = "audited_user_zip"
    else:
        urllib.request.urlretrieve(
            f"https://github.com/xingyueye5/RiskProp/archive/{SOURCE_COMMIT}.zip", archive,
        )
        origin = "pinned_official_github_commit"
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0].split("/")[0] + "/"
        for member in z.infolist():
            relative = Path(member.filename[len(prefix):])
            if not relative.parts or relative == Path("."):
                continue
            destination = (root / relative).resolve()
            if root.resolve() not in destination.parents:
                raise RuntimeError("Unsafe source ZIP member.")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(z.read(member))
    reference = json.loads((Path(__file__).parent / "source_hashes.json").read_text())
    for relative, expected in reference.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Official source differs from audited reference: {relative}")
    patches = source_patches(root)
    info = {
        "commit": SOURCE_COMMIT, "origin": origin, "archive_sha256": sha256(archive),
        "patches": patches, "original_hashes": reference,
        "patched_hashes": {p: sha256(root / p) for p in reference},
    }
    write_json(root / "reproduction_source.json", info)
    return root


def author_validation(root):
    tree = ast.parse((Path(root) / "taa/splits.py").read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "nexar_val" for t in node.targets
        ):
            result = ast.literal_eval(node.value)
            if len(result) != 300 or sum(result.values()) != 150:
                raise RuntimeError("Unexpected author validation split.")
            return result
    raise RuntimeError("Author validation IDs not found.")


def mp4_info(path):
    """Read the video sample table without decoding or assuming frame_count=duration*30."""
    def boxes(f, start, end):
        pos = start
        while pos + 8 <= end:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            header = 8
            if size == 1:
                size, = struct.unpack(">Q", f.read(8))
                header = 16
            elif size == 0:
                size = end - pos
            if size < header or pos + size > end:
                raise ValueError(f"Invalid MP4 box: {path}")
            yield kind, pos + header, pos + size
            pos += size

    def payload(f, box):
        f.seek(box[1])
        return f.read(box[2] - box[1])

    with Path(path).open("rb") as f:
        moov = next(b for b in boxes(f, 0, Path(path).stat().st_size) if b[0] == b"moov")
        for track in boxes(f, moov[1], moov[2]):
            if track[0] != b"trak":
                continue
            mdia = next(b for b in boxes(f, track[1], track[2]) if b[0] == b"mdia")
            media = list(boxes(f, mdia[1], mdia[2]))
            handler = payload(f, next(b for b in media if b[0] == b"hdlr"))
            if handler[8:12] != b"vide":
                continue
            mdhd = payload(f, next(b for b in media if b[0] == b"mdhd"))
            offset = 20 if mdhd[0] == 1 else 12
            scale, = struct.unpack_from(">I", mdhd, offset)
            duration, = struct.unpack_from(">Q" if mdhd[0] == 1 else ">I", mdhd, offset + 4)
            minf = next(b for b in media if b[0] == b"minf")
            stbl = next(b for b in boxes(f, minf[1], minf[2]) if b[0] == b"stbl")
            stsz = payload(f, next(b for b in boxes(f, stbl[1], stbl[2]) if b[0] == b"stsz"))
            count, = struct.unpack_from(">I", stsz, 8)
            if count <= 0:
                raise ValueError(f"Empty video track: {path}")
            return count, duration / scale
    raise ValueError(f"Video track absent: {path}")


def preprocessing_annotations(source):
    """Use the author's published table, not regenerated 30-Hz annotations."""
    directory = Path(__file__).parent / "author_preprocessing"
    hashes = json.loads((directory / "hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"Published preprocessing file changed: {name}")
    annotation = directory / "annotations.csv"
    if sha256(annotation) != ANNOTATIONS_SHA256:
        raise RuntimeError("Unexpected author Nexar annotation release.")
    records = read_csv(annotation)
    if len(records) != 2844 or len({r["id"] for r in records}) != 2844:
        raise RuntimeError("Author annotation IDs are incomplete or duplicated.")
    val = author_validation(source)
    provided = read_csv(directory / "nexar_val_annotations.csv")
    if {r["video_id"]: int(r["have_accident"]) for r in provided} != val:
        raise RuntimeError("Preprocessing and source validation splits differ.")
    return annotation, {r["id"]: r for r in records}


def prepare_data(settings, source):
    snapshot = Path(__file__).parent / "metadata_snapshot"
    inventory = read_csv(snapshot / "video_inventory.csv")
    if len(inventory) != 2844 or len({r["id"] for r in inventory}) != 2844:
        raise RuntimeError("Broken reference video inventory.")
    search = Path(settings.get("data_root") or settings["input_root"])
    if not search.exists():
        raise FileNotFoundError(f"Dataset directory missing: {search}")
    paths = {}
    for p in search.rglob("*.mp4"):
        paths.setdefault(p.stem, []).append(p)
    destination = Path(settings["workspace"]) / "prepared_nexar"
    destination.mkdir(parents=True, exist_ok=True)
    val = author_validation(source)
    annotation_path, author_rows = preprocessing_annotations(source)
    if set(author_rows) != {r["id"] for r in inventory}:
        raise RuntimeError("Author annotations and verified HF inventory differ.")
    identities = []
    for index, row in enumerate(inventory):
        candidates = paths.get(row["id"], [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"ID {row['id']}: found {len(candidates)} MP4 files. Attach all videos once; "
                "set DATA_ROOT if you attached duplicate datasets."
            )
        path = candidates[0]
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Wrong video size: {path}")
        if settings["verify_video_hashes"] and sha256(path) != row["sha256"]:
            raise RuntimeError(f"Video differs from verified HF release: {path}")
        is_test = row["split"] != "train"
        published = author_rows[row["id"]]
        if int(published["is_test"]) != int(is_test) or int(published["target"]) != (-1 if is_test else int(row["target"])):
            raise RuntimeError(f"Author annotation split/label differs: {row['id']}")
        original_count, duration = mp4_info(path)
        if original_count != int(published["total_frames"]) or abs(original_count / duration - float(published["fps"])) > 1e-6:
            raise RuntimeError(f"Video frame count/FPS differs from author annotations: {row['id']}")
        for field, timestamp in [("accident_frame", "time_of_event"), ("abnormal_start_frame", "time_of_alert")]:
            expected = int(float(row[timestamp]) * float(published["fps"])) if not is_test and row["target"] == "1" else -1
            if int(published[field]) != expected or expected >= original_count:
                raise RuntimeError(f"Author timestamp/anchor mismatch: {row['id']} {field}")
        if row["id"] in val and int(row["target"]) != val[row["id"]]:
            raise RuntimeError("Author validation label mismatch.")
        link = destination / ("test" if is_test else "train") / path.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != path.resolve():
                raise RuntimeError(f"Existing link points to another dataset: {link}")
        elif link.exists():
            if not os.path.samefile(link, path):
                raise RuntimeError(f"Refusing to replace existing non-link video: {link}")
        else:
            try:
                link.symlink_to(path.resolve())
            except OSError as error:
                if os.name != "nt" or getattr(error, "winerror", None) != 1314:
                    raise
                # Windows diagnostic only: same-volume hard link, no copied video bytes.
                os.link(path.resolve(), link)
        identities.append({"id": row["id"], "sha256": row["sha256"], "bytes": row["bytes"]})
        if (index + 1) % 250 == 0:
            print(f"Verified/indexed {index + 1}/2844 videos", flush=True)
    # Preserve the original order/schema/bytes, including unknown test target=-1.
    shutil.copyfile(annotation_path, destination / "annotations.csv")
    fingerprint = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    info = {
        "data_revision": DATA_REVISION, "video_count": 2844, "train_count": 1200,
        "validation_count": 300, "test_count": 1344, "inventory_fingerprint": fingerprint,
        "verified_content_hashes": settings["verify_video_hashes"],
        "preprocessing_archive_sha256": PREPROCESS_ZIP_SHA256,
        "annotations_sha256": ANNOTATIONS_SHA256,
        "timestamp_mapping": "reported: original author CSV, int(timestamp * native video FPS), zero-based",
        "decode": "OpenCV native-frame decode + default JPEG encode/decode on demand; avoids full JPG materialization",
        "sampling_clock": "reported source: fps=30 in loader, stride=3 original frame indices; not uniform physical 10 Hz for non-30-FPS videos",
        "validation_ids": sorted(val),
    }
    write_json(destination / "data_manifest.json", info)
    return destination


def config(settings, source, data, phase="train"):
    sys.path.insert(0, str(source))
    from mmengine.config import Config

    cfg = Config.fromfile(str(source / "configs/predict_anomaly_snippet.py"))
    for pipeline in (cfg.train_pipeline_frame, cfg.val_pipeline_frame):
        for index, transform in enumerate(pipeline):
            if transform["type"] == "RawFrameDecode":
                pipeline[index] = dict(type="NativeJpegDecode")
    world = settings["world_size"]
    batch = settings["batch_per_gpu"]
    denominator = batch * world
    if settings["effective_batch_size"] % denominator:
        raise ValueError("Effective batch size must be divisible by batch_per_gpu * world_size.")
    accumulation = settings["effective_batch_size"] // denominator
    cfg.custom_imports = dict(imports=["taa", "kaggle_adapter", "amp_adapter"], allow_failed_imports=False)
    cfg.work_dir = str(Path(settings["workspace"]) / "run")
    cfg.launcher = "pytorch" if world > 1 else "none"
    cfg.randomness = dict(seed=settings["seed"], diff_rank_seed=False, deterministic=False)
    cfg.env_cfg.mp_cfg.mp_start_method = "spawn" if os.name == "nt" else "fork"
    cfg.model_wrapper_cfg = dict(type="MMDistributedDataParallel", find_unused_parameters=False)
    cfg.train_dataloader.batch_size = batch
    cfg.train_dataloader.num_workers = settings["workers_per_gpu"]
    cfg.train_dataloader.persistent_workers = settings["workers_per_gpu"] > 0
    dataset = dict(
        type="NexarSubsetDataset", nexar=dict(
            data_root=str(data), ann_file="annotations.csv", filename_tmpl="{:06}.jpg", start_index=0,
        ),
        pipeline_video=cfg.train_pipeline_frame, pipeline_frame=cfg.train_pipeline_frame,
        modality="rgb", test_mode=False, split="train", seed=settings["seed"],
    )
    cfg.train_dataloader.dataset = dataset
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = settings["workers_per_gpu"]
    cfg.val_dataloader.persistent_workers = settings["workers_per_gpu"] > 0
    cfg.val_dataloader.dataset = dict(
        dataset, pipeline_video=cfg.val_pipeline_frame, pipeline_frame=cfg.val_pipeline_frame,
        test_mode=True, split="validation",
    )
    cfg.val_evaluator = dict(type="SavingAnticipationMetric", fpr_max=0.1, vis_list=[], output_dir=None)
    cfg.test_dataloader = dict(cfg.val_dataloader)
    cfg.test_dataloader["dataset"] = dict(cfg.val_dataloader.dataset, split="test")
    cfg.test_evaluator = dict(type="TestPredictionMetric")
    cfg.optim_wrapper.type = "AmpOptimWrapper" if settings["amp"] else "OptimWrapper"
    cfg.optim_wrapper.accumulative_counts = accumulation
    if settings["amp"]:
        cfg.optim_wrapper.dtype = "float16"
        cfg.optim_wrapper.loss_scale = dict(init_scale=1024., growth_interval=2000)
    cfg.train_cfg.max_epochs = settings["epochs"]
    cfg.val_cfg = dict(type="ValLoop", fp16=False)
    cfg.test_cfg = dict(type="TestLoop", fp16=False)
    # Limit inference views only; it has no effect on training or causal snippet selection.
    cfg.model.test_cfg = dict(max_testing_views=10)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="mAUC@", rule="greater",
    )
    cfg.custom_hooks = [
        dict(type="EpochHook"),
        dict(type="ReproductionStateHook", deadline=settings["deadline"], seed=settings["seed"]),
    ]
    cfg.visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend")])
    if phase == "smoke":
        cfg.work_dir = str(Path(settings["workspace"]) / "smoke")
        cfg.resume = False
    else:
        resume = Path(cfg.work_dir) / "resume_latest.pth"
        cfg.resume = resume.exists()
        if cfg.resume:
            cfg.load_from = str(resume)
    return cfg


def restore_run(settings):
    previous = settings.get("resume_root")
    if not previous:
        return
    previous = Path(previous)
    if not (previous / "run_manifest.json").exists():
        raise RuntimeError("RESUME_ROOT must contain run_manifest.json, resume_latest.pth and best*.pth.")
    info = json.loads((previous / "run_manifest.json").read_text())
    contract = run_contract(settings)
    if info["contract"] != contract:
        raise RuntimeError("Resume protocol does not match current settings; refusing a mixed experiment.")
    status_path = previous / 'training_status.json'
    if not status_path.exists():
        raise RuntimeError('Resume requires the complete saved run, including training_status.json.')
    status = json.loads(status_path.read_text())
    if Path(status['best_checkpoint']).name != status['best_checkpoint']:
        raise RuntimeError('Best checkpoint must be a filename inside the saved run directory.')
    required = ['resume_latest.pth', status['best_checkpoint']] + [
        f"rng_epoch{status['completed_epochs']}_rank{rank}.pth" for rank in range(settings['world_size'])]
    missing = [name for name in required if not (previous / name).is_file()]
    if missing:
        raise RuntimeError(f'Resume artifacts missing: {missing}. Attach the entire previous run output.')
    target = Path(settings["workspace"]) / "run"
    target.mkdir(parents=True, exist_ok=True)
    for p in previous.iterdir():
        if p.is_file() and (p.suffix in {".pth", ".json", ".jsonl", ".csv"} or p.name == "last_checkpoint"):
            destination = target / p.name
            if destination.exists() and sha256(destination) != sha256(p):
                raise RuntimeError(f"Resume would overwrite a different artifact: {destination}")
            if not destination.exists():
                shutil.copyfile(p, destination)


def run_contract(settings):
    return {
        "protocol": PROTOCOL, "source_commit": SOURCE_COMMIT, "data_revision": DATA_REVISION,
        **{k: settings[k] for k in [
            "seed", "epochs", "batch_per_gpu", "effective_batch_size", "world_size", "amp",
        ]},
        "precision": "train AMP float16, published losses FP32, validation/test FP32; dynamic scale starts 1024",
        "loss": "published_code_no_algorithmic_changes", "preprocessing": "author-native-frame-opencv-jpeg-v2",
        "preprocessing_archive_sha256": PREPROCESS_ZIP_SHA256, "annotations_sha256": ANNOTATIONS_SHA256,
        "runtime_contract": settings.get("runtime_contract"),
        "launcher_sha256": {name: sha256(Path(__file__).parent / name)
                            for name in ('repro.py', 'worker.py', 'kaggle_adapter.py', 'evaluate.py', 'amp_adapter.py', 'run_vast.py', 'run_three_seeds.py', 'process_control.py', 'checkpoint_safety.py')},
    }


def launch(settings, source, phase):
    command = [sys.executable]
    if settings["world_size"] > 1:
        command += ["-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={settings['world_size']}"]
    command += [str(Path(__file__).parent / "worker.py"), "--settings", settings["settings_file"], "--phase", phase]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = "2"
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # Only official pretraining and own trusted checkpoints.
    subprocess.run(command, check=True, env=env, cwd=Path(settings["workspace"]))


def validate_profile(settings):
    if type(settings.get("seed")) is not int or settings["seed"] not in (42, 43, 44):
        raise ValueError("Three-seed experiment requires seed 42, 43 or 44.")
    expected = dict(epochs=50, batch_per_gpu=2, effective_batch_size=16,
                    world_size=1, amp=True, workers_per_gpu=2, verify_video_hashes=True)
    for key, value in expected.items():
        if type(settings.get(key)) is not type(value) or settings[key] != value:
            raise ValueError(f"Locked code-batch2 protocol requires {key}={value!r}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "smoke", "train", "evaluate"])
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    settings_path = Path(args.settings).resolve()
    settings = json.loads(settings_path.read_text())
    settings["settings_file"] = str(settings_path)
    validate_profile(settings)
    source = prepare_source(settings)
    if args.stage == "prepare":
        data = prepare_data(settings, source)
        write_json(Path(settings["workspace"]) / "preflight.json", {
            "source": str(source), "data": str(data), "protocol": PROTOCOL,
            "paper_values": PAPER_VALUES, "training_not_yet_run": True,
        })
        return
    data = Path(settings["workspace"]) / "prepared_nexar"
    if not (data / "data_manifest.json").exists():
        raise RuntimeError("Run the prepare cell first.")
    if args.stage == "train":
        smoke_path = Path(settings["workspace"]) / "smoke_report.json"
        if not smoke_path.is_file():
            raise RuntimeError("Run the GPU smoke test first; no automatic memory fallback.")
        smoke = json.loads(smoke_path.read_text())
        if not smoke.get("passed") or not smoke.get("mini_validation_passed") or smoke.get("contract") != run_contract(settings):
            raise RuntimeError("Smoke result is stale or from a different protocol; rerun smoke.")
        restore_run(settings)
        work = Path(settings["workspace"]) / "run"
        work.mkdir(parents=True, exist_ok=True)
        manifest = work / "run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["contract"] != run_contract(settings):
            raise RuntimeError("Existing run settings differ; use a new workspace.")
        write_json(manifest, {
            "contract": run_contract(settings), "paper_values": PAPER_VALUES,
            "deviations": [
                "Vast.ai single RTX 4080 replaces paper 8 x A800 and Kaggle 2 x T4; separate experiment, never mixed with Kaggle checkpoints.",
                "README Python 3.8.5 / MMEngine 0.10.7 / full MMCV 2.2.0 / author MMAction2 1.2.0; Torch 2.1.0 CUDA 11.8 and other pins remain inferred compatible choices.",
                "OpenCV native-frame/JPEG roundtrip on demand replaces pre-extracting every JPEG; validate decoder/build parity.",
                "Author native-FPS annotation CSV is used unchanged; original loader nominal fps=30 is retained, not a uniform physical-time grid.",
                "AMP training is an explicit 16-GB GPU memory adaptation, NOT released FP32 OptimWrapper. Dynamic GradScaler starts at 1024; amp_adapter calls the untouched published loss in FP32. Validation/test stay FP32.",
                "True local batch=2 matches released snippet config; effective batch=16 is inferred from README eight-GPU command, NOT the paper batch=64.",
                "One GPU uses eight accumulation steps: 2 x 1 x 8 = effective16. No DDP. Loss is still evaluated on each local batch of two. Sampling, BN running statistics and floating-point reduction differ from Kaggle world2 and paper; not an isolated causal ablation.",
                "Published dist_train.sh separately uses six GPUs, so effective batch=16 is a selected README/config reconstruction, not proof of Table 1 training settings.",
                "Workers per GPU=2 instead of released config 8; validation/test batch=1 to satisfy the released max_testing_views batch-1 assertion. Training remains batch=2.",
                "Fresh from original pretraining. Same local batch2/effective16 as Kaggle v6, but different GPU/rank grouping. v3/v4/v5/Kaggle-v6 checkpoints are rejected.",
                "Published code applies FFR to negatives and AMC mean over the microbatch; paper differs.",
                "Published code pos_weight=1; paper collision weight is unspecified.",
                f"Explicit seed {settings['seed']} is a reproduction choice, not an author-reported seed.",
                "Original augmentation RNG is retained. Seed is explicit; worker RNG after session restart is not bitwise-equivalent when workers>0.",
                "Published positive training jitter of 0..2 clock frames is retained, unlike strict pre-event wording.",
                "Best checkpoint uses published-code validation full mAUC@, not README partial-AUC description.",
                "Validation proxies and official held-out test metrics have different scopes.",
            ],
        })
    if args.stage == "evaluate":
        manifest = json.loads((Path(settings["workspace"])/"run/run_manifest.json").read_text())
        if manifest["contract"] != run_contract(settings):
            raise RuntimeError("Evaluation code/runtime differs from the completed training contract.")
    launch(settings, source, args.stage)


if __name__ == "__main__":
    main()
