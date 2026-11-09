"""Vast operations for ONE immutable medium arm; research arithmetic stays in legacy modules."""
import argparse
import csv
import datetime
import getpass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from process_control import exclusive_job, run_checked, DEADLINE_ENV, GROUP_ENV
from rq2_profile import VARIANT, scientific_profile
from run_rq2 import verify_package, state, evaluated

HERE = Path(__file__).resolve().parent
REPO = "nexar-ai/nexar_collision_prediction"
REVISION = "7535d0656dac31d7da2846913bb69c6331d2a70a"
SEEDS = (42, 43, 44)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def inventory():
    with (HERE / "metadata_snapshot/video_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2844 or len({r["id"] for r in rows}) != 2844:
        raise RuntimeError("Broken pinned inventory.")
    return {r["id"]: r for r in rows}


def video_paths(data, expected):
    found = {}
    for path in Path(data).rglob("*.mp4"):
        if path.stem not in expected or path.stem in found:
            raise RuntimeError("Unknown or duplicated MP4 ID: " + path.stem)
        found[path.stem] = path
    return found


def verify_data(data, expected):
    found = video_paths(data, expected)
    if set(found) != set(expected):
        raise RuntimeError("Require 2844 distinct MP4 files; found %d." % len(found))
    for count, (ident, row) in enumerate(sorted(expected.items()), 1):
        path = found[ident]
        if path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
            raise RuntimeError("Dataset size/SHA256 mismatch: " + ident)
        if count % 250 == 0:
            print("Verified %d/2844 videos" % count, flush=True)
    print("All 2844 pinned video hashes match.", flush=True)


def safe_paths(workspace, data, output, runtime):
    paths = {"code": HERE.resolve(), "data": data.resolve(), "output": output.resolve(),
             "runtime": runtime.resolve()}
    for name, first in paths.items():
        for other, second in paths.items():
            if name < other and (first == second or first in second.parents or second in first.parents):
                raise ValueError("Code/data/output/runtime must be separate, non-nested: %s / %s" % (name, other))
    if workspace.resolve() == HERE.resolve() or HERE.resolve() in workspace.resolve().parents:
        raise ValueError("Workspace must not be inside the code directory.")


def ensure_system():
    if sys.platform != "linux":
        raise RuntimeError("Training/download operations target Linux Vast; status/verify/backup also work locally.")
    required = ("ffmpeg", "ffprobe", "nvidia-smi")
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing system tools %s. Install ffmpeg and use a CUDA-enabled container." % missing)
    import ctypes
    for library in ("libGL.so.1", "libglib-2.0.so.0"):
        try:
            ctypes.CDLL(library)
        except OSError:
            raise RuntimeError("Install libgl1 libglib2.0-0 before running.")


def child(command, deadline, log=None):
    env = dict(os.environ)
    env.pop(GROUP_ENV, None)
    env.pop(DEADLINE_ENV, None)
    env["RQ2_VARIANT"] = VARIANT
    print("RUN:", " ".join(str(x) for x in command), flush=True)
    return run_checked([str(x) for x in command], env=env, deadline=deadline, log_path=log)


def prepare(workspace, data, output, deadline, skip_download=False):
    expected = inventory()
    data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    present = video_paths(data, expected)
    # Each job keeps full recovery snapshots; reserve beyond the dataset/runtime.
    reserve = 24 << 30
    if shutil.disk_usage(output).free < reserve:
        raise RuntimeError("Need >=24 GiB free on output filesystem for three runs and verified backup.")
    if len(present) == 2844:
        verify_data(data, expected)
        return
    if skip_download:
        raise RuntimeError("--skip-download selected, but dataset is incomplete.")
    remaining = sum(max(0, int(row["bytes"]) - (present[k].stat().st_size if k in present else 0))
                    for k, row in expected.items())
    if shutil.disk_usage(data).free < remaining + reserve:
        raise RuntimeError("Not enough disk for remaining pinned videos plus 24 GiB reserve.")
    tools = workspace / ("rq2_download_tools_h%d" % scientific_profile()["history_tokens"])
    python = tools / "bin/python"
    if not python.exists():
        child([sys.executable, "-m", "venv", tools], deadline)
    child([python, "-m", "pip", "install", "--quiet", "huggingface_hub==0.26.5"], deadline)
    child([python, HERE / "medium_ops.py", "_download", "--workspace", workspace,
           "--data-root", data], deadline)
    verify_data(data, expected)


def download(data):
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or getpass.getpass(
        "HF Read token (hidden; account must have accepted Nexar access): ")
    if not token:
        raise RuntimeError("HF Read token required; no token was saved.")
    try:
        snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                          local_dir=str(data), token=token, max_workers=4,
                          allow_patterns=["*.mp4", "**/*.mp4", "LICENSE"])
    except Exception as error:
        raise RuntimeError("HF download failed (%s). Check access/network/disk and rerun the same command."
                           % type(error).__name__) from None
    finally:
        token = None


def queue(data, output, runtime, seeds, deadline, phase):
    remaining = (deadline - time.time()) / 3600.
    if remaining <= 0:
        raise RuntimeError("Total budget reached; instance is still billed.")
    command = [sys.executable, HERE / "run_rq2.py", "--data-root", data,
               "--output-root", output, "--runtime-root", runtime,
               "--variants", VARIANT, "--seeds"] + list(seeds) + [
               "--total-hours", str(remaining)]
    if phase in ("train", "evaluate"):
        command.append("--train")
    if phase == "train":
        command.append("--validation-only")
    if phase == "evaluate":
        for seed in seeds:
            _, saved = state(output, VARIANT, seed)
            if not saved or not saved["complete"]:
                raise RuntimeError("Evaluation cannot start training: seed%d has not completed50 epochs." % seed)
    return child(command, deadline, output / "logs" / (phase + "_master.log"))


def all_training_complete(output, seeds):
    return all((state(output, VARIANT, seed)[1] or {}).get("complete", False) for seed in seeds)


def all_evaluated(output, seeds):
    return all(evaluated(*state(output, VARIANT, seed)) for seed in seeds)


def status(output, seeds, epochs=False):
    print("ARM:", VARIANT, "| H =", scientific_profile()["history_tokens"], "| OUTPUT:", output)
    print("seed epoch/50 best_checkpoint train_complete test_marker")
    rows = []
    for seed in seeds:
        run = output / VARIANT / ("seed_%d" % seed) / "run"
        path = run / "training_status.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        print(seed, "%s/50" % saved.get("completed_epochs", 0),
              saved.get("best_checkpoint", "-"), saved.get("complete", False),
              (run / "evaluation_complete.json").exists())
        history = run / "validation_history.jsonl"
        if history.exists():
            for line in history.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    rows.append((seed, record))
    if epochs:
        print("Seed Ep    mAUC      mAP     AP@.5     AP@1    AP@1.5  pAUCstd  mTTAidx")
        for seed, record in rows:
            metrics = record.get("metrics", {})
            keys = ("mAUC@", "mAP@", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC@0.1", "mtta@0.1")
            print("%4d %2d " % (seed, record["epoch"]) +
                  " ".join("%8.4f" % float(metrics.get(k, float("nan"))) for k in keys))
        print("Saved validation rows:", len(rows), "/ expected", 50 * len(seeds))
    print("Read-only live snapshot; a completion marker alone is not a fresh integrity audit.")
    return rows


def export_history(output, seeds):
    rows = status(output, seeds, epochs=True)
    fields = sorted({k for _, r in rows for k in r.get("metrics", {})})
    path = output / "all_validation_epochs.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "history", "seed", "epoch"] + fields)
        for seed, row in rows:
            writer.writerow([VARIANT, scientific_profile()["history_tokens"], seed, row["epoch"]] +
                            [row.get("metrics", {}).get(k, "") for k in fields])
    temporary.replace(path)
    print("Saved all validation epochs:", path, flush=True)


def backup(workspace, output):
    if not output.is_dir():
        raise RuntimeError("No output directory to back up.")
    # The outer operations lock is already held. Reject a direct legacy worker.
    with exclusive_job(output / "job.lock"):
        destination = workspace / "rq2_medium_backups"
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        final = destination / ("RQ2_%s_%s_UTC_%d.tar" % (VARIANT, stamp, os.getpid()))
        partial = final.with_suffix(".tar.partial")
        entries = []
        for source, prefix in ((output, "results"), (HERE, "code")):
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                if ("__pycache__" in relative.parts or path.suffix.lower() in (".pyc", ".tmp", ".partial", ".mp4")
                        or path.name in ("job.lock", ".operations.lock")):
                    continue
                if path.is_symlink():
                    raise RuntimeError("Linked backup member refused: " + str(path))
                entries.append((path, prefix + "/" + relative.as_posix()))
        required = sum(path.stat().st_size for path, _ in entries)
        if shutil.disk_usage(destination).free < required + (1 << 30):
            raise RuntimeError("Not enough disk for backup. Existing output is untouched.")
        manifest = {"kind": "RQ2_MEDIUM_RESULTS_AND_CODE", "variant": VARIANT,
                    "videos_included": False, "files": {name: digest(path) for path, name in entries}}
        with tarfile.open(partial, "w") as archive:
            raw = json.dumps(manifest, indent=2).encode("utf-8")
            item = tarfile.TarInfo("BACKUP_MANIFEST.json")
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
            for path, name in entries:
                archive.add(path, arcname=name, recursive=False)
        with tarfile.open(partial, "r") as archive:
            for member in archive:
                if not member.isfile() or member.name == "BACKUP_MANIFEST.json":
                    continue
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4 << 20), b""):
                        h.update(block)
                if h.hexdigest() != manifest["files"][member.name]:
                    raise RuntimeError("Backup checksum mismatch: " + member.name)
        partial.replace(final)
        sha = digest(final)
        final.with_suffix(".tar.sha256").write_text(sha + "  " + final.name + "\n", encoding="ascii")
        print("VERIFIED BACKUP:", final, "\nSHA256:", sha, flush=True)
        print("Verify downloaded backup locally before destroying the instance. Billing is still active.", flush=True)
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "prepare", "preflight", "train", "evaluate",
                                         "status", "epochs", "backup", "verify", "_download"))
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE", "/workspace"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--runtime-root")
    parser.add_argument("--hours", type=float, default=float(os.environ.get("HOURS", "72")))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Select distinct seeds.")
    workspace = Path(args.workspace).resolve()
    data = Path(args.data_root).resolve() if args.data_root else workspace / "Nexar_HF"
    output = Path(args.output_root).resolve() if args.output_root else workspace / ("rq2_" + VARIANT)
    runtime = Path(args.runtime_root).resolve() if args.runtime_root else workspace / ("rq2_runtime_" + VARIANT)
    if args.stage == "_download":
        download(data)
        return 0
    verify_package()
    if args.stage == "verify":
        print("PACKAGE VERIFIED:", VARIANT, "H=", scientific_profile()["history_tokens"])
        return 0
    if args.stage in ("status", "epochs"):
        status(output, args.seeds, epochs=args.stage == "epochs")
        return 0
    safe_paths(workspace, data, output, runtime)
    workspace.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_job(output / ".operations.lock"):
        if args.stage == "backup":
            export_history(output, args.seeds)
            backup(workspace, output)
            return 0
        if not math.isfinite(args.hours) or args.hours <= 0:
            raise ValueError("--hours must be finite and positive.")
        deadline = time.time() + args.hours * 3600.
        ensure_system()
        try:
            if args.stage in ("all", "prepare", "preflight", "train"):
                prepare(workspace, data, output, deadline, args.skip_download)
            if args.stage == "prepare":
                return 0
            if args.stage == "preflight":
                queue(data, output, runtime, [args.seeds[0]], deadline, "preflight")
                return 0
            if args.stage in ("all", "train"):
                queue(data, output, runtime, args.seeds, deadline, "train")
                if not all_training_complete(output, args.seeds):
                    raise RuntimeError("PARTIAL training; rerun SAME command to resume. Later stages remain locked.")
                export_history(output, args.seeds)
            if args.stage in ("all", "evaluate"):
                queue(data, output, runtime, args.seeds, deadline, "evaluate")
                if not all_evaluated(output, args.seeds):
                    raise RuntimeError("Evaluation incomplete; rerun SAME command to resume curves.")
                export_history(output, args.seeds)
            backup(workspace, output)
            print("COMPLETE:", VARIANT, "| instance is NOT stopped/destroyed automatically.", flush=True)
            return 0
        except BaseException:
            print("STOPPED: outputs preserved. No automatic recipe change. Instance is still billed.", flush=True)
            print("Use status/backup; rerun SAME all/train command for strict resume.", flush=True)
            raise


if __name__ == "__main__":
    sys.exit(main())
