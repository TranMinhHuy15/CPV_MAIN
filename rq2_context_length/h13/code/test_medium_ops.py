"""Operations tests use synthetic bytes and mocks, never HF access or GPU."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import medium_ops as ops
from rq2_profile import VARIANT


class MediumOperationsTests(unittest.TestCase):
    def test_incomplete_training_cannot_start_evaluation_or_training(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ops, "state", return_value=(Path(folder), None)), patch.object(ops, "child") as launch:
            with self.assertRaisesRegex(RuntimeError, "cannot start training"):
                ops.queue(Path("data"), Path(folder), Path("runtime"), [42], time.time()+60, "evaluate")
            launch.assert_not_called()

    def test_training_phase_always_keeps_test_locked(self):
        with patch.object(ops, "child") as launch:
            ops.queue(Path("data"), Path("output"), Path("runtime"), [42,43], time.time()+60, "train")
            command = [str(x) for x in launch.call_args.args[0]]
            self.assertIn("--validation-only", command)
            self.assertIn("--train", command)
            self.assertIn(VARIANT, command)

    def test_evaluation_phase_requires_all_selected_runs_complete(self):
        with patch.object(ops, "state", return_value=(Path("run"), {"complete": True})), patch.object(ops, "child") as launch:
            ops.queue(Path("data"), Path("output"), Path("runtime"), [42,43,44], time.time()+60, "evaluate")
            command = [str(x) for x in launch.call_args.args[0]]
            self.assertNotIn("--validation-only", command)
            self.assertIn("--train", command)

    def test_duplicate_and_unknown_video_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"a").mkdir(); (root/"b").mkdir()
            (root/"a/00001.mp4").write_bytes(b"a")
            (root/"b/00001.mp4").write_bytes(b"a")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                ops.video_paths(root, {"00001": {}})
            with self.assertRaisesRegex(RuntimeError, "Unknown"):
                ops.video_paths(root, {})

    def test_data_hash_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"00001.mp4"; path.write_bytes(b"wrong")
            expected = {"%05d"%k: {"bytes":"5", "sha256":hashlib.sha256(b"right").hexdigest()}
                        for k in range(2844)}
            found = {key:path for key in expected}
            with patch.object(ops, "video_paths", return_value=found):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    ops.verify_data(Path(folder), expected)

    def test_nested_output_and_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "non-nested"):
                ops.safe_paths(root, root/"data", root/"data/output", root/"runtime")

    def test_history_export_contains_all_seeds_and_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for seed in (42,43,44):
                run = output/VARIANT/("seed_%d"%seed)/"run"; run.mkdir(parents=True)
                (run/"training_status.json").write_text(json.dumps({"completed_epochs":50,"complete":True}))
                records = [{"epoch":i,"metrics":{"mAUC@":.8,"mAP@":.7}} for i in range(1,51)]
                (run/"validation_history.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            ops.export_history(output, (42,43,44))
            with (output/"all_validation_epochs.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),150)
            self.assertEqual({(int(r["seed"]),int(r["epoch"])) for r in rows},
                             {(s,e) for s in (42,43,44) for e in range(1,51)})

    def test_backup_verifies_tar_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output=root/"output"; code=root/"code"
            output.mkdir(); code.mkdir()
            (output/"checkpoint.pth").write_bytes(b"fixture-checkpoint-not-a-model")
            prepared=output/"medium/seed_42/prepared_nexar/train"
            prepared.mkdir(parents=True)
            (prepared/"00001.mp4").write_bytes(b"video-must-not-enter-backup")
            (code/"fixture.py").write_text("pass\n")
            with patch.object(ops, "HERE", code):
                path=ops.backup(root,output)
            self.assertTrue((output/"checkpoint.pth").is_file())
            recorded=path.with_suffix(".tar.sha256").read_text().split()[0]
            self.assertEqual(recorded,ops.digest(path))
            with tarfile.open(path) as archive:
                info=json.load(archive.extractfile("BACKUP_MANIFEST.json"))
                self.assertEqual(info["variant"],VARIANT)
                self.assertEqual(set(info["files"]),{"results/checkpoint.pth","code/fixture.py"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
