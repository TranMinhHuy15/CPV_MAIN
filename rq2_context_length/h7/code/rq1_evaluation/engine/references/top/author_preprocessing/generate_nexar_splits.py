import csv
import os
import sys
import importlib.util


def main() -> None:
    # Resolve project root and load nexar_val from splits.py without importing the whole package
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_path = os.path.join(project_root, "accident_anticipation", "splits.py")
    spec = importlib.util.spec_from_file_location("_aa_splits", splits_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load splits.py module spec")
    splits_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(splits_module)
    nexar_val = getattr(splits_module, "nexar_val")

    data_dir = os.path.dirname(__file__)
    src_csv = os.path.join(data_dir, "annotations.csv")

    out_train = os.path.join(data_dir, "nexar_train_annotations.csv")
    out_val = os.path.join(data_dir, "nexar_val_annotations.csv")
    out_test = os.path.join(data_dir, "nexar_test_annotations.csv")

    header_out = [
        "video_id",
        "path",
        "fps",
        "total_frames",
        "abnormal_start_frame",
        "accident_occur_frame",
        "have_accident",
    ]

    # Open output files
    with open(out_train, "w", newline="") as f_train, \
         open(out_val, "w", newline="") as f_val, \
         open(out_test, "w", newline="") as f_test, \
         open(src_csv, "r", newline="") as f_src:

        reader = csv.DictReader(f_src)
        train_writer = csv.DictWriter(f_train, fieldnames=header_out)
        val_writer = csv.DictWriter(f_val, fieldnames=header_out)
        test_writer = csv.DictWriter(f_test, fieldnames=header_out)

        train_writer.writeheader()
        val_writer.writeheader()
        test_writer.writeheader()

        for row in reader:
            # Input columns: id,is_test,total_frames,fps,abnormal_start_frame,accident_frame,target
            raw_id = row.get("id", "").strip()
            if not raw_id:
                continue
            video_id = raw_id.zfill(5)

            is_test = int(row.get("is_test", 0))
            total_frames = row.get("total_frames", "")
            fps = row.get("fps", "")
            abnormal_start_frame = row.get("abnormal_start_frame", "")
            accident_occur_frame = row.get("accident_frame", "")
            have_accident = row.get("target", "")

            # Build path field: use different root for test vs train/val
            if is_test == 1:
                path = f"test_raw_frames/{video_id}"
            else:
                path = f"train_raw_frames/{video_id}"

            out_row = {
                "video_id": video_id,
                "path": path,
                "fps": fps,
                "total_frames": total_frames,
                "abnormal_start_frame": abnormal_start_frame,
                "accident_occur_frame": accident_occur_frame,
                "have_accident": have_accident,
            }

            if is_test == 1:
                test_writer.writerow(out_row)
            else:
                # decide train or val based on nexar_val dict
                if video_id in nexar_val:
                    val_writer.writerow(out_row)
                else:
                    train_writer.writerow(out_row)


if __name__ == "__main__":
    main()


