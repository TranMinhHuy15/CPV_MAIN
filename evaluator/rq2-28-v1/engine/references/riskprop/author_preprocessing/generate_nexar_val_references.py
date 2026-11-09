import csv
import os


def main() -> None:
    data_dir = os.path.dirname(__file__)
    ann_csv = os.path.join(data_dir, "nexar_val_annotations.csv")
    out_csv = os.path.join(data_dir, "nexar_val_references.csv")

    with open(ann_csv, "r", newline="") as f_in, open(out_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["video_id", "frame_id", "have_accident"])
        writer.writeheader()

        for row in reader:
            video_id = str(row["video_id"]).zfill(5)
            total_frames = int(float(row["total_frames"]))
            accident_frame = int(float(row["accident_occur_frame"]))
            have_accident = int(float(row["have_accident"]))

            # 生成20帧，间隔为3
            if have_accident == 1:
                end = accident_frame
                start = end - 57
            else:
                end = total_frames - 1
                start = end - 57

            frames = list(range(start, end + 1, 3))
            # 防御式：确保恰好20帧
            if len(frames) != 20:
                # 若因为边界导致不足或超出，做裁剪/补齐（理论上按给定规则应当正好20帧）
                frames = frames[:20]
                if len(frames) < 20:
                    # 向后补齐（不应发生），用最后一帧重复
                    frames += [frames[-1]] * (20 - len(frames))

            for fid in frames:
                writer.writerow({
                    "video_id": video_id.zfill(6) if len(video_id) == 6 else video_id.zfill(5),
                    "frame_id": fid,
                    "have_accident": 1 if have_accident == 1 else 0,
                })


if __name__ == "__main__":
    main()


