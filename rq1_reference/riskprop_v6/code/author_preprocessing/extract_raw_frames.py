import os
import cv2
import pandas as pd
from tqdm import tqdm


def extract_frames(video_path, output_folder, extract_frames=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if extract_frames:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(output_folder, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

    cap.release()
    return frame_count, fps


def process_videos(video_folder):
    video_info = dict()
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]

    with tqdm(video_files, desc="Processing Videos", unit="video") as video_bar:
        for filename in video_bar:
            if filename.lower().endswith(".mp4"):
                video_path = os.path.join(video_folder, filename)
                video_name = os.path.splitext(filename)[0]
                output_folder = os.path.join(video_folder + "_raw_frames", video_name)

                total_frames, fps = extract_frames(video_path, output_folder)
                video_info[video_name] = dict(total_frames=total_frames, fps=fps)
    return video_info


if __name__ == "__main__":
    train_video_folder = "train"
    test_video_folder = "test"
    train_csv = "train.csv"
    test_csv = "test.csv"
    output_csv = "annotations.csv"

    train_info = process_videos(train_video_folder)
    test_info = process_videos(test_video_folder)

    data = dict(id=[], is_test=[], total_frames=[], fps=[], abnormal_start_frame=[], accident_frame=[], target=[])

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    for index, row in train_df.iterrows():
        id = str(int(row["id"])).zfill(5)
        if id not in train_info:
            print(f"Video {id} not found in train folder")
            continue
        data["id"].append(id)
        data["is_test"].append(0)
        data["total_frames"].append(train_info[id]["total_frames"])
        data["fps"].append(train_info[id]["fps"])
        if row["target"] == 1:
            data["abnormal_start_frame"].append(int(row["time_of_alert"] * train_info[id]["fps"]))
            data["accident_frame"].append(int(row["time_of_event"] * train_info[id]["fps"]))
        else:
            data["abnormal_start_frame"].append(-1)
            data["accident_frame"].append(-1)
        data["target"].append(int(row["target"]))

    for index, row in test_df.iterrows():
        id = str(int(row["id"])).zfill(5)
        if id not in test_info:
            print(f"Video {id} not found in test folder")
            continue
        data["id"].append(id)
        data["is_test"].append(1)
        data["total_frames"].append(test_info[id]["total_frames"])
        data["fps"].append(test_info[id]["fps"])
        data["abnormal_start_frame"].append(-1)
        data["accident_frame"].append(-1)
        data["target"].append(-1)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print("Annotations saved to", output_csv)
