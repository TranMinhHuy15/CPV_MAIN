import os
import cv2
import shutil
import pandas as pd
from PIL import Image, ImageDraw
import pickle
from skimage.metrics import structural_similarity as ssim


def extract_feature(input_folder):
    features = {}
    for root, dirs, files in os.walk(input_folder):
        if not root.endswith("s"):
            # 获取相对于输入目录的路径
            relative_path = os.path.relpath(root, input_folder)
            image_path = os.path.join(root, "000000.jpg")

            # 处理文件夹中的图像
            video_id = relative_path

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (90, 60), interpolation=cv2.INTER_AREA)
            features[video_id] = image
            print("Processed: ", image_path)
    return features


if __name__ == "__main__":
    input_folder = "data/nexar-collision-prediction/test_raw_frames"  # 输入文件夹路径
    features = extract_feature(input_folder)

    # with open("data.pkl", "wb") as pickle_file:
    #     pickle.dump(features, pickle_file)

    # with open("data.pkl", "rb") as pickle_file:
    #     features = pickle.load(pickle_file)

    for key1 in features.keys():
        for key2 in features.keys():
            if key1 < key2:
                image1 = features[key1]
                image2 = features[key2]
                score = ssim(image1, image2)
                if score > 0.6:
                    with open("score_90.csv", "a") as file:
                        print(key1, key2, score, sep=",", file=file)
