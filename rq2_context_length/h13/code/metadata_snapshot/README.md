---
license: other
license_name: nexar-open-data-license
license_link: LICENSE
language:
- en
pretty_name: Nexar Collision Prediction Dataset
task_categories:
- video-classification
tags:
- automotive
- dashcam
- collision
- prediction
size_categories:
- 1K<n<10K
extra_gated_heading: >-
  If you access this dataset, you consent to sharing your contact information with
  us. If you do not wish to share such data, do not access this dataset.
extra_gated_prompt: >-
  Access to this dataset is governed by the Nexar Open Data License, included in
  this repository as LICENSE. It permits free use with attribution, and prohibits
  resale, reidentification of people or vehicles, weapons development, and use in
  systems intended to cause harm. Please read it before requesting access.
extra_gated_fields:
  First Name: text
  Last Name: text
  Title: text
  Company Email: text
  Company or institution: text
  Topic of Interest:
    type: select
    options:
    - Research
    - Commercial
    - Education
    - Other
  I have read and agree to the Nexar Open Data License: checkbox
---

# Nexar Collision Prediction Dataset

This dataset is part of the [Nexar Dashcam Crash Prediction Challenge on Kaggle](https://www.kaggle.com/competitions/nexar-collision-prediction/).

## Dataset

The Nexar collision prediction dataset comprises videos from Nexar dashcams. Videos have a resolution of 1280x720 at 30 frames per second and typically have about 40 seconds of duration. The dataset contains 1500 videos where half show events where there was a collision or a collision was eminent (positive cases), and the other half shows regular driving (negative cases). The time of the event (collision or near-miss) is available for positive cases. The dataset is available in the train folder. 


## Goal

The goal of this dataset is to help build models to predict the time of collision. Models should be able to predict if a collision is about to happen or not. Both collisions and near-misses are treated equally as positive cases.


## Model Assessment and Test Set

Models should be able to predict that a collision is about to happen as soon as possible, while minimizing false positives. Assessment scripts will be made available shortly that calculate the mean average precision across different times before collision (e.g. 500ms, 1000ms, 1500ms). 

For this purpose, a test set is provided where videos have about 10 sec and end at 500/1000/1500ms before the event. The `time_to_accident` column tells how much time before the event the video was clipped (this columns is not available in the training set where videos are not clipped). 

The test set is devided into public and private subsets to mirror [Kaggle's competition](https://www.kaggle.com/competitions/nexar-collision-prediction/). During the competition, teams only had access to scores computed on the public subset. At the end of the competition, teams were ranked using the scores on the private subset. 

More details are available in the [paper](https://arxiv.org/abs/2503.03848). 


## Usage

### Loading training data

``` python
from datasets import load_dataset

dataset = load_dataset("videofolder", data_dir="/your/path/nexar_collision_prediction", split="train", drop_labels=False)
```

A positive example would look like this:

``` bash
{'video': <decord.video_reader.VideoReader object at 0x7f5a97c22670>, 'label': 1, 'time_of_event': 20.367, 'time_of_alert': 19.299, 'light_conditions': 'Normal', 'weather': 'Cloudy', 'scene': 'Urban', 'time_to_accident': None}
```

and a negative example like this:

``` bash
{'video': <decord.video_reader.VideoReader object at 0x7ff190129b50>, 'label': 0, 'time_of_event': None, 'time_of_alert': None, 'light_conditions': 'Normal', 'weather': 'Cloudy', 'scene': 'Urban', 'time_to_accident': None}
```

### Running an evaluation

Included is a script that calculates mAP scores for the public and private test sets. The input is a CSV with one line per test video with the video ID and score (see `sample_submission.csv`).

``` bash
$ python evaluate_submission.py sample_submission.csv
mAP (Public): 0.841203
mAP (Private): 0.861791
```

## Paper and Citation

A [paper](https://arxiv.org/abs/2503.03848) is available describing the dataset and the evaluation framework used on the [Nexar Dashcam Crash Prediction Challenge](https://www.kaggle.com/competitions/nexar-collision-prediction/).

Please **use the following reference when citing** this dataset:

>Daniel C. Moura, Shizhan Zhu, and Orly Zvitia . **Nexar Dashcam Collision Prediction Dataset and Challenge**. https://arxiv.org/abs/2503.03848, 2025. 

BibTeX:

```bibtex
@misc{nexar2025dashcamcollisionprediction,
      title={Nexar Dashcam Collision Prediction Dataset and Challenge}, 
      author={Daniel C. Moura and Shizhan Zhu and Orly Zvitia},
      year={2025},
      eprint={2503.03848},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.03848}, 
}
```