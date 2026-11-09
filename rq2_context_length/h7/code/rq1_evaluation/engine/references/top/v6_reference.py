# Copyright (c) OpenMMLab. All rights reserved.
"""Vendored V6 equations/sampler, not TOP's native AUC implementation.
Source: xingyueye5/RiskProp @ 579376fa1d879f28a9d68c2565d1e449de24b666.
Method bodies unchanged; sampler class and calculate function renamed only.
"""
import numpy as np
from collections import OrderedDict
from typing import Dict
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class V6SampleFramesBeforeAccident(BaseTransform):
    """Sample frames for anticipation task.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self, clip_len: int=1, num_clips: int=50, test_mode: bool=False, **kwargs) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results: dict) -> dict:
        """Perform the SampleFramesBeforeAccident loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        assert results['fps'] in [30, 20, 10]
        frame_interval = results['fps'] // 10
        accident_frame = results['accident_frame']
        start_index = results['start_index']
        clip_inds = np.concatenate(np.arange(self.num_clips)[:, None] + np.arange(self.clip_len)[None, :])
        clip_inds *= frame_interval
        clip_inds_max = clip_inds[-1]
        if self.test_mode:
            if results['target'] is True:
                clip_inds += accident_frame - start_index - clip_inds_max
            elif results['target'] is False:
                if accident_frame is None:
                    if total_frames > clip_inds_max:
                        clip_inds += 0
                    else:
                        clip_inds += total_frames - 1 - clip_inds_max
                elif accident_frame - start_index - frame_interval * 30 > clip_inds_max:
                    clip_inds += 0
                else:
                    clip_inds += accident_frame - start_index - frame_interval * 30 - clip_inds_max
            elif results['target'] is None:
                clip_inds += total_frames - 1 - clip_inds_max
        elif results['target'] is True:
            accident_ind = accident_frame - start_index + np.random.randint(0, frame_interval)
            clip_inds += min(accident_ind, total_frames - 1) - clip_inds_max
        elif results['target'] is False:
            if accident_frame is None:
                if total_frames > clip_inds_max:
                    clip_inds += np.random.randint(0, total_frames - clip_inds_max)
                else:
                    clip_inds += total_frames - 1 - clip_inds_max
            elif accident_frame - start_index - frame_interval * 30 > clip_inds_max:
                clip_inds += np.random.randint(0, accident_frame - start_index - frame_interval * 30 - clip_inds_max)
            else:
                clip_inds += accident_frame - start_index - frame_interval * 30 - clip_inds_max
        frame_inds = np.maximum(clip_inds, 0) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['frame_interval'] = frame_interval
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results

def v6_calculate(self, preds, labels, abnormal_start_inds, accident_inds, sep) -> Dict:
    """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
    eval_results = OrderedDict()
    if len(preds[0].shape) > 1:
        preds = [np.max(pred, axis=-1) for pred in preds]
    preds_n = -np.sort(-np.concatenate([pred for pred, label in zip(preds, labels) if not label]))
    eval_results[f'threshold{sep}{self.fpr_max:.2f}'] = preds_n[int(len(preds_n) * self.fpr_max)]
    ttas = [(j - i - np.argmax(pred[i:j + 1, None] >= preds_n, axis=0)) / 10 * np.any(pred[i:j + 1, None] >= preds_n, axis=0) for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds) if label]
    ttas = np.array(ttas).mean(axis=0)
    eval_results[f'tta{sep}0.01'] = ttas[int(len(ttas) * 0.01)]
    eval_results[f'tta{sep}0.05'] = ttas[int(len(ttas) * 0.05)]
    eval_results[f'tta{sep}0.1'] = ttas[int(len(ttas) * self.fpr_max)]
    eval_results[f'mtta{sep}0.1'] = ttas[:int(len(ttas) * self.fpr_max)].mean()
    eval_results[f'tta{sep}1'] = ttas[int(len(ttas) * 0.99)]
    preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
    preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label]
    preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label]
    preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label]
    preds_n = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if not label]
    y = [1] * len(preds_0) + [0] * len(preds_n)
    eval_results[f'AUC{sep}0.0s'], fpr_0, tpr_0 = auc(np.array(y), np.array(preds_0 + preds_n), self.fpr_max)
    eval_results[f'AUC{sep}0.5s'], fpr_5, tpr_5 = auc(np.array(y), np.array(preds_5 + preds_n), self.fpr_max)
    eval_results[f'AUC{sep}1.0s'], fpr_10, tpr_10 = auc(np.array(y), np.array(preds_10 + preds_n), self.fpr_max)
    eval_results[f'AUC{sep}1.5s'], fpr_15, tpr_15 = auc(np.array(y), np.array(preds_15 + preds_n), self.fpr_max)
    eval_results[f'mAUC{sep}0.1'] = (eval_results[f'AUC{sep}0.5s'] + eval_results[f'AUC{sep}1.0s'] + eval_results[f'AUC{sep}1.5s']) / 3
    eval_results[f'AUC_full{sep}0.0s'], fpr_0, tpr_0 = auc(np.array(y), np.array(preds_0 + preds_n), 1)
    eval_results[f'AUC_full{sep}0.5s'], fpr_5, tpr_5 = auc(np.array(y), np.array(preds_5 + preds_n), 1)
    eval_results[f'AUC_full{sep}1.0s'], fpr_10, tpr_10 = auc(np.array(y), np.array(preds_10 + preds_n), 1)
    eval_results[f'AUC_full{sep}1.5s'], fpr_15, tpr_15 = auc(np.array(y), np.array(preds_15 + preds_n), 1)
    eval_results[f'mAUC{sep}'] = (eval_results[f'AUC_full{sep}0.5s'] + eval_results[f'AUC_full{sep}1.0s'] + eval_results[f'AUC_full{sep}1.5s']) / 3
    eval_results[f'AP{sep}0.0s'] = average_precision_score(np.array(y), np.array(preds_0 + preds_n))
    eval_results[f'AP{sep}0.5s'] = average_precision_score(np.array(y), np.array(preds_5 + preds_n))
    eval_results[f'AP{sep}1.0s'] = average_precision_score(np.array(y), np.array(preds_10 + preds_n))
    eval_results[f'AP{sep}1.5s'] = average_precision_score(np.array(y), np.array(preds_15 + preds_n))
    eval_results[f'mAP{sep}'] = (eval_results[f'AP{sep}0.5s'] + eval_results[f'AP{sep}1.0s'] + eval_results[f'AP{sep}1.5s']) / 3
    eval_results[f'num_samples{sep}'] = len(preds)
    return eval_results

def auc(y_true, y_scores, fpr_max=0.1):
    score = roc_auc_score(y_true, y_scores, max_fpr=fpr_max)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return (score, fpr, tpr)
