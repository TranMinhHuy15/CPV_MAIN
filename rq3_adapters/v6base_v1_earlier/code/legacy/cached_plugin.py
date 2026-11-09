"""Reuse original registered head/loss; replace only the CNN input path during fit."""
from pathlib import Path
import numpy as np
import torch
from cache_store import load, read_record


def cached_extract(model, inputs, stage='neck'):
    if inputs.ndim != 6 or inputs.shape[1:3] != (55, 2048) or inputs.shape[3] < 3:
        raise ValueError('Invalid packed cached input.')
    batch, tokens = inputs.shape[:2]
    x = inputs.reshape(-1, *inputs.shape[2:])
    extra = dict(rq2_batch=batch, rq2_tokens=tokens)
    return (model.cls_head(x, **extra) if stage == 'head' else x), extra


def register():
    from mmaction.registry import MODELS, DATASETS
    from mmaction.structures import ActionDataSample
    from mmengine.model import BaseDataPreprocessor
    from torch.utils.data import Dataset
    if MODELS.get('RQ3CachedRecognizer') is not None:
        return
    BaseRecognizer = MODELS.get('RQ3H7Recognizer')
    if BaseRecognizer is None:
        raise RuntimeError('Register original RQ3 model first.')

    @MODELS.register_module()
    class CachedFeaturePreprocessor(BaseDataPreprocessor):
        def forward(self, data, training=False):
            data = self.cast_data(data)
            if isinstance(data['inputs'], (tuple, list)):
                data['inputs'] = torch.stack(data['inputs'])
            data['inputs'] = data['inputs'].float()
            return data

    @MODELS.register_module()
    class RQ3CachedRecognizer(BaseRecognizer):
        def extract_feat(self, inputs, stage='neck', data_samples=None, test_mode=False):
            return cached_extract(self, inputs, stage)

    @DATASETS.register_module()
    class CachedFeatureDataset(Dataset):
        def __init__(self, cache_root, split, manifest_sha256):
            from cache_store import sha
            self.root = Path(cache_root).resolve()
            if sha(self.root / 'cache_manifest.json') != manifest_sha256:
                raise ValueError('Cache manifest changed after preflight.')
            value = load(self.root / 'cache_manifest.json')
            self.header_sha = value['header_sha256']
            self.records = [r for r in value['records'] if r['split'] == split]
            self.metainfo = {}

        def __len__(self):
            return len(self.records)

        def get_data_info(self, index):
            return dict(self.records[index]['metadata'])

        def __getitem__(self, index):
            row = self.records[index]
            # Recheck per-file hash each read; Input must not mutate during a run.
            features = read_record(self.root, row, self.header_sha)
            meta = dict(row['metadata'])
            meta['frame_inds'] = np.asarray(meta['frame_inds'], dtype=np.int32)
            sample = ActionDataSample()
            sample.set_data(meta)
            return dict(inputs=torch.from_numpy(features), data_samples=sample)
