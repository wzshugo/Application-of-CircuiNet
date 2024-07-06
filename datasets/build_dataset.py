# Copyright 2022 CircuitNet. All rights reserved.

from torch.utils.data import DataLoader, Subset
import datasets
import time
import numpy as np

from .augmentation import Flip, Rotation


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(opt, data_ratio=1.0):
    aug_methods = {'Flip': Flip(), 'Rotation': Rotation(**opt)}
    pipeline=[aug_methods[i] for i in opt.pop('aug_pipeline')] if 'aug_pipeline' in opt and not opt['test_mode'] else None
    dataset = datasets.__dict__[opt.pop('dataset_type')](**opt, pipeline=pipeline)
    
    if data_ratio < 1.0:
        dataset_len = len(dataset)
        subset_len = int(dataset_len * data_ratio)
        indices = np.random.choice(dataset_len, subset_len, replace=False)
        dataset = Subset(dataset, indices)
        
    if opt['test_mode']:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(DataLoader(dataset=dataset, num_workers=16, batch_size=opt.pop('batch_size'), shuffle=True, drop_last=True, pin_memory=True))
