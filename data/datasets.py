import random

import numpy
import torch
from torch.utils.data import DataLoader

from config import SEED
from data.nyu_reduced import get_NYU_dataset

torch.manual_seed(SEED)

"""
Preparation of dataloaders for Datasets
"""

def get_dataloader(dataset_name,
                   path,
                   split='train',
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear',
                   batch_size=8,
                   workers=4,
                   uncompressed=False):
    if dataset_name == 'nyu_reduced':
        dataset = get_NYU_dataset(path,
                split,
                resolution=resolution,
                uncompressed=uncompressed)
    else:
        print('Dataset not existant')
        exit(0)

    def seed_worker(worker_id):
        numpy.random.seed(SEED)
        random.seed(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=='train'),
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader
