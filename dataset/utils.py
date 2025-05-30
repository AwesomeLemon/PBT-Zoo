import random

import numpy as np
import torch
from torch.utils.data import dataloader

def get_loader(cfg, dataset, seed, **kwargs):
    def worker_init(worker_id):
        np.random.seed(42 + worker_id + seed)
        random.seed(42 + worker_id + seed)
    n_workers = cfg.task.data.num_workers
    return {'train': torch.utils.data.DataLoader(dataset['train'], shuffle=True, num_workers=n_workers, worker_init_fn=worker_init, **kwargs),
            'val': torch.utils.data.DataLoader(dataset['val'], shuffle=False, num_workers=n_workers, worker_init_fn=worker_init, **kwargs),
            'test': torch.utils.data.DataLoader(dataset['test'], shuffle=False, num_workers=n_workers, worker_init_fn=worker_init, **kwargs)
    }