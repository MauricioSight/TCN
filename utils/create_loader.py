import random

import torch
import numpy as np
from torch import device as TorchDevice
from torch.utils.data import DataLoader

def create_loader(data: list[tuple[int, int]], idx: list[int], batch_size: int, device: TorchDevice, g) -> DataLoader:
    def collate_gpu(batch):
        x, t, l = torch.utils.data.dataloader.default_collate(batch)
        return x.to(device=device), t.to(device=device), l
        
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=idx, generator=g, 
                                               worker_init_fn=seed_worker, collate_fn=collate_gpu)
        
    return loader