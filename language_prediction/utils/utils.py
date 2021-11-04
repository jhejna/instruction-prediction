import torch
import numpy as np

def to_device(batch, device):
    if isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
    else:
        batch = batch.to(device)
    return batch

def to_tensor(batch):
    if isinstance(batch, dict):
        batch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in batch]
    else:
        batch = torch.from_numpy(batch)
    return batch

def unsqueeze(batch, dim):
    if isinstance(batch, dict):
        batch = {k: v.unsqueeze(dim) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [v.unsqueeze(dim) if isinstance(v, torch.Tensor) else v for v in batch]
    else:
        batch = batch.unsqueeze(dim)
    return batch
