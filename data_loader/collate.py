# data_loader/collate.py
import numbers

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def _ok_type(x):
    return isinstance(x, (torch.Tensor, np.ndarray, numbers.Number))


def collate_jointdit(batch):
    """
    Batch is a list of dicts. Keep only keys whose values are present in ALL
    examples and are collatable (tensor/ndarray/number). Drop None/strings/etc.
    """
    if not batch:
        return {}

    keys = set(batch[0].keys())
    for ex in batch:
        keys &= {k for k, v in ex.items() if (v is not None and _ok_type(v))}

    out = {}
    for k in sorted(keys):
        out[k] = default_collate([ex[k] for ex in batch])
    return out
