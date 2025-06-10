import h5py
import torch
import numpy as np
import importlib
from omegaconf import DictConfig


def instantiate_from_config(config: DictConfig, **override_kwargs):
    assert "_target_" in config
    module, cls = config["_target_"].rsplit(".", 1)
    params = dict()
    for k, v in config.items():
        if k != "_target_":
            params[k] = v
    params.update(override_kwargs)

    return getattr(importlib.import_module(module), cls)(**params)

def get_data_shape(data, min_max=True, dtype=True):
    if isinstance(data, torch.Tensor):
        res = (data.shape, )
        if dtype:
            res += (data.dtype, )
        if min_max:
            res += (f"{torch.min(data).item():.4f}, {torch.max(data).item():.4f}", )
        return res
    if isinstance(data, (np.ndarray, h5py.Dataset)):
        res = (data.shape, )
        if dtype:
            res += (data.dtype, )
        if min_max:
            res += (f"{np.min(data).item():.4f}, {np.max(data).item():.4f}", )
        return res
    shape = {}
    if isinstance(data, (dict, h5py.Group)):
        for k, v in data.items():
            shape[k] = get_data_shape(v, min_max, dtype)
    return shape