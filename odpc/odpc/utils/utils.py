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

def get_data_shape(data):
    if isinstance(data, torch.Tensor):
        return data.shape, f"{torch.min(data).item():.4f}, {torch.max(data).item():.4f}"
    if isinstance(data, (np.ndarray, h5py.Dataset)):
        return data.shape, f"{np.min(data).item():.4f}, {np.max(data).item():.4f}"
    shape = {}
    if isinstance(data, (dict, h5py.Group)):
        for k, v in data.items():
            shape[k] = get_data_shape(v)
    return shape