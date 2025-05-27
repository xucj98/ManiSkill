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
