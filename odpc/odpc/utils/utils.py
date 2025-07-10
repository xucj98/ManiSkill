import os
import h5py
import torch
import numpy as np
import importlib
from typing import Union
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


def parse_config_expr(expr: str):
    """
    解析配置表达式：
    - 时间表达式: "[now:'%Y%m%d_%H%M%S']"
    """    
    while expr.find("[") != -1 and expr.find("]") != -1:
        start = expr.find("[")
        end = expr.find("]")
        subexpr = expr[start + 1 : end]
        if subexpr.startswith("now:"):
            subexpr = datetime.now().strftime(subexpr[4:])
        else:
            raise ValueError(f"Invalid expression: {expr}")
        expr = expr[:start] + subexpr + expr[end+1:]
    return expr


def load_config_with_defaults(config_path: str):
    raw_config = OmegaConf.load(config_path)
    
    if "defaults" not in raw_config:
        return raw_config

    merged_config = OmegaConf.create({})
    dirname = os.path.dirname(config_path)
    defaults = raw_config.pop("defaults")

    for default in defaults:
        if isinstance(default, str):
            if default == "_self_":
                merged_config = OmegaConf.merge(merged_config, raw_config)
            else:
                cfg = load_config_with_defaults(os.path.join(dirname, default + ".yaml"))
                merged_config = OmegaConf.merge(merged_config, cfg)

        elif isinstance(default, DictConfig):
            key, value = default.popitem()
            cur_dir = dirname 
            while not os.path.exists(os.path.join(cur_dir, key)) and cur_dir != "/":
                cur_dir = os.path.dirname(cur_dir)
            if cur_dir == "/":
                raise ValueError(f"Default config not found: {default}")
            if not os.path.exists(os.path.join(cur_dir, key, value + ".yaml")):
                raise ValueError(f"Default config not found: {default}")
            cfg = OmegaConf.create({})
            cfg[key] = load_config_with_defaults(os.path.join(cur_dir, key, value + ".yaml"))
            merged_config = OmegaConf.merge(merged_config, cfg)

        else:
            raise ValueError(f"Invalid default config: {default}")

    return merged_config


def instantiate_from_config(config: DictConfig, **override_kwargs):
    assert "_target_" in config
    module, cls = config["_target_"].rsplit(".", 1)
    params = dict()
    for k, v in config.items():
        if k != "_target_":
            params[k] = v
    params.update(override_kwargs)

    return getattr(importlib.import_module(module), cls)(**params)

def get_data_shape(
        data: Union[torch.Tensor, np.ndarray, h5py.Dataset, dict, h5py.Group], 
        min_max: bool = False, 
        dtype: bool = False
) -> Union[tuple, dict]:
    if isinstance(data, torch.Tensor):
        res = (data.shape, )
        if dtype:
            res += (data.dtype, )
        if min_max:
            res += (f"{torch.min(data).item():.4f}, {torch.max(data).item():.4f}", )
        return res
    elif isinstance(data, (np.ndarray, h5py.Dataset)):
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


def expand_dim_to(
        a: Union[np.ndarray, torch.Tensor], dim: int, length: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    将 a 的维度 dim 扩展到指定长度 length。
    通过将维度 dim 的最后一个切片复制。
    """
    if a.shape[dim] > length:
        return a
    
    if isinstance(a, np.ndarray):
        expand = np.repeat(np.take(a, indices=[-1], axis=dim), length - a.shape[dim], axis=dim)
        a = np.concatenate((a, expand), axis=dim)
    elif isinstance(a, torch.Tensor):
        expand = torch.repeat_interleave(torch.narrow(a, dim, -1, 1), length - a.shape[dim], dim)
        a = torch.cat((a, expand), dim=dim)
    else:
        raise ValueError(f"Unsupported type: {type(a)}")

    return a

def get_nested_value(data_structure: Union[dict, list], key_path: str):
    """
    从嵌套的字典或列表中根据点分隔的路径字符串获取值。
    如果路径无效或键/索引不存在，则会引发相应的错误。

    Args:
        data_structure (dict or list): 要从中获取数据的字典或列表。
        key_path (str): 点分隔的键路径，例如 "sensor_data.1.rgb.shape"。
                        如果 key_path 为空字符串，则返回原始 data_structure。

    Returns:
        any: 找到的值。
    """
    if not key_path: # 如果 key_path 为空，返回原始数据结构
        return data_structure

    keys = key_path.split('.')
    current_data = data_structure

    for key_part in keys:
        if isinstance(current_data, dict):
            current_data = current_data[key_part]
        elif isinstance(current_data, (list, tuple)):
            index = int(key_part)
            current_data = current_data[index]
        else:
            raise TypeError(
                f"Cannot access key '{key_part}' on element of type '{type(current_data).__name__}' found in path '{key_path}'."
            )
            
    return current_data
