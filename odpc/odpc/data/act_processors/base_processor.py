import torch
import numpy as np
from typing import Optional

from mani_skill.utils import common

class BaseActProcessor:
    def __init__(self):
        pass

    def process(self, act: dict) -> dict:
        return act

    def to_tensor(self, obs: dict, device: Optional[torch.device] = None) -> dict:
        return common.to_tensor(obs, device)
    
    def to_numpy(self, obs:dict, dtype: Optional[np.dtype] = None) -> dict:
        return common.to_numpy(obs, dtype)
    