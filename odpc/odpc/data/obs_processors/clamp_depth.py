import torch
import numpy as np
from typing import Union

from odpc.data.obs_processors.base_processor import BaseObsProcessor


class ClampDepthProcessor(BaseObsProcessor):
    def __init__(
            self, 
            depth_threshold: float = 3000,
            clamp_to_zero: bool = True,
    ):
        super().__init__()
        self.depth_threshold = depth_threshold
        self.clamp_to_zero = clamp_to_zero

    def _clamp(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        clamp_max = self.depth_threshold if not self.clamp_to_zero else 0
        data[data < 0] = 0
        data[data > self.depth_threshold] = clamp_max
        return data

    def process(self, obs: dict) -> dict:
        for sensor in obs["sensor_data"].values():
            for modality in sensor.keys():
                if modality == "depth":
                    self._clamp(sensor[modality])
        return obs