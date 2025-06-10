import torch
import numpy as np
from typing import Union

from odpc.data.obs_processors.base_processor import BaseObsProcessor

class SimpleNormalizeProcessor(BaseObsProcessor):
    def __init__(
            self, 
    ):
        super().__init__()


    def process(self, obs: dict) -> dict:
        for sensor in obs["sensor_data"].values():
            for modality in sensor.keys():
                if modality == "depth":
                    sensor[modality] = sensor[modality] / 1024.0
                elif modality == "rgb":
                    sensor[modality] = sensor[modality] / 255.0
        return obs
