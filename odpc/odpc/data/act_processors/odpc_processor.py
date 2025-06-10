
import torch
import numpy as np

from omegaconf import DictConfig

from odpc.utils.utils import instantiate_from_config
from odpc.data.act_processors.base_processor import BaseActProcessor
from odpc.data.data_conversion import DataConversion


class ODPCProcessor(BaseActProcessor):
    def __init__(
            self, 
            dc_config: DictConfig,
            output_key="actions",
    ):
        super().__init__()
        self.dc: DataConversion = instantiate_from_config(dc_config)
        self.output_key = output_key


    def process(self, data: dict) -> dict:
        poses_obj = data["peg_pose"]
        poses_camera_world = data["cam0_world_pose"]

        is_numpy = False
        if isinstance(poses_obj, np.ndarray):
            is_numpy = True
            poses_obj = self.to_tensor(poses_obj)
            poses_camera_world = self.to_tensor(poses_camera_world)

        data[self.output_key] = self.dc.raw_to_pred(
            poses_obj=poses_obj,    
            poses_camera_world=poses_camera_world,
        )

        if is_numpy:
            data[self.output_key] = self.to_numpy(data[self.output_key])

        return data
