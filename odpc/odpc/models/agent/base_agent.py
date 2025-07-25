import os
from typing import List
from omegaconf import ListConfig


import torch
import torch.nn as nn

from odpc.models.policy.base_policy import BasePolicy
from odpc.data.obs_processors import BaseObsProcessor
from odpc.utils.utils import instantiate_from_config


class BaseAgent(nn.Module):
    def __init__(self,
            model: BasePolicy,
            num_envs: int,
            act_horizon: int,
            pred_horizon: int,
            video_dir: str = None,
            obs_processor_configs: ListConfig = [],
    ):
        super().__init__()
        self.model = model
        self.num_envs = num_envs
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.video_dir = video_dir

        self.obs_processors: List[BaseObsProcessor] = []
        for obs_processor_config in obs_processor_configs:
            self.obs_processors.append(instantiate_from_config(obs_processor_config))

        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)

        self._action_step = 0

    @torch.no_grad()
    def get_action(self, obs: dict, channel_last: bool = False) -> torch.Tensor:
        self._action_step += 1
        
    def reset(self, obs: dict, channel_last: bool = False):
        self._action_step = 0
        
    def close(self):
        pass

    @staticmethod
    def permute_obs(obs: dict, dims: List[int]) -> dict:
        if "sensor_data" in obs:
            for sensor_data in obs["sensor_data"].values():
                for modality in sensor_data:
                    sensor_data[modality] = sensor_data[modality].permute(*dims)
        return obs
