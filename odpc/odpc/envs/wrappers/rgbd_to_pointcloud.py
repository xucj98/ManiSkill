"""Wrapper that stacks frames. Adapted from gymnasium package to support GPU vectorizated environments."""
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from typing import Tuple, List

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

from odpc.data.utils import create_point_cloud, batch_farthest_point_sampling

class RGBDToPointcloud(gym.ObservationWrapper):
    """Observation wrapper that converts RGBD observation to pointcloud observation.

    1. Get XYZRGB from RGBD observation
    2. Crop pointcloud according to the bounding box of ROI
    3. FPS (Farthest Point Sampling)
    """

    def __init__(
            self, 
            env: gym.Env, 
            roi_region: Tuple[float, float, float, float, float, float] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            num_points: int = 1024,
    ):
        """Observation wrapper that converts RGBD observation to pointcloud observation.

        Args:
            env (Env): The environment to apply the wrapper
            roi_region (Tuple[float]): The region of the pointcloud to crop
            num_points (int): The number of points to sample
        """
        gym.ObservationWrapper.__init__(self, env)

        self.roi_region = roi_region
        self.num_points = num_points

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, raw_obs):
        for sensor_name, sensor_data in raw_obs["sensor_data"].items():
            if "rgb" in sensor_data and "depth" in sensor_data:
                rgb = common.to_numpy(sensor_data["rgb"])
                depth = common.to_numpy(sensor_data["depth"])
                intrinsic = common.to_numpy(raw_obs["sensor_param"][sensor_name]["intrinsic_cv"])
                extrinsic_3x4 = common.to_numpy(raw_obs["sensor_param"][sensor_name]["extrinsic_cv"])
                pcs: List[np.ndarray] = []  # (num_points, 6)
                if rgb.ndim == 4:
                    for i in range(rgb.shape[0]):
                        pc = create_point_cloud(
                            rgb[i], depth[i], intrinsic[i], 
                            extrinsic_3x4=extrinsic_3x4[i], 
                            crop_range=self.roi_region
                        )
                        pcs.append(pc)
                elif rgb.ndim == 3:
                    pc = create_point_cloud(
                        rgb, depth, intrinsic, 
                        extrinsic_3x4=extrinsic_3x4, 
                        crop_range=self.roi_region
                    )
                    pcs.append(pc)
                else:
                    raise ValueError(f"RGB image has {rgb.ndim} dimensions, expected 3 or 4")
                pcs = batch_farthest_point_sampling(pcs, self.num_points, return_numpy=False)
                if rgb.ndim == 3:
                    pcs = pcs[0]
                if isinstance(sensor_data["rgb"], torch.Tensor):
                    pcs = common.to_tensor(pcs, device=sensor_data["rgb"].device)
                else:
                    pcs = pcs.cpu().numpy()
                sensor_data["point_cloud"] = pcs
        return raw_obs

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info
