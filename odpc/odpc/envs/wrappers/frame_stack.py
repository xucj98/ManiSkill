"""Wrapper that stacks frames. Adapted from gymnasium package to support GPU vectorizated environments."""
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv


class FrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'PickCube-v1', the original observation
    is an array with shape [42], so if we stack 4 observations, the processed observation
    has shape [4, 42].

    This wrapper also supports dict observations, and will stack the leafs of the dictionary accordingly.

    Note:
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.
    """

    def __init__(self, env: gym.Env, num_stack: int):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
        """
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        for _ in range(self.num_stack):
            self.frames.append(self.base_env._init_raw_obs)
        new_obs = self.observation(self.frames)
        self.base_env.update_obs_space(new_obs)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, frames):
        assert len(frames) == self.num_stack, (len(frames), self.num_stack)
        if isinstance(frames[0], dict):
            return {
                k: self.observation([x[k] for x in frames]) for k in frames[0].keys()
            }
        else:
            return torch.stack(list(frames), dim=1)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(self.frames), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment with kwargs.

        Args:
            seed: The seed for the environment reset
            options: The options for the environment reset

        Returns:
            The stacked observations
        """
        if (
            isinstance(options, dict)
            and "env_idx" in options
            and len(options["env_idx"]) < self.base_env.num_envs
        ):
            raise RuntimeError(
                "partial environment reset is currently not supported for the FrameStack wrapper at this moment for GPU parallelized environments"
            )
        # NOTE (stao): we can support partial reset easily using tensordicts
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self.observation(self.frames), info
