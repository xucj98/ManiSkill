from typing import Dict, List
import numpy as np

from odpc.evaluation.rollout_analysis.key_frames_finder.base import BaseKeyFrameFinder
from odpc.models.policy import BasePolicy
from torch.utils.data import Dataset


class RandomFramesFinder(BaseKeyFrameFinder):
    def __init__(
            self, 
            model: BasePolicy, 
            train_dataset: Dataset, 
            num_random_frames: int = 1,
            **kwargs
    ):
        super().__init__(model, train_dataset, **kwargs)
        self.num_random_frames = num_random_frames
        self.rng = np.random.default_rng(seed=self.seed)

    def find_key_frames_from_trajectory(self, trajectory: dict) -> dict:
        num_frames = trajectory['frame_idx'].shape[0]
        key_indices = self.rng.choice(range(num_frames), min(self.num_random_frames, num_frames))

        metric_values = np.zeros(num_frames)
        metric_values[key_indices] = 1.0
        is_key_frame = metric_values.astype(bool)

        return {
            'is_key_frame': is_key_frame,
            'metric_values': metric_values,
        }
