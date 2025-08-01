from typing import Dict, List
import numpy as np

from odpc.evaluation.rollout_analysis.key_frames_finder.base import BaseKeyFrameFinder
from odpc.models.policy import BasePolicy
from torch.utils.data import Dataset


class LastFramesFinder(BaseKeyFrameFinder):
    def __init__(
            self, 
            model: BasePolicy, 
            train_dataset: Dataset, 
            num_last_frames: int, 
            **kwargs
    ):
        super().__init__(model, train_dataset, **kwargs)
        self.num_last_frames = num_last_frames

    def find_key_frames_from_trajectory(self, trajectory: dict) -> dict:
        num_frames = trajectory['frame_idx'].shape[0]


        start_index = max(0, num_frames - self.num_last_frames)
        key_indices = list(range(start_index, num_frames))

        metric_values = np.zeros(num_frames)
        metric_values[key_indices] = 1.0
        is_key_frame = metric_values.astype(bool)

        return {
            'is_key_frame': is_key_frame,
            'metric_values': metric_values,
        }
