from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from odpc.models.policy import BasePolicy
from torch.utils.data import Dataset


class BaseKeyFrameFinder(ABC):
    def __init__(self, model: BasePolicy, train_dataset: Dataset, seed: int = 42):
        self.model = model
        self.train_dataset = train_dataset
        self.seed = seed

    @abstractmethod
    def find_key_frames_from_trajectory(self, trajectory: dict) -> dict:
        """
        从单条轨迹中分析并发现关键帧。

        Args:
            trajectory (dict): 一个包含单条轨迹所有数据的字典。

        Returns:
            A dictionary containing the analysis results:
            {
                'is_key_frame': np.ndarray, # 和trajectory长度相同的数组，表示每一帧是否是关键帧
                'metric_values': np.ndarray # 和trajectory长度相同的数组，表示每一帧的的量化指标
            }
        """
        pass
