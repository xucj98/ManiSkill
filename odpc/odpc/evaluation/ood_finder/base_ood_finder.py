import torch
from torch.utils.data import Dataset

from odpc.models.policy import BasePolicy


class BaseOODFinder:
    def __init__(
            self, 
            model: BasePolicy,
            dataset: Dataset,
            patience: int = 2,
            **kwargs,
    ):
        self.model = model
        self.dataset = dataset
        self.patience = patience

    def find_ood_samples(
            self, 
            observations: dict,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据输入的observations和actions，返回一组bool值，表示每个样本是否是OOD的。

        Args:
            observations: dict, (B, obs_horizon, *), 输入的observations
            actions: torch.Tensor, (B, pred_horizon, *), 输入的actions
        
        Returns:
            torch.Tensor, (B,), 每个样本是否是OOD的bool值。
        """
        pass

    def find_ood_samples_in_trajectory(
            self,
            observations: dict,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据输入的observations和actions，返回一组bool值，表示每个样本是否是OOD的。
        输入的observations和actions按顺序组成一条trajectory的，
        为了防止异常值，需要连续多个samples是OOD才判定为OOD。

        Args:
            observations: dict, (B, obs_horizon, *), 输入的observations
            actions: torch.Tensor, (B, pred_horizon, *), 输入的actions
        """
        ood_samples = self.find_ood_samples(observations, actions)
        ood_samples_seq = ood_samples.clone()

        # 连续多个samples是OOD才判定为OOD
        patience = self.patience - 1
        ood_samples_seq[:patience] = False
        for i in range(patience, len(ood_samples)):
            if ood_samples[i - patience: i + 1].all():
                ood_samples_seq[i] = True

        return ood_samples_seq
