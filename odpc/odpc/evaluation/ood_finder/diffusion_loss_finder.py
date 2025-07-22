
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mani_skill.utils import common

from odpc.models.policy import BasePolicy
from odpc.evaluation.ood_finder.base_ood_finder import BaseOODFinder


class DiffusionLossOODFinder(BaseOODFinder):
    def __init__(
            self, 
            model: BasePolicy, 
            dataset: Dataset, 
            cdf_alpha: float = 0.99,
            patience: int = 2,
            n_timesteps: int = 4,
            batch_size: int = 16,
            num_workers: int = 8,
    ):
        super().__init__(model, dataset, patience)
        self.n_timesteps = n_timesteps
        
        print("========= initialize DiffusionLossOODFinder =========")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        losses = []
        for batch in tqdm(dataloader, desc="Computing diffusion loss threshold"):
            batch = common.to_tensor(batch, device)
            loss = self.model.compute_avg_loss(
                obs=batch["observations"],
                action=batch["actions"],
                n_timesteps=n_timesteps,
            )
            losses += loss.cpu().numpy().tolist()

        losses = np.array(losses)
        threshold = np.percentile(losses, 100 * cdf_alpha)
        self.threshold = threshold
        print(f"Diffusion loss threshold: {threshold}")
        

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
        loss = self.model.compute_avg_loss(
            obs=observations,
            action=actions,
            n_timesteps=self.n_timesteps,
        )
        return loss > self.threshold
        
    def compute_diffusion_loss(
            self,
            observations: dict,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.model.compute_avg_loss(
            obs=observations,
            action=actions,
            n_timesteps=self.n_timesteps,
        )
        return loss
        
        
        