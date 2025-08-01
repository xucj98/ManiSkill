
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mani_skill.utils import common

from odpc.models.policy import BasePolicy
from odpc.evaluation.rollout_analysis.key_frames_finder.base import BaseKeyFrameFinder


class DiffusionLossFinder(BaseKeyFrameFinder):
    def __init__(
            self, 
            model: BasePolicy, 
            train_dataset: Dataset, 
            cdf_alpha: float = 0.99,
            patience: int = 2,
            n_timesteps: int = 4,
            batch_size: int = 16,
            num_workers: int = 8,
    ):
        super().__init__(model, train_dataset)
        self.n_timesteps = n_timesteps
        
        print("========= initialize DiffusionLossOODFinder =========")
        
        dataloader = DataLoader(
            train_dataset,
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
        print(f"Diffusion loss mean: {losses.mean()}")
        print("========= initialize DiffusionLossOODFinder =========")
        
    @torch.no_grad()
    def find_key_frames_from_trajectory(
            self,
            trajectory: dict,
    ) -> dict:
        loss = self.compute_diffusion_loss(
            observations=trajectory["observations"],
            actions=trajectory["actions"],
        )
        is_key_frame = loss > self.threshold
        return {
            "is_key_frame": is_key_frame.cpu().numpy(),
            "metric_values": loss.cpu().numpy(),
        }
        
    @torch.no_grad()
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