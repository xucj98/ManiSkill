import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(
            self, 
            obs_horizon,
            pred_horizon,
            output_dim,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.output_dim = output_dim

    def compute_loss(
            self, 
            obs: dict, 
            action: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def get_action(
            self, 
            obs: dict,
    ) -> torch.Tensor:
        pass