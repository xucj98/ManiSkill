import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Union


class BaseNormalizer(nn.Module, ABC):
    """
    Abstract base class for all normalizer modules.
    A normalizer is an nn.Module that learns to normalize and unnormalize data,
    with its statistics being part of the model's state_dict.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        self.input_dim = input_dim

    @torch.no_grad()
    def update_stats(self, data: torch.Tensor):
        """
        Updates the internal statistics of the normalizer based on the incoming data.
        This method is only called when the model is in training mode.
        It should be decorated with @torch.no_grad() to prevent tracking gradients.
        
        Args:
            data (torch.Tensor): A batch of data. Expected shape (B, ..., D).
        """
        # Default implementation does nothing, for normalizers that don't need updates.
        pass

    @abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies normalization to the input tensor using the current statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Applies the inverse normalization to the input tensor."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass handles the logic for training vs. evaluation.
        During training, it updates statistics before normalizing.
        During evaluation, it only normalizes.
        """
        if self.training:
            self.update_stats(x)
        return self.normalize(x)
        
    def get_stats(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Returns the current statistics of the normalizer."""
        return {}


class MinMaxNormalizer(BaseNormalizer):
    """Normalizes data to the range [-1, 1] by learning the min and max values."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.register_buffer('min_val', torch.full((input_dim,), float('inf')))
        self.register_buffer('max_val', torch.full((input_dim,), float('-inf')))

    @torch.no_grad()
    def update_stats(self, data: torch.Tensor):
        if data.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim to be {self.input_dim}, but got {data.shape[-1]}")
        data = data.reshape(-1, self.input_dim)
        if data.shape[0] == 0: return
        current_min = torch.min(data, dim=0).values
        current_max = torch.max(data, dim=0).values
        self.min_val = torch.minimum(self.min_val, current_min)
        self.max_val = torch.maximum(self.max_val, current_max)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        range_ = self.max_val - self.min_val
        safe_range = torch.where(range_ > 1e-8, range_, torch.ones_like(range_))
        return 2 * (x - self.min_val) / safe_range - 1

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        range_ = self.max_val - self.min_val
        safe_range = torch.where(range_ > 1e-8, range_, torch.ones_like(range_))
        return (x_norm + 1) / 2 * safe_range + self.min_val
        
    def get_stats(self) -> Dict[str, torch.Tensor]:
        return {'min': self.min_val.clone(), 'max': self.max_val.clone()}


class GaussianNormalizer(BaseNormalizer):
    """Normalizes data to have zero mean and unit variance (z-score)."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('std', torch.ones(input_dim))
        self.register_buffer('m2', torch.zeros(input_dim))
        self.register_buffer('count', torch.tensor(0, dtype=torch.float32))

    @torch.no_grad()
    def update_stats(self, data: torch.Tensor):
        if data.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim to be {self.input_dim}, but got {data.shape[-1]}")
        data = data.reshape(-1, self.input_dim)
        batch_count = data.shape[0]
        if batch_count == 0: return
        new_count = self.count + batch_count
        delta = data - self.mean.unsqueeze(0)
        self.mean += torch.sum(delta / new_count, dim=0)
        delta2 = data - self.mean.unsqueeze(0)
        self.m2 += torch.sum(delta * delta2, dim=0)
        self.count = new_count
        if self.count > 1:
            self.std = torch.sqrt(self.m2 / (self.count - 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * (self.std + 1e-8) + self.mean
    
    def get_stats(self) -> Dict[str, torch.Tensor]:
        return {'mean': self.mean.clone(), 'std': self.std.clone()}


class NullNormalizer(BaseNormalizer):
    """A pass-through normalizer that performs no operation."""
    # No need to implement update_stats as the base class has a pass
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm